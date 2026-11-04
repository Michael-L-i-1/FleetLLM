"""Model-free checks for frozen input identity, streaming failure accounting and QA grading."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch

import throughput_workflows as bench


class WorkflowsTest(unittest.TestCase):
    def fixture(self, root):
        source = root/'src'
        source.mkdir()
        (source/'kv.h').write_text('#define KV_PAGE 16\n#define KV_NONE UINT32_MAX\n')
        (source/'ngram.h').write_text('#define NG_N 3 // window length\n')
        (source/'engine.c').write_text('e->row_budget = 128;\nif (!e->spec_k) e->spec_k = 15;\n')
        (source/'kv.c').write_text('static bool palloc(kv_pool *p, uint32_t *out) { return true; }\n')

    def test_frozen_inputs_survive_source_changes_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            with patch.object(bench, 'ROOT', root):
                a = bench.prepare_workload(root/'a.json', agents=2, turns=3, tool_seconds=0)
                b = bench.prepare_workload(root/'b.json', agents=2, turns=3, tool_seconds=0)
            self.assertEqual(a['workload_sha256'], b['workload_sha256'])
            (root/'src/engine.c').write_text('changed after preparation')
            loaded = bench.load_workload(root/'a.json')
            self.assertEqual(a, loaded)
            loaded['sessions'][0]['requests'][0]['messages'][1]['content'] = 'edited'
            bench.save_json(root/'tampered.json', loaded)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                bench.load_workload(root/'tampered.json')

    def server(self, fail=False, quality=False):
        recorded = []
        counts = {}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:
                    recorded.append(body)
                    match = re.search(r'Session (\d+):', body['messages'][1]['content'])
                    name = match.group(1) if match else 'quality'
                    turn = counts.get(name, 0)
                    counts[name] = turn+1
                if fail is True and name == '0' and turn == 1:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b'injected unavailable request')
                    return
                answer, finish = 'ok', 'stop'
                if quality:
                    prompt = body['messages'][-1]['content'].rsplit('\n\n', 1)[-1]
                    answer = next((value for key, value in [('KV_PAGE', '16'), ('KV_NONE', 'UINT32_MAX'),
                                   ('NG_N', '3'), ('row_budget', '128'), ('spec_k', '15'),
                                   ('allocator', 'palloc')] if key in prompt), 'unknown')
                    # A correct-looking answer with a cap is not a passed quality probe.
                    if 'KV_NONE' in prompt:
                        finish = 'length'
                    if quality == 'wrong' and 'NG_N' in prompt:
                        answer = '999'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                chunks = [
                    {'choices': [{'delta': {'reasoning_content': 'Inspect.'}, 'finish_reason': None}]},
                    {'choices': [{'delta': {'content': answer}, 'finish_reason': None}]},
                    {'choices': [{'delta': {}, 'finish_reason': finish}]},
                    {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 4,
                                             'prompt_tokens_details': {'cached_tokens': 10}}},
                ]
                for chunk in chunks:
                    self.wfile.write(b'data: '+json.dumps(chunk).encode()+b'\n\n')
                    self.wfile.flush()
                if fail == 'truncate' and name == '0' and turn == 1:
                    return
                self.wfile.write(b'data: [DONE]\n\n')

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_port}/v1', recorded

    def test_streamed_partial_failure_is_durable_and_cannot_report_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            with patch.object(bench, 'ROOT', root):
                workload = bench.prepare_workload(root/'workload.json', agents=2, turns=2, tool_seconds=0)
            base, recorded = self.server(fail=True)
            runner = bench.Runner(workload, root/'result', base, 'fake', concurrency=2)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = runner.run()
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['expected_requests'], 4)
            self.assertEqual(summary['recorded_requests'], 4)
            self.assertEqual(summary['failed_requests'], 1)
            self.assertEqual(summary['completion_tokens'], 12)
            self.assertEqual(summary['cached_fraction'], .1)
            self.assertEqual(len(recorded), 4)
            self.assertEqual(len(list((root/'result/responses').glob('*.json'))), 4)
            calls = [json.loads(x) for x in (root/'result/calls.jsonl').read_text().splitlines()]
            self.assertEqual(sum(x['error'] is not None for x in calls), 1)
            successes = [x for x in calls if not x['error']]
            self.assertTrue(all(x['ttft_seconds'] is not None for x in successes))
            self.assertTrue(all(x['content'] == 'ok' and x['reasoning'] == 'Inspect.' for x in successes))
            self.assertTrue((root/'result/summary.csv').exists())

    def test_truncated_stream_keeps_partial_text_and_usage_but_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            with patch.object(bench, 'ROOT', root):
                workload = bench.prepare_workload(root/'workload.json', agents=1, turns=2, tool_seconds=0)
            base, _ = self.server(fail='truncate')
            runner = bench.Runner(workload, root/'result', base, 'fake', concurrency=1)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = runner.run()
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['failed_requests'], 1)
            self.assertEqual(summary['completion_tokens'], 4)
            self.assertEqual(summary['partial_error_completion_tokens'], 4)
            partial = json.loads((root/'result/responses/session-000-turn-001.json').read_text())
            self.assertEqual(partial['content'], 'ok')
            self.assertIn('missing DONE', partial['error'])

    def test_graded_multi_turn_qa_preserves_growth_and_grades_every_answer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            with patch.object(bench, 'ROOT', root):
                workload = bench.prepare_workload(root/'qa.json', agents=1, turns=3, tool_seconds=0, task_mode='qa')
            self.assertEqual(workload['settings']['reasoning'], 'none')
            requests = workload['sessions'][0]['requests']
            self.assertEqual([r['expected_answer'] for r in requests], ['16', 'UINT32_MAX', '3'])
            self.assertEqual(requests[1]['messages'][-2], {'role':'assistant','content':'16'})
            self.assertEqual(requests[2]['messages'][-2], {'role':'assistant','content':'UINT32_MAX'})
            self.assertGreater(len(requests[2]['messages']), len(requests[1]['messages']))
            self.assertNotIn('Continue the review', requests[2]['messages'][-1]['content'])
            base, recorded = self.server(quality='wrong')
            runner = bench.Runner(workload, root/'result', base, 'fake', concurrency=1)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = runner.run()
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['graded_total'], 3)
            self.assertEqual(summary['graded_correct'], 1)
            self.assertAlmostEqual(summary['graded_accuracy'], 1/3)
            self.assertFalse(summary['graded_all_correct'])
            self.assertTrue(all(r['reasoning_effort']=='none' for r in recorded))
            self.assertEqual([r['correct'] for r in runner.results], [True, False, False])

    def test_output_cap_clone_preserves_frozen_source_and_all_history(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            with patch.object(bench, 'ROOT', root):
                original = bench.prepare_workload(root/'original.json', agents=2, turns=3, tool_seconds=0, task_mode='qa')
            (root/'src/engine.c').write_text('invalid changed source after freeze')
            with patch.object(bench, 'ROOT', root/'absent-live-repo'):
                cloned = bench.clone_workload(root/'original.json', root/'derived.json', 256)
            bench.load_workload(root/'derived.json')
            self.assertNotEqual(original['workload_sha256'], cloned['workload_sha256'])
            self.assertEqual(cloned['corpus'], original['corpus'])
            self.assertEqual(cloned['quality'], original['quality'])
            for before, after in zip(original['sessions'], cloned['sessions']):
                for a, b in zip(before['requests'], after['requests']):
                    self.assertEqual(b['max_tokens'], 256)
                    self.assertNotEqual(a['request_sha256'], b['request_sha256'])
                    self.assertEqual({k:v for k,v in a.items() if k not in ['max_tokens','request_sha256']},
                                     {k:v for k,v in b.items() if k not in ['max_tokens','request_sha256']})

    def test_quality_scores_answers_and_rejects_length_capped_answers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            with patch.object(bench, 'ROOT', root):
                workload = bench.prepare_workload(root/'workload.json', agents=1, turns=1, tool_seconds=0)
            base, _ = self.server(quality=True)
            runner = bench.Runner(workload, root/'quality', base, 'fake', concurrency=1, mode='quality')
            with contextlib.redirect_stdout(io.StringIO()):
                summary = runner.run()
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['quality_correct'], 5)
            self.assertEqual(summary['quality_total'], 6)
            self.assertFalse(summary['quality_all_correct'])
            capped = next(x for x in runner.results if x['session'] == 'kv-none')
            self.assertEqual(capped['content'], 'UINT32_MAX')
            self.assertFalse(capped['correct'])


if __name__ == '__main__':
    unittest.main()
