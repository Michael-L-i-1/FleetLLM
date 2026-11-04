#!/usr/bin/env python3
"""Prepare and run frozen multi-turn agent-shaped throughput/quality workloads.

The trace uses fixed reference assistant messages, not generated replies, between
turns. It measures serving/cache behavior on identical histories across variants;
no generated action is executed and no coding success is implied. Output is capped,
not fixed work. Actual API token counts and output bytes are preserved.

  python3 bench/throughput_workflows.py prepare --output /tmp/fleet-workload.json
  python3 bench/throughput_workflows.py run --workload /tmp/fleet-workload.json \
    --output /tmp/fleet-control --model qwen35-4b-mlx4 --concurrency 8
  python3 bench/throughput_workflows.py run --mode quality --workload /tmp/fleet-workload.json \
    --output /tmp/fleet-quality --model qwen35-4b-mlx4 --concurrency 4
  python3 bench/throughput_workflows.py prepare --task-mode qa --output /tmp/fleet-graded-qa.json
  python3 bench/throughput_workflows.py clone --input /tmp/fleet-workload.json \
    --output /tmp/fleet-cap256.json --max-tokens 256

QA trace mode grades every response against a correct frozen reference answer;
it still uses fixed prior history, not autonomous model-selected actions. The
standalone quality mode is a separate short-snippet probe. Transport completeness
and answer correctness are reported independently.

Run one server/configuration at a time externally. This program never starts or
stops model servers. Development tests use a fake HTTP server and load no model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fleet-frozen-workflows-v1'


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def quality_items(corpus):
    specs = [
        ('kv-page', 'src/kv.h', r'#define KV_PAGE (\d+)', 'What decimal integer is KV_PAGE?'),
        ('kv-none', 'src/kv.h', r'#define KV_NONE (\w+)', 'What exact identifier is KV_NONE defined as?'),
        ('ng-window', 'src/ngram.h', r'#define NG_N (\d+)', 'What decimal integer is NG_N?'),
        ('row-budget', 'src/engine.c', r'e->row_budget = (\d+);', 'What decimal default is assigned to e->row_budget?'),
        ('ng-drafts', 'src/engine.c', r'if \(!e->spec_k\) e->spec_k = (\d+);',
         'What decimal default is assigned to e->spec_k when it is zero?'),
        ('page-allocator', 'src/kv.c', r'static bool (palloc)\(', 'What is the name of this page allocator function?'),
    ]
    items = []
    for key, filename, pattern, question in specs:
        text = corpus[filename]['text']
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f'Quality fact absent from frozen source: {key}')
        line = text.count('\n', 0, match.start())
        lines = text.splitlines()
        snippet = '\n'.join(lines[max(0, line-2):line+4])
        messages = [
            {'role': 'system', 'content': 'Answer the exact question from the provided frozen source snippet. '
             'Reply with only the requested integer or identifier; no explanation or Markdown.'},
            {'role': 'user', 'content': f'{filename}:\n```c\n{snippet}\n```\n\n{question}'},
        ]
        items.append({'id': key, 'source': filename, 'source_sha256': corpus[filename]['sha256'],
                      'answer': match.group(1), 'messages': messages, 'max_tokens': 64, 'reasoning': 'none'})
    return items


def prepare_workload(output, agents=8, turns=4, shared_chars=12000, tool_chars=4000,
                     tool_seconds=0.5, seed=0, max_tokens=64, reasoning='low', task_mode='review'):
    if agents < 1 or turns < 1 or shared_chars < 0 or tool_chars < 1 or not 0 <= tool_seconds < 2:
        raise ValueError('Positive sessions/turns, nonnegative shared size and 0 <= tool seconds < 2 required')
    if max_tokens < 1:
        raise ValueError('max_tokens must be positive')
    if task_mode not in {'review', 'qa'}:
        raise ValueError('task_mode must be review or qa')
    if task_mode == 'qa':
        reasoning = 'none'
    files = sorted(list((ROOT/'src').glob('*.c')) + list((ROOT/'src').glob('*.h')))
    corpus = {}
    for path in files:
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        corpus[str(path.relative_to(ROOT))] = {'text': text, 'sha256': hashlib.sha256(raw).hexdigest()}
    names = list(corpus)
    shared = ''.join(f'\n### {name}\n{corpus[name]["text"][:2400]}' for name in names)[:shared_chars]
    system = ('You are a coding agent reviewing a C inference engine. Inspect the supplied source and '
              'give one concrete next step and a short correctness risk. Keep the answer concise. '
              'Tool observations and prior assistant messages in this benchmark are fixed reference history.\n'
              'Shared frozen project context:\n' + shared)
    if task_mode == 'qa':
        system = ('You answer factual questions about a frozen C inference-engine repository. '
                  'Use the focused source snippet in the latest user message. Reply with ONLY the '
                  'requested integer or identifier, no explanation or Markdown. Earlier assistant '
                  'answers are fixed correct reference history. Other source context is background.\n'
                  'Shared frozen project context:\n' + shared)
    facts = quality_items(corpus)
    rng = random.Random(seed)
    sessions = []
    for agent in range(agents):
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': f'Session {agent}: review inference state ownership and batching. '
                     'Identify the next useful inspection and its correctness risk.'}]
        if task_mode == 'qa':
            first_fact = facts[agent % len(facts)]
            messages[1]['content'] = f'Session {agent}: answer this source question.\n\n' + first_fact['messages'][1]['content']
        requests = []
        for turn in range(turns):
            fact = facts[(agent+turn) % len(facts)]
            request = {'turn': turn, 'messages': list(messages), 'max_tokens': max_tokens,
                       'reasoning': reasoning, 'delay_before_seconds': 0.0 if turn == 0 else round(rng.uniform(0, min(1.999, 2*tool_seconds)), 6)}
            if task_mode == 'qa':
                request.update(expected_answer=fact['answer'], fact_id=fact['id'])
            request['request_sha256'] = digest(request)
            requests.append(request)
            if turn + 1 < turns:
                filename = names[(agent*3+turn*5) % len(names)]
                body = corpus[filename]['text']
                # A fixed file segment is a real source observation, not model output.
                offset = min((agent % 3)*240, max(0, len(body)-tool_chars))
                observed = body[offset:offset+tool_chars]
                if task_mode == 'qa':
                    following = facts[(agent+turn+1) % len(facts)]
                    assistant = fact['answer']
                    observation = (f'read_file({filename}) returned frozen bytes [{offset}:{offset+len(observed)}]:\n'
                                   f'{observed}\n\nFocused source question:\n' + following['messages'][1]['content'])
                else:
                    assistant = ('Reference action: inspect the next source file before changing state ownership. '
                                 'Keep copy-on-write isolation and causal ordering intact.')
                    observation = (f'read_file({filename}) returned frozen bytes [{offset}:{offset+len(observed)}]:\n'
                                   f'{observed}\nContinue the review with one next step and its risk.')
                messages = messages + [{'role': 'assistant', 'content': assistant},
                                       {'role': 'user', 'content': observation}]
        sessions.append({'id': f'session-{agent:03d}', 'requests': requests})
    workload = {'schema': SCHEMA, 'kind': 'fixed agent-shaped histories; generated replies are not executed or fed back',
                'output_work': 'variable generated output despite identical caps; compare actual token counts',
                'settings': {'task_mode': task_mode, 'agents': agents, 'turns': turns, 'shared_chars': shared_chars, 'tool_chars': tool_chars,
                             'tool_seconds_mean_target': tool_seconds, 'tool_delay_max_seconds': min(1.999, 2*tool_seconds),
                             'seed': seed, 'max_tokens': max_tokens, 'reasoning': reasoning,
                             'context_size_note': 'Character budgets are approximate; actual API prompt tokens are measured.'},
                'corpus': corpus, 'sessions': sessions, 'quality': facts}
    workload['workload_sha256'] = digest(workload)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    save_json(output, workload)
    return workload


def load_workload(path):
    workload = json.loads(Path(path).read_text())
    if workload.get('schema') != SCHEMA:
        raise ValueError('Unsupported workload schema')
    recorded = workload.get('workload_sha256')
    unhashed = {k: v for k, v in workload.items() if k != 'workload_sha256'}
    if recorded != digest(unhashed):
        raise ValueError('Workload hash mismatch; prepare a new workload instead of editing a frozen one')
    for session in workload['sessions']:
        for request in session['requests']:
            if request['request_sha256'] != digest({k: v for k, v in request.items() if k != 'request_sha256'}):
                raise ValueError('Request hash mismatch')
    return workload


def clone_workload(input_path, output_path, max_tokens):
    """Derive a trace-cap sensitivity case without touching the live corpus."""
    if max_tokens < 1:
        raise ValueError('max_tokens must be positive')
    original = load_workload(input_path)
    workload = copy.deepcopy(original)
    workload['settings']['max_tokens'] = max_tokens
    for session in workload['sessions']:
        for request in session['requests']:
            request['max_tokens'] = max_tokens
            request['request_sha256'] = digest({k: v for k, v in request.items() if k != 'request_sha256'})
    workload['derivation'] = {'parent_workload_sha256': original['workload_sha256'],
                              'operation': 'trace output cap only; corpus, messages, delays, facts and quality probes unchanged',
                              'max_tokens': max_tokens}
    workload['workload_sha256'] = digest({k: v for k, v in workload.items() if k != 'workload_sha256'})
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, workload)
    return workload


def percentile(values, quantile):
    values = sorted(v for v in values if v is not None)
    return values[max(0, math.ceil(quantile*len(values))-1)] if values else None


class Runner:
    def __init__(self, workload, output, base, model, concurrency=8, sessions=None, mode='trace', timeout=300,
                 qwen_thinking_template=False):
        self.workload = workload
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/'responses').mkdir()
        self.base, self.model = base.rstrip('/'), model
        self.concurrency, self.mode, self.timeout = concurrency, mode, timeout
        self.qwen_thinking_template = qwen_thinking_template
        if concurrency < 1:
            raise ValueError('Concurrency must be positive')
        self.sessions = workload['sessions'][:sessions] if sessions else workload['sessions']
        if sessions is not None and not 1 <= sessions <= len(workload['sessions']):
            raise ValueError('Session count must fit the prepared workload')
        self.lock = threading.Lock()
        self.results = []
        self.started = time.monotonic()
        self.config = {'workload_sha256': workload['workload_sha256'], 'mode': mode, 'base': self.base,
                       'model': model, 'concurrency': concurrency, 'sessions': len(self.sessions), 'timeout': timeout,
                       'selected_session_ids': [s['id'] for s in self.sessions],
                       'prepared_settings': workload['settings'],
                       'sampling': {'temperature': 0, 'max_tokens': 64 if mode == 'quality' else workload['settings']['max_tokens'],
                                    'reasoning': 'none' if mode == 'quality' else workload['settings']['reasoning']},
                       'history': 'fixed reference messages; generated replies never affect subsequent requests',
                       'output_work': 'capped, not fixed; actual generated token counts are reported'}
        self.config['qwen_thinking_template'] = qwen_thinking_template
        save_json(self.output/'config.json', self.config)
        self.event('start', **self.config)

    def event(self, kind, **fields):
        value = {'event': kind, 'elapsed': time.monotonic()-self.started, **fields}
        with self.lock:
            with (self.output/'events.jsonl').open('a') as f:
                f.write(json.dumps(value, ensure_ascii=False)+'\n')
                f.flush()
        return value

    def request(self, session, turn, item, answer=None):
        if answer is None:
            answer = item.get('expected_answer')
        identifier = f'{session}-turn-{turn:03d}'
        body = {'model': self.model, 'messages': item['messages'], 'temperature': 0,
                'max_tokens': item['max_tokens'], 'reasoning_effort': item['reasoning'],
                'stream': True, 'stream_options': {'include_usage': True}}
        if self.qwen_thinking_template:
            body['chat_template_kwargs'] = {'enable_thinking': item['reasoning'] != 'none'}
        sent_hash = digest(body)
        self.event('request', id=identifier, session=session, turn=turn, request_sha256=sent_hash)
        started = time.monotonic()
        content, reasoning, usage = '', '', {}
        first, finish, error = None, None, None
        saw_done, valid_stream = False, False
        try:
            req = urllib.request.Request(self.base+'/chat/completions', encode(body), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if 'text/event-stream' in response.headers.get('Content-Type', ''):
                    valid_stream = True
                    for line in response:
                        if not line.startswith(b'data:'):
                            continue
                        data = line[5:].strip()
                        if data == b'[DONE]':
                            saw_done = True
                            break
                        chunk = json.loads(data)
                        if chunk.get('error'):
                            raise ValueError(f'Stream error: {chunk["error"]}')
                        if chunk.get('usage'):
                            usage = chunk['usage']
                        for choice in chunk.get('choices', []):
                            delta = choice.get('delta', {})
                            c, r = delta.get('content') or '', delta.get('reasoning_content') or ''
                            if (c or r) and first is None:
                                first = time.monotonic()-started
                            content += c
                            reasoning += r
                            if choice.get('finish_reason') is not None:
                                finish = choice['finish_reason']
                    if not saw_done or finish is None:
                        raise ValueError('Truncated SSE response: missing DONE or finish reason')
                else:
                    reply = json.load(response)
                    if reply.get('error'):
                        raise ValueError(f'API error: {reply["error"]}')
                    choice = reply['choices'][0]
                    content = choice['message'].get('content') or ''
                    reasoning = choice['message'].get('reasoning_content') or ''
                    finish, usage = choice.get('finish_reason'), reply.get('usage') or {}
                    # Nonstreaming cannot supply TTFT; leave it unknown.
                    if finish is None:
                        raise ValueError('Missing finish reason')
            if any(type(usage.get(key)) is not int or usage[key] < 0 for key in ['completion_tokens', 'prompt_tokens']):
                raise ValueError('Missing actual API token usage; refusing to estimate output tokens')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            if isinstance(exc, urllib.error.HTTPError):
                error += ': '+exc.read(2000).decode(errors='replace')
                exc.close()
        elapsed = time.monotonic()-started
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        correct = None if answer is None or error else finish == 'stop' and normalize_answer(content) == normalize_answer(answer)
        result = {'id': identifier, 'session': session, 'turn': turn, 'request_sha256': sent_hash,
                  'frozen_request_sha256': item.get('request_sha256') or digest(item),
                  'seconds': elapsed, 'ttft_seconds': first, 'streaming': valid_stream,
                  'finish_reason': finish, 'error': error, 'usage': usage, 'cached_tokens': cached,
                  'content': content, 'reasoning': reasoning, 'expected_answer': answer, 'correct': correct}
        # Persist each response before reporting completion, including partial/error text.
        save_json(self.output/'responses'/f'{identifier}.json', {'request': body, **result})
        with self.lock:
            self.results.append(result)
            with (self.output/'calls.jsonl').open('a') as f:
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
                f.flush()
        self.event('response', id=identifier, session=session, turn=turn, seconds=elapsed,
                   ttft_seconds=first, usage=usage, error=error, correct=correct)
        return result

    def run_session(self, session):
        for item in session['requests']:
            delay = item['delay_before_seconds']
            if delay:
                self.event('tool_wait', session=session['id'], turn=item['turn'], seconds=delay)
                time.sleep(delay)
            # Continue other fixed turns after an error so expected work remains explicit.
            self.request(session['id'], item['turn'], item)

    def run(self):
        self.started = time.monotonic()
        expected = len(self.workload['quality']) if self.mode == 'quality' else sum(len(s['requests']) for s in self.sessions)
        failures = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            if self.mode == 'quality':
                futures = [pool.submit(self.request, item['id'], 0, item, item['answer']) for item in self.workload['quality']]
            else:
                futures = [pool.submit(self.run_session, session) for session in self.sessions]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures.append(f'{type(error).__name__}: {error}')
                    self.event('runner_error', error=failures[-1])
        seconds = time.monotonic()-self.started
        success_calls = [x for x in self.results if not x['error']]
        output_tokens = sum(x['usage'].get('completion_tokens', 0) for x in success_calls)
        partial_tokens = sum(x['usage'].get('completion_tokens', 0) for x in self.results if x['error'])
        prompt_tokens = sum(x['usage'].get('prompt_tokens', 0) for x in success_calls)
        cache_known = all(isinstance(x['cached_tokens'], int) for x in success_calls) and bool(success_calls)
        cached = sum(x['cached_tokens'] or 0 for x in success_calls)
        correct = sum(x['correct'] is True for x in self.results) if self.mode == 'quality' else None
        graded_total = (expected if self.mode == 'trace' and self.workload['settings'].get('task_mode') == 'qa' else 0)
        graded_correct = sum(x['correct'] is True for x in self.results) if graded_total else None
        failed_count = len(self.results)-len(success_calls)
        complete = len(self.results) == expected and failed_count == 0 and not failures
        summary = {**self.config, 'complete': complete, 'expected_requests': expected,
                   'recorded_requests': len(self.results), 'successful_requests': len(success_calls),
                   'failed_requests': failed_count, 'missing_requests': max(0, expected-len(self.results)),
                   'runner_errors': failures, 'wall_seconds': seconds,
                   'completion_tokens': output_tokens, 'partial_error_completion_tokens': partial_tokens,
                   'generation_tokens_per_second': output_tokens/seconds,
                   'prompt_tokens': prompt_tokens, 'cached_prompt_tokens': cached if cache_known else None,
                   'cached_fraction': cached/prompt_tokens if cache_known and prompt_tokens else None,
                   'latency_p50_seconds': percentile([x['seconds'] for x in success_calls], .5),
                   'latency_p95_seconds': percentile([x['seconds'] for x in success_calls], .95),
                   'ttft_p50_seconds': percentile([x['ttft_seconds'] for x in success_calls], .5),
                   'ttft_p95_seconds': percentile([x['ttft_seconds'] for x in success_calls], .95),
                   'length_capped_requests': sum(x['finish_reason'] == 'length' for x in success_calls),
                   'graded_correct': graded_correct, 'graded_total': graded_total or None,
                   'graded_accuracy': graded_correct/graded_total if graded_total else None,
                   'graded_all_correct': graded_correct == graded_total if graded_total else None,
                   'grading_note': 'Fixed-history multi-turn source QA; not autonomous tool choice or coding success.' if graded_total else None,
                   'quality_correct': correct, 'quality_total': expected if self.mode == 'quality' else None,
                   'quality_all_correct': correct == expected if self.mode == 'quality' else None,
                   'quality_accuracy': correct/expected if self.mode == 'quality' else None,
                   'quality_note': 'Exact-answer snippet probes; not a coding-task or general model-quality evaluation.' if self.mode == 'quality' else None}
        save_json(self.output/'summary.json', summary)
        with (self.output/'summary.csv').open('w') as f:
            row = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in summary.items()}
            writer = csv.DictWriter(f, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        self.event('complete', **summary)
        print(json.dumps(summary, indent=2))
        return summary


def normalize_answer(text):
    # Formatting-only normalization; prose, extra facts and wrong identifiers fail.
    return text.strip().strip('`').rstrip('.').strip().casefold()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Freeze source, histories, delays, caps and quality facts')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--agents', type=int, default=8)
    prepare.add_argument('--turns', type=int, default=4)
    prepare.add_argument('--shared-chars', type=int, default=12000)
    prepare.add_argument('--tool-chars', type=int, default=4000)
    prepare.add_argument('--tool-seconds', type=float, default=.5)
    prepare.add_argument('--seed', type=int, default=0)
    prepare.add_argument('--max-tokens', type=int, default=64)
    prepare.add_argument('--task-mode', choices=['review', 'qa'], default='review',
                         help='qa: grade every multi-turn response against frozen source facts; forces reasoning none')
    prepare.add_argument('--reasoning', choices=['none', 'low', 'medium', 'high'], default='low')
    clone = commands.add_parser('clone', help='Derive a trace output cap without reading live source')
    clone.add_argument('--input', type=Path, required=True)
    clone.add_argument('--output', type=Path, required=True)
    clone.add_argument('--max-tokens', type=int, required=True)
    run = commands.add_parser('run', help='Run against an externally managed server; one configuration at a time')
    run.add_argument('--workload', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--base', default='http://127.0.0.1:8090/v1')
    run.add_argument('--model', required=True)
    run.add_argument('--mode', choices=['trace', 'quality'], default='trace')
    run.add_argument('--sessions', type=int, help='Use the first N frozen sessions')
    run.add_argument('--concurrency', type=int, default=8)
    run.add_argument('--timeout', type=float, default=300)
    run.add_argument('--qwen-thinking-template', action='store_true',
                     help='Explicitly align Qwen enable_thinking with reasoning_effort for cross-engine serving')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        workload = prepare_workload(args.output, args.agents, args.turns, args.shared_chars,
                                    args.tool_chars, args.tool_seconds, args.seed, args.max_tokens, args.reasoning, args.task_mode)
        print(json.dumps({'output': str(args.output), 'workload_sha256': workload['workload_sha256'],
                          'sessions': args.agents, 'turns': args.turns, 'quality_questions': len(workload['quality'])}, indent=2))
        return 0
    if args.command == 'clone':
        workload = clone_workload(args.input, args.output, args.max_tokens)
        print(json.dumps({'output': str(args.output), 'workload_sha256': workload['workload_sha256'],
                          'derived_from': workload['derivation']}, indent=2))
        return 0
    workload = load_workload(args.workload)
    runner = Runner(workload, args.output, args.base, args.model, args.concurrency, args.sessions, args.mode, args.timeout,
                    args.qwen_thinking_template)
    summary = runner.run()
    return 0 if summary['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
