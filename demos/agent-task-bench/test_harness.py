"""Model-free regression of actual concurrent edits and independent success checks."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest

spec=importlib.util.spec_from_file_location('real_task_runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class HarnessTest(unittest.TestCase):
    def fixture(self,root):
        task=root/'task';fixture=task/'fixture';fixture.mkdir(parents=True)
        files=[f'module{i}.txt' for i in range(8)]
        for f in files:(fixture/f).write_text('TODO')
        (fixture/'contract.md').write_text('Implement all8modules withthe string OK. Protected contract.')
        checker=task/'check.py';checker.write_text('import json,pathlib,sys\np=pathlib.Path(sys.argv[1]);ok=all((p/f"module{i}.txt").read_text()=="OK" for i in range(8));print(json.dumps({"passed":ok}));sys.exit(0 if ok else 1)\n')
        (task/'task.json').write_text(json.dumps({'name':'fixture','fixture':'fixture','instructions':'Implement all modules.',
            'editable_files':files,'workers':[{'name':f'worker{i}','file':f,'brief':'Write OK and finish.'} for i,f in enumerate(files)],
            'validation_command':['python3','{task_dir}/check.py','{workspace}']}))
        return task

    def server(self,parallel=False,truncate=False,multiround=False,invalid=False,capped=False,stall=False):
        self.observed=[]
        self.stream_started=threading.Event()
        self.stream_release=threading.Event()
        barrier=threading.Barrier(8) if parallel else None
        active=peak=0;lock=threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                nonlocal active,peak
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self_outer.observed.append(body)
                messages=body['messages'];initial=messages[1]['content']
                round_index=max(i for i,m in enumerate(messages) if m['role']=='user' and m['content'].startswith('CURRENT ROUND'))
                round_number=int(re.search(r'CURRENT ROUND (\d+)',messages[round_index]['content']).group(1))
                owned=json.loads(re.search(r'EDITABLE FILES: (\[[^\n]+\])',initial).group(1))
                writes=set()
                for m in messages[round_index:] if multiround else messages:
                    if m['role']=='assistant':
                        try:
                            a=json.loads(m['content'])
                            if a.get('action')=='write':writes.add(a['path'])
                        except ValueError:pass
                todo=[x for x in owned if x not in writes]
                action={'action':'write','path':todo[0],'content':f'OK{round_number}' if multiround else 'OK'} if todo else {'action':'done','summary':'implemented'}
                if invalid:action={'action':'unavailable'}
                with lock:active+=1;peak=max(peak,active)
                if barrier and not writes:barrier.wait(timeout=10)
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                if stall:
                    self.wfile.write(b'data: {"choices":[]}\n\n');self.wfile.flush()
                    self_outer.stream_started.set();self_outer.stream_release.wait(timeout=5)
                    with lock:active-=1
                    return
                chunks=[{'choices':[{'delta':{'content':json.dumps(action)},'finish_reason':None}]},
                        {'choices':[{'delta':{},'finish_reason':'length' if capped else 'stop'}]},
                        {'choices':[],'usage':{'prompt_tokens':100,'completion_tokens':10}}]
                for chunk in chunks:self.wfile.write(b'data: '+json.dumps(chunk).encode()+b'\n\n')
                if not truncate:self.wfile.write(b'data: [DONE]\n\n')
                with lock:active-=1
        self_outer=self
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_port}/v1',lambda:peak

    def args(self,task,out,base,agents):
        return argparse.Namespace(task=task,output=out,base=base,model='fake',agents=agents,max_calls=56,
             worker_calls=6,max_tokens=4096,token_budget=96000,max_prompt_chars=50000,request_timeout=30,wall_limit=60,capture=False)

    def multiround_fixture(self,root):
        task=self.fixture(root)
        spec=json.loads((task/'task.json').read_text())
        spec['rounds']=[{'id':i,'title':f'Round {i}','instructions':f'Write OK{i} in every editable module.',
                         'worker_instructions':{f'module{j}.txt':f'Write OK{i}.' for j in range(8)}} for i in (1,2,3)]
        spec['validation_command'] += ['{round}']
        (task/'task.json').write_text(json.dumps(spec))
        (task/'check.py').write_text('import json,pathlib,sys\np=pathlib.Path(sys.argv[1]);r=sys.argv[2];ok=all((p/f"module{i}.txt").read_text()=="OK"+r for i in range(8));print(json.dumps({"passed":ok,"round":r}));sys.exit(0 if ok else 1)\n')
        return task

    def test_multiround_persistent_workers_and_same_shared_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.multiround_fixture(root);base,_=self.server(multiround=True)
            a=self.args(task,root/'run',base,8);a.max_calls=108;a.worker_calls=3
            b=runner.Benchmark(a);s=b.run()
            self.assertTrue(s['passed']);self.assertTrue(s['all_rounds_passed'])
            self.assertEqual([r['calls'] for r in s['rounds']],[16,16,16])
            self.assertEqual([r['round'] for r in s['rounds']],[1,2,3])
            for name,history in b.histories.items():
                self.assertEqual(sum(m['role']=='system' for m in history),1)
                self.assertEqual(sum(m['content'].startswith('CURRENT ROUND') for m in history),3)
                self.assertIn('OK1',json.dumps(history));self.assertIn('OK3',json.dumps(history))
            prefixes={request['messages'][1]['content'].split('YOUR ROLE:')[0] for request in self.observed}
            self.assertEqual(len(prefixes),1)
            self.assertTrue(all((b.workspace/f'module{i}.txt').read_text()=='OK3' for i in range(8)))
            self.assertGreaterEqual(s['time_to_first_passing_check_seconds'],s['rounds'][2]['time_to_first_passing_check_seconds'])
            events=[json.loads(line) for line in (b.out/'events.jsonl').read_text().splitlines()]
            self.assertEqual([e['round'] for e in events if e['event']=='round_start'],[1,2,3])
            self.assertTrue(all('phase' in e and 'round' in e for e in events if e['event']=='file_write'))

    def test_multiround_single_uses_persistent_whole_project_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.multiround_fixture(root);base,_=self.server(multiround=True)
            a=self.args(task,root/'run',base,1);a.max_calls=108
            b=runner.Benchmark(a);s=b.run()
            self.assertTrue(s['passed']);self.assertEqual([r['calls'] for r in s['rounds']],[9,9,9])
            self.assertEqual(set(b.histories),{'single'})
            self.assertEqual(sum(m['content'].startswith('CURRENT ROUND') for m in b.histories['single']),3)

    def test_three_invalid_or_capped_actions_stop_agent_then_continue_rounds(self):
        for capped in (False,True):
            with self.subTest(capped=capped),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);task=self.multiround_fixture(root)
                base,_=self.server(multiround=True,invalid=not capped,capped=capped)
                a=self.args(task,root/'run',base,1);a.max_calls=108
                b=runner.Benchmark(a);s=b.run()
                self.assertFalse(s['passed']);self.assertEqual([r['calls'] for r in s['rounds']],[3,3,3])
                self.assertTrue(all(r['workers'][0]['status']=='three_invalid_actions' for r in s['rounds']))
                self.assertEqual((b.workspace/'module0.txt').read_text(),'TODO')
                self.assertIsNone(s['time_to_first_passing_check_seconds'])
                self.assertEqual(s['length_capped_calls'],9 if capped else 0)

    def test_context_compaction_keeps_all_stakeholder_round_instructions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.fixture(root);base,_=self.server()
            a=self.args(task,root/'run',base,1);a.max_prompt_chars=1500
            b=runner.Benchmark(a)
            messages=[{'role':'system','content':'contract'},{'role':'user','content':'shared'}]
            for i in (1,2,3):
                messages += [{'role':'user','content':f'CURRENT ROUND {i}: requirement'},
                             {'role':'assistant','content':'x'*1000},{'role':'user','content':'tool result'}]
            compacted=b.compact(messages,'single')
            self.assertLess(len(compacted),len(messages))
            self.assertEqual(sum(m['content'].startswith('CURRENT ROUND') for m in compacted),3)
            self.assertEqual(messages[2]['content'],'CURRENT ROUND 1: requirement')

    def test_cancel_interrupts_active_stream_and_preserves_partial_response(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.fixture(root);base,_=self.server(stall=True)
            b=runner.Benchmark(self.args(task,root/'run',base,1))
            def cancel():
                self.assertTrue(self.stream_started.wait(5))
                time.sleep(.05)
                b.cancel('mock active cancellation')
            thread=threading.Thread(target=cancel);thread.start()
            try:
                started=time.monotonic();s=b.run()
                self.assertLess(time.monotonic()-started,3)
                self.assertTrue(s['cancelled']);self.assertFalse(s['passed'])
                self.assertEqual(s['calls'],1);self.assertEqual(s['failed_calls'],1)
                self.assertEqual((b.workspace/'module0.txt').read_text(),'TODO')
                self.assertEqual(len(list((b.out/'responses').glob('*.json'))),1)
            finally:self.stream_release.set();thread.join()

    def test_cancel_preserves_summary_without_model_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.multiround_fixture(root);base,_=self.server()
            b=runner.Benchmark(self.args(task,root/'run',base,1));b.cancel('mock cancellation');s=b.run()
            self.assertFalse(s['passed']);self.assertTrue(s['cancelled']);self.assertEqual(s['calls'],0)
            self.assertTrue((b.out/'summary.json').exists())

    def test_eight_workers_really_overlap_and_solve(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.fixture(root);base,peak=self.server(parallel=True)
            b=runner.Benchmark(self.args(task,root/'run',base,8));s=b.run()
            self.assertTrue(s['passed']);self.assertEqual(peak(),8);self.assertEqual(s['calls'],16)
            self.assertIsNotNone(s['time_to_first_passing_check_seconds'])
            with self.assertRaises(ValueError):b.path('../outside')
            (b.workspace/'contract.md').write_text('tampered')
            self.assertFalse(b.check()['passed'])

    def test_single_agent_gets_same_whole_task(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.fixture(root);base,peak=self.server()
            s=runner.Benchmark(self.args(task,root/'run',base,1)).run()
            self.assertTrue(s['passed']);self.assertEqual(peak(),1);self.assertEqual(s['calls'],9)

    def test_truncated_stream_does_not_execute_a_complete_looking_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.fixture(root);base,_=self.server(truncate=True)
            b=runner.Benchmark(self.args(task,root/'run',base,1));s=b.run()
            self.assertFalse(s['passed']);self.assertIsNone(s['time_to_first_passing_check_seconds'])
            self.assertEqual((b.workspace/'module0.txt').read_text(),'TODO')
            self.assertEqual(s['failed_calls'],1)

    def test_validator_crash_is_infrastructure_not_a_repair_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.fixture(root);base,_=self.server()
            (task/'check.py').write_text('raise RuntimeError("broken grader interface")\n')
            s=runner.Benchmark(self.args(task,root/'run',base,1)).run()
            self.assertFalse(s['passed']);self.assertTrue(s['infrastructure_error'])
            self.assertEqual(s['calls'],9)


if __name__=='__main__':unittest.main()
