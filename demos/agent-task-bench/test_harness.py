"""Model-free regression of actual concurrent edits and independent success checks."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import threading
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

    def server(self,parallel=False,truncate=False):
        barrier=threading.Barrier(8) if parallel else None
        active=peak=0;lock=threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                nonlocal active,peak
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                messages=body['messages'];initial=messages[1]['content']
                owned=json.loads(re.search(r'EDITABLE FILES: (\[[^\n]+\])',initial).group(1))
                writes=set()
                for m in messages:
                    if m['role']=='assistant':
                        try:
                            a=json.loads(m['content'])
                            if a.get('action')=='write':writes.add(a['path'])
                        except ValueError:pass
                todo=[x for x in owned if x not in writes]
                action={'action':'write','path':todo[0],'content':'OK'} if todo else {'action':'done','summary':'implemented'}
                with lock:active+=1;peak=max(peak,active)
                if barrier and not writes:barrier.wait(timeout=10)
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                chunks=[{'choices':[{'delta':{'content':json.dumps(action)},'finish_reason':None}]},
                        {'choices':[{'delta':{},'finish_reason':'stop'}]},
                        {'choices':[],'usage':{'prompt_tokens':100,'completion_tokens':10}}]
                for chunk in chunks:self.wfile.write(b'data: '+json.dumps(chunk).encode()+b'\n\n')
                if not truncate:self.wfile.write(b'data: [DONE]\n\n')
                with lock:active-=1
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_port}/v1',lambda:peak

    def args(self,task,out,base,agents):
        return argparse.Namespace(task=task,output=out,base=base,model='fake',agents=agents,max_calls=56,
             worker_calls=6,max_tokens=4096,token_budget=96000,max_prompt_chars=50000,request_timeout=30,wall_limit=60,capture=False)

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


if __name__=='__main__':unittest.main()
