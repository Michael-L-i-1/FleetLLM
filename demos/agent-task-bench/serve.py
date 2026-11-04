#!/usr/bin/env python3
"""Own one GPU server for a real task or a small untimed configuration pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',type=Path)
    p.add_argument('--agents',type=int,choices=[1,8],default=8)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--capture',action='store_true')
    p.add_argument('--wall-limit',type=float,default=2700)
    a=p.parse_args();cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    base=cfg.get('base','http://127.0.0.1:8090/v1');health=base.removesuffix('/v1')+'/health'
    try:urllib.request.urlopen(health,timeout=1)
    except OSError:pass
    else:raise RuntimeError('Refusing occupied model endpoint')
    meta={'config':cfg,'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
          'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'model_binary_sha256':hashlib.sha256(Path(cfg['command'][0]).resolve().read_bytes()).hexdigest(),
          'complete':False,'pilot':a.pilot,'agents':a.agents,'task':str(a.task) if a.task else None}
    server=None;samplers=[]
    def request(messages,max_tokens):
        body={'model':cfg.get('model','glimmer-mlx4'),'messages':messages,'temperature':0,'max_tokens':max_tokens,
              'reasoning_effort':'low','chat_template_kwargs':{'reasoning_strength':'low'}}
        req=urllib.request.Request(base+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
        t=time.monotonic()
        with urllib.request.urlopen(req,timeout=600) as response:d=json.load(response)
        return {'seconds':time.monotonic()-t,'request':body,'response':d}
    try:
        with (a.output/'server.log').open('w') as log:
            server=subprocess.Popen(cfg['command'],cwd=ROOT,env=dict(os.environ,**cfg.get('env',{})),
                                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            meta['server_pid']=server.pid
            for pid,name in [(server.pid,'model-resources.csv'),(os.getpid(),'run-resources.csv')]:
                samplers.append(subprocess.Popen([sys.executable,str(HERE/'sample_memory.py'),str(pid),str(a.output/name)]))
            t=time.monotonic()
            while True:
                if server.poll() is not None:raise RuntimeError(f'Server exited {server.returncode}')
                try:urllib.request.urlopen(health,timeout=1);break
                except OSError:
                    if time.monotonic()-t>180:raise TimeoutError('Model startup timeout')
                    time.sleep(.5)
            meta['startup_seconds']=time.monotonic()-t
            warm=request([{'role':'user','content':'Return exactly one JSON object: {"action":"done","summary":"ready"}'}],128)
            (a.output/'warmup.json').write_text(json.dumps(warm,indent=2)+'\n')
            if a.pilot:
                common='You are a coding agent. Return one JSON object with action=write, path and content, no Markdown. '
                common+='Use self-contained concise code. '+''.join(f'Project contract {i}: use only local standard library functions.\n' for i in range(60))
                messages=[]
                for i in range(8):
                    problem=('Write Rust function pub fn slug(input: &str) -> String that lowercases ASCII and collapses nonalphanumeric runs to one hyphen, trimming hyphens.' if i%2==0 else 'Write an accessible HTML coffee menu section with scoped CSS, three cards Espresso $3, Latte $5, Tea $4, responsive grid, and a labelled search input.')
                    messages.append([{'role':'system','content':common},{'role':'user','content':f'Worker {i}. {problem} Path: part{i}.'}])
                t=time.monotonic()
                with ThreadPoolExecutor(max_workers=8) as pool:responses=list(pool.map(lambda x:request(x,512),messages))
                wall=time.monotonic()-t;tokens=sum(d['response'].get('usage',{}).get('completion_tokens',0) for d in responses)
                pilot={'wall_seconds':wall,'completion_tokens':tokens,'tokens_per_second':tokens/wall,'responses':responses}
                (a.output/'pilot.json').write_text(json.dumps(pilot,indent=2)+'\n');meta['pilot_summary']={k:v for k,v in pilot.items() if k!='responses'}
            else:
                if a.task is None:raise ValueError('--task required')
                command=[sys.executable,str(HERE/'run.py'),'--task',str(a.task.resolve()),'--output',str(a.output/'task'),
                         '--base',base,'--model',cfg.get('model','glimmer-mlx4'),'--agents',str(a.agents),'--wall-limit',str(a.wall_limit)]
                if a.capture:command+=['--capture']
                meta['harness_command']=command
                meta['task_start_unix']=time.time()
                with (a.output/'harness.log').open('w') as log:
                    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_limit+180)
                meta['task_end_unix']=time.time()
                meta['harness_returncode']=result.returncode
                if (a.output/'task/summary.json').exists():
                    summary=json.loads((a.output/'task/summary.json').read_text());meta['task_passed']=summary['passed']
                    if summary.get('infrastructure_error'):
                        raise RuntimeError('Protected checker infrastructure failure; trial cannot be scored')
                else:raise RuntimeError('Harness did not preserve a summary')
                # A scored task failure is a complete trial, not an infrastructure success.
            meta['complete']=True
    except Exception as error:meta['error']=str(error)
    finally:
        if server:
            try:os.killpg(server.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL);server.wait()
        for sampler in samplers:
            # The run-wide sampler follows this driver, so terminate it explicitly.
            sampler.terminate()
            try:sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:sampler.kill();sampler.wait()
        (a.output/'config.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:meta.get(k) for k in ['complete','error','task_passed','pilot_summary']}))
    return 0 if meta['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
