#!/usr/bin/env python3
"""Own one Fleet server and persist one controlled workflow/quality experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--label', required=True)
p.add_argument('--binary', default='build/fleet-serve')
p.add_argument('--workload', default='results/throughput-exploration/workload32.json')
p.add_argument('--results', default='results/throughput-exploration/runs')
p.add_argument('--sessions', type=int, default=8)
p.add_argument('--max-active', type=int)
p.add_argument('--kv-gb', type=float, default=4)
p.add_argument('--recurrent-fraction', type=float, default=.25)
p.add_argument('--row-budget', type=int, default=128)
p.add_argument('--kv-int8', action='store_true')
p.add_argument('--dedup', action='store_true')
p.add_argument('--checkpoints', action='store_true')
p.add_argument('--nax256', action='store_true')
p.add_argument('--state-move', action='store_true')
p.add_argument('--cache-priority', action='store_true')
p.add_argument('--wall-limit', type=float, default=360)
p.add_argument('--port', type=int, default=8090)
a = p.parse_args()
r = ROOT/a.results/a.label
r.mkdir(parents=True, exist_ok=False)
model = str(Path.home()/'.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-4bit/snapshots/0e7ffd5c629ef7719d4cbc04069232580bfa9d9c')
command = [str((ROOT/a.binary).resolve()), '--model', model, '--device', 'metal', '--name', 'qwen35-4b-mlx4',
           '--budget-gb', '16', '--kv-gb', str(a.kv_gb), '--max-active', str(a.max_active or a.sessions),
           '--admit', '--row-budget', str(a.row_budget), '--reasoning', 'low', '--max-tokens', '4096', '--port', str(a.port)]
if a.recurrent_fraction != .25:
    command += ['--recurrent-fraction', str(a.recurrent_fraction)]
if a.kv_int8:
    command.append('--kv-int8')
if a.dedup:
    command.append('--recurrent-prefix-dedup')
if a.checkpoints:
    command.append('--prompt-checkpoints')
if a.state_move:
    command.append('--recurrent-state-move')
if a.cache_priority:
    command.append('--recurrent-cache-priority')
env = dict(os.environ, FLEET_Q35_NAX_ATTN='1' if a.nax256 else '0', FLEET_TRACE_STATS='1')
metadata = {'settings':vars(a), 'server_command':command,
            'binary_sha256':hashlib.file_digest(open(command[0],'rb'),'sha256').hexdigest(),
            'environment':{key:env[key] for key in ['FLEET_Q35_NAX_ATTN','FLEET_TRACE_STATS']},
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
(r/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
server = sampler = None
base = f'http://127.0.0.1:{a.port}/v1'
def request(body):
    req = urllib.request.Request(base+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=60))
try:
    # This driver must own its port, not accidentally warm or measure another server.
    try:
        urllib.request.urlopen(base.removesuffix('/v1')+'/health',timeout=1)
    except OSError:
        pass
    else:
        raise RuntimeError('Port already serves a model; refusing to attach to an unrelated process')
    with (r/'server.log').open('w') as serverlog:
        server = subprocess.Popen(command, cwd=ROOT,env=env,stdout=serverlog,stderr=subprocess.STDOUT)
        metadata['server_pid'] = server.pid
        sampler = subprocess.Popen([sys.executable,str(ROOT/'demos/site-refactor/sample_process.py'),str(server.pid),str(r/'resources.csv')])
        start = time.monotonic()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Server exited {server.returncode}')
            try:
                with urllib.request.urlopen(base.removesuffix('/v1')+'/health',timeout=1) as health:
                    if health.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic()-start>90:
                raise RuntimeError('Startup timeout')
            time.sleep(.25)
        metadata['startup_seconds'] = time.monotonic()-start
        warmup = request({'model':'qwen35-4b-mlx4','messages':[{'role':'user','content':'Reply exactly READY.'}],
                          'temperature':0,'max_tokens':16,'reasoning_effort':'none'})
        (r/'warmup.json').write_text(json.dumps(warmup,indent=2)+'\n')
        for mode in ['trace','quality']:
            run = [sys.executable,str(ROOT/'bench/throughput_workflows.py'),'run','--workload',str((ROOT/a.workload).resolve()),
                   '--output',str(r/mode),'--base',base,'--model','qwen35-4b-mlx4','--mode',mode,
                   '--sessions',str(a.sessions),'--concurrency',str(a.sessions if mode=='trace' else 2),'--timeout',str(a.wall_limit)]
            metadata[mode+'_command'] = run
            with (r/(mode+'.log')).open('w') as log:
                result = subprocess.run(run,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_limit)
            metadata[mode+'_returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError(mode+' did not complete; preserve partial events and do not count this as successful throughput')
        metadata['complete'] = True
except Exception as error:
    metadata['complete'] = False
    metadata['error'] = str(error)
finally:
    if server is not None:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    if sampler is not None:
        sampler.wait(timeout=10)
    (r/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'label':a.label,'complete':metadata.get('complete'),'error':metadata.get('error')}))
raise SystemExit(0 if metadata.get('complete') else 1)
