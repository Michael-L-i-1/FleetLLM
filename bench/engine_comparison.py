#!/usr/bin/env python3
"""Run one configured engine against a frozen workload, preserving provenance."""
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

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server-config', type=Path, required=True)
    p.add_argument('--workload', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sessions', type=int, default=8)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--startup-timeout', type=float, default=180)
    a = p.parse_args()
    cfg = json.loads(a.server_config.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    base = cfg['base'].rstrip('/')
    health = base.removesuffix('/v1') + '/health'
    try:
        urllib.request.urlopen(health, timeout=1)
    except OSError:
        pass
    else:
        raise RuntimeError('Refusing occupied health endpoint')
    binary = Path(cfg['command'][0]).resolve()
    meta = {'server_config': cfg, 'workload': str(a.workload.resolve()),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'warmup': '8 concurrent unrelated ~2k-token text prompts, cap16, thinking disabled; excluded',
            'sessions': a.sessions, 'complete': False}
    server = sampler = None

    def request(body):
        req = urllib.request.Request(base + '/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=a.timeout))

    try:
        with (a.output/'server.log').open('w') as log:
            server = subprocess.Popen(cfg['command'], cwd=ROOT, env=dict(os.environ, **cfg.get('env', {})),
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            meta['server_pid'] = server.pid
            sampler = subprocess.Popen([sys.executable, str(ROOT/'demos/site-refactor/sample_process.py'),
                                        str(server.pid), str(a.output/'resources.csv')])
            start = time.monotonic()
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited {server.returncode}')
                try:
                    urllib.request.urlopen(health, timeout=1)
                    break
                except OSError:
                    if time.monotonic()-start > a.startup_timeout:
                        raise TimeoutError('Server startup timeout')
                    time.sleep(.5)
            meta['startup_seconds'] = time.monotonic()-start
            probes = []
            for i in range(8):
                probes.append({'model': cfg['model'], 'messages': [
                    {'role': 'system', 'content': 'Unrelated warmup. Reply with only READY.'},
                    {'role': 'user', 'content': f'Warmup request {i}.\n' +
                     ''.join(f'Note {j}: unrelated synthetic warmup material.\n' for j in range(150)) + '\nReply READY.'}],
                    'temperature': 0, 'max_tokens': 16, 'reasoning_effort': 'none',
                    'chat_template_kwargs': {'enable_thinking': False}})
            with ThreadPoolExecutor(max_workers=8) as pool:
                warm = list(pool.map(request, probes))
            (a.output/'warmup.json').write_text(json.dumps(warm, indent=2)+'\n')
            for phase in ['trace', 'quality']:
                command = [sys.executable, str(ROOT/'bench/throughput_workflows.py'), 'run',
                           '--workload', str(a.workload.resolve()), '--output', str(a.output/phase),
                           '--base', base, '--model', cfg['model'], '--mode', phase,
                           '--sessions', str(a.sessions), '--concurrency', str(a.sessions if phase == 'trace' else 2),
                           '--timeout', str(a.timeout), '--qwen-thinking-template']
                meta[phase+'_command'] = command
                with (a.output/(phase+'.log')).open('w') as out:
                    result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, timeout=a.timeout)
                meta[phase+'_returncode'] = result.returncode
                if result.returncode:
                    raise RuntimeError(f'{phase} incomplete; preserve partial results without a successful speedup')
            try:
                with urllib.request.urlopen(base.removesuffix('/v1')+'/metrics', timeout=5) as response:
                    (a.output/'metrics.txt').write_bytes(response.read())
            except OSError:
                pass
            meta['complete'] = True
    except Exception as error:
        meta['error'] = str(error)
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        if sampler is not None:
            try:
                sampler.wait(timeout=15)
            except subprocess.TimeoutExpired:
                sampler.terminate()
                sampler.wait(timeout=5)
        (a.output/'config.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'output': str(a.output), 'complete': meta['complete'], 'error': meta.get('error')}))
    return 0 if meta['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
