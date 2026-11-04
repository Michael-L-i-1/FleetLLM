#!/usr/bin/env python3
"""Run one engine at a time, warm it, measure the demo, and stop the owned server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', required=True)
parser.add_argument('--server-command', type=Path, required=True, help='JSON array of exact server argv')
parser.add_argument('--base', default='http://127.0.0.1:8090/v1')
parser.add_argument('--model', default='glimmer-mlx4')
parser.add_argument('--results', type=Path, default=REPO / 'results/mac/2026-11-03/site-refactor')
parser.add_argument('--reasoning', default='low', choices=['none','low','medium','high'])
parser.add_argument('--max-consecutive-errors', type=int, default=0)
parser.add_argument('--context-prune', action='store_true')
parser.add_argument('--workflow-replay-source', type=Path)
parser.add_argument('--replay-source', type=Path)
args = parser.parse_args()
args.results.mkdir(parents=True, exist_ok=True)
command = json.loads(args.server_command.read_text())
output = HERE / 'runs' / args.label
if output.exists():
    raise SystemExit(f'Output exists: {output}')
log = (args.results / f'{args.label}.server.log').open('w')
server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
sampler = subprocess.Popen([sys.executable, str(HERE / 'sample_process.py'), str(server.pid),
                            str(args.results / f'{args.label}.resources.csv')])
metadata = {'command': command, 'server_pid': server.pid, 'model': args.model, 'base': args.base}
start = time.monotonic()
try:
    while time.monotonic() - start < 240:
        if server.poll() is not None:
            raise RuntimeError(f'Server exited: {server.returncode}. Read server log.')
        try:
            with urllib.request.urlopen(args.base.removesuffix('/v1') + '/health', timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('Server health timeout')
    metadata['startup_seconds'] = time.monotonic() - start
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': 'Reply with exactly READY.'}],
            'max_tokens': 64, 'temperature': 0, 'reasoning_effort': 'low'}
    req = urllib.request.Request(args.base + '/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        warmup = json.load(response)
    (args.results / f'{args.label}.warmup.json').write_text(json.dumps(warmup, indent=2))
    if args.workflow_replay_source:
        run = [sys.executable, str(HERE / 'workflow-replay.py'), '--source', str(args.workflow_replay_source),
               '--output', str(output), '--base', args.base, '--model', args.model]
        if args.context_prune:
            run.append('--context-prune')
    elif args.replay_source:
        run = [sys.executable, str(HERE / 'replay.py'), '--source', str(args.replay_source),
               '--output', str(output), '--base', args.base, '--model', args.model]
    else:
        run = [sys.executable, str(HERE / 'run.py'), 'run', '--label', args.label,
               '--output', str(output), '--base', args.base, '--model', args.model]
    if not args.replay_source and not args.workflow_replay_source:
        run += ['--reasoning', args.reasoning, '--max-consecutive-errors', str(args.max_consecutive_errors)]
    if args.context_prune and not args.replay_source and not args.workflow_replay_source:
        run.append('--context-prune')
    metadata['harness_command'] = run
    with (args.results / f'{args.label}.harness.log').open('w') as harness_log:
        result = subprocess.run(run, stdout=harness_log, stderr=subprocess.STDOUT, timeout=1200, cwd=REPO)
    metadata['harness_returncode'] = result.returncode
    for file in ['summary.json', 'summary.csv', 'calls.json', 'events.jsonl']:
        if (output / file).exists():
            (args.results / f'{args.label}.{file}').write_bytes((output / file).read_bytes())
except Exception as error:
    metadata['error'] = str(error)
    print(str(error), file=sys.stderr)
finally:
    server.terminate()
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    sampler.wait(timeout=10)
    log.close()
    (args.results / f'{args.label}.config.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata))
raise SystemExit(0 if metadata.get('harness_returncode') == 0 else 1)
