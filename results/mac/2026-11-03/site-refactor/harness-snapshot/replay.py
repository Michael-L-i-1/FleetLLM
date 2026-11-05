#!/usr/bin/env python3
"""Replay a fixed set of real worker prompts concurrently; measures inference, not task success."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from run import Demo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--base', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--max-tokens', type=int, default=256)
parser.add_argument('--timeout', type=float, default=300)
parser.add_argument('--reasoning', default='low')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
prompts = []
for path in sorted(args.source.glob('worker-*.jsonl')):
    turns = [json.loads(line) for line in path.read_text().splitlines()]
    turn = max(turns, key=lambda row: row['response'].get('usage', {}).get('completion_tokens', 0))
    prompts.append({'agent': path.stem, 'messages': turn['messages']})
if len(prompts) != 4:
    raise SystemExit('Need a trace for each of four workers')
(args.output / 'prompts.json').write_text(json.dumps(prompts, indent=2))
demo = Demo.__new__(Demo)
demo.args = args
demo.root = args.output
demo.started = time.monotonic()
demo.lock = threading.Lock()
demo.calls = []

def request(prompt):
    text, reply = demo.request(prompt['agent'], prompt['messages'])
    (args.output / f"{prompt['agent']}.json").write_text(json.dumps(reply, indent=2))
    return text

with ThreadPoolExecutor(max_workers=args.workers) as pool:
    list(pool.map(request, prompts))
seconds = time.monotonic() - demo.started
summary = {'seconds': seconds, 'requests': len(prompts), 'concurrency': args.workers,
           'completion_tokens': sum(c['usage'].get('completion_tokens', 0) for c in demo.calls),
           'prompt_tokens': sum(c['usage'].get('prompt_tokens', 0) for c in demo.calls)}
summary['aggregate_completion_tokens_per_second'] = summary['completion_tokens'] / seconds
(args.output / 'calls.json').write_text(json.dumps(demo.calls, indent=2))
(args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
