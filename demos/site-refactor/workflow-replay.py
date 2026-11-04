#!/usr/bin/env python3
"""Replay fixed recorded agent histories; generated actions are never executed.

This is an inference/cache experiment, not a task-success evaluation. Histories
remain fixed even if generated responses differ. Output is capped, not fixed work.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import statistics
import threading
import time

from run import Demo

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', required=True)
p.add_argument('--base', default='http://127.0.0.1:8090/v1')
p.add_argument('--model', default='qwen35-4b-mlx4')
p.add_argument('--context-prune', action='store_true')
a = p.parse_args()
a.label = Path(a.output).name
a.reasoning, a.max_tokens, a.timeout = 'low', 128, 180
demo = Demo(a)
source = {}
for path in sorted(a.source.glob('worker-*.jsonl')):
    source[path.stem] = [json.loads(line)['messages'] for line in path.read_text().splitlines()[:3]]
if len(source) != 4:
    raise SystemExit('Need four recorded workers')
barrier = threading.Barrier(4)

def worker(item):
    name, histories = item
    barrier.wait(timeout=10)
    for turn, messages in enumerate(histories):
        demo.event('request', agent=name, turn=turn)
        _, reply = demo.request(name, messages)
        with (demo.root/f'{name}.jsonl').open('a') as f:
            f.write(json.dumps({'messages': messages, 'response': reply,
                                'masked_message_indices': demo.context_drops[name]})+'\n')

start = time.monotonic()
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(worker, source.items()))
seconds = time.monotonic()-start
summary = {'label': a.label, 'model': a.model, 'success': None, 'seconds': seconds,
           'calls': len(demo.calls), 'error': None,
           'prompt_tokens': sum(c['usage'].get('prompt_tokens',0) for c in demo.calls),
           'completion_tokens': sum(c['usage'].get('completion_tokens',0) for c in demo.calls),
           'workload': 'fixed recorded histories, three turns per worker, 128-token cap, no tool execution'}
summary['aggregate_completion_tokens_per_second'] = summary['completion_tokens']/seconds
ttfts = [c['ttft_seconds'] for c in demo.calls if c['ttft_seconds'] is not None]
summary['median_ttft_seconds'] = statistics.median(ttfts) if ttfts else None
(demo.root/'calls.json').write_text(json.dumps(demo.calls,indent=2)+'\n')
(demo.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (demo.root/'summary.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=summary)
    w.writeheader()
    w.writerow(summary)
demo.event('complete', **summary)
