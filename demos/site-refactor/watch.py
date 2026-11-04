#!/usr/bin/env python3
"""Terminal view of saved demo events; q/Ctrl-C exits without stopping agents."""
import argparse
import json
import re
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
parser.add_argument('--server-log', type=Path)
args = parser.parse_args()
root = args.run
try:
    while True:
        path = root / 'events.jsonl'
        states = {}
        calls = 0
        tokens = 0
        elapsed = 0
        complete = False
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            elapsed = event['elapsed']
            name = event.get('agent')
            if name:
                states[name] = event.get('action', event['event'])
            if event['event'] == 'response':
                calls += 1
                tokens += event.get('usage', {}).get('completion_tokens', 0)
            if event['event'] == 'complete':
                complete = True
                states['run'] = 'checks passed' if event['success'] else event.get('error', 'failed')
        print('\033[2J\033[H', end='')
        if args.server_log and args.server_log.exists():
            logs = args.server_log.read_text(errors='replace')
            rates = re.findall(r'(?:active \| |Avg generation throughput: )([\d.]+) (?:tok/s|tokens/s)', logs)
            if rates:
                print(f'Latest server generation window: {rates[-1]} aggregate tokens/s')
        print(f'FleetLLM site refactor — {root.name}\nLast event: {elapsed:.1f}s | calls: {calls} | generated tokens: {tokens}\n')
        for name, state in sorted(states.items()):
            print(f'{name:16} {state}')
        print('\nCtrl-C closes this viewer only. Events and outputs remain on disk.')
        if complete:
            break
        time.sleep(1)
except KeyboardInterrupt:
    pass
