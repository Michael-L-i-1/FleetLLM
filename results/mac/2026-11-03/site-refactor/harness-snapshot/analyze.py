#!/usr/bin/env python3
"""Summarize saved task/replay runs without conflating inference throughput with success."""
import argparse
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
args = parser.parse_args()
rows = []
for path in sorted(args.results.glob('*.summary.json')):
    label = path.name.removesuffix('.summary.json')
    summary = json.loads(path.read_text())
    calls_path = args.results / f'{label}.calls.json'
    calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
    cache_known = any((c['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') is not None for c in calls)
    cached = sum((c['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0 for c in calls)
    ttft = sorted(c['ttft_seconds'] for c in calls if c.get('ttft_seconds') is not None)
    row = {**summary, 'run': label, 'cached_prompt_tokens': cached if cache_known else None,
           'cached_fraction': cached / max(summary.get('prompt_tokens', 0), 1) if cache_known else None,
           'median_ttft_seconds': statistics.median(ttft) if ttft else None,
           'p95_ttft_seconds': ttft[max(0, int(.95*len(ttft)+.999)-1)] if ttft else None,
           'average_inflight_api_requests': sum(c['seconds'] for c in calls) / max(summary['seconds'], 1e-9),
           'length_capped_calls': sum(c.get('finish_reason') == 'length' for c in calls)}
    events_path = args.results / f'{label}.events.jsonl'
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        delegated = [e['elapsed'] for e in events if e['event'] == 'delegated']
        integrated = [e['elapsed'] for e in events if e['event'] == 'integrated']
        if delegated and integrated:
            seconds = integrated[0] - delegated[0]
            worker_tokens = sum(c['usage'].get('completion_tokens', 0) for c in calls if c['agent'].startswith('worker'))
            row.update(worker_phase_seconds=seconds, worker_tokens=worker_tokens,
                       worker_phase_completion_tokens_per_second=worker_tokens/seconds)
    server = args.results / f'{label}.server.log'
    if not server.exists() and label == 'fleet-mlx4-span-01':
        server = args.results / 'fleet-mlx4-span.server.log'
    if server.exists():
        log = server.read_text(errors='replace')
        rates = re.findall(r'active \| ([\d.]+) tok/s generated', log)
        if rates:
            row['peak_logged_generation_window_tokens_per_second'] = max(map(float, rates))
        snapshots = re.findall(r'\[serve\] \d+ active \| .*', log)
        if snapshots:
            row['last_server_snapshot'] = snapshots[-1]
    run_folder = Path(__file__).resolve().parent / 'runs' / label
    failed_checks = 0
    for transcript in run_folder.glob('*.jsonl'):
        if transcript.name == 'events.jsonl':
            continue
        for line in transcript.read_text().splitlines():
            turn = json.loads(line)
            messages = turn.get('messages', [])
            if messages and messages[-1].get('role') == 'user':
                try:
                    feedback = json.loads(messages[-1]['content']).get('tool_result', {})
                    if isinstance(feedback, dict) and feedback.get('passed') is False:
                        failed_checks += 1
                except (ValueError, AttributeError):
                    pass
    row['failed_check_feedbacks'] = failed_checks
    rows.append(row)
(args.results / 'analysis.json').write_text(json.dumps(rows, indent=2))
for row in rows:
    print(row['run'], 'success=', row.get('success', 'replay'),
          'seconds=', round(row['seconds'], 2),
          'aggregate_tok_s=', round(row['aggregate_completion_tokens_per_second'], 2),
          'cached=', round(row['cached_fraction']*100, 1) if row['cached_fraction'] is not None else 'not reported', '%')
