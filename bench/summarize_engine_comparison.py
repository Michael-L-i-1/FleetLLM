#!/usr/bin/env python3
"""Summarize complete cross-engine trials separately from setup and tuning."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--plan', type=Path, help='Use a separately frozen validation plan')
    p.add_argument('--prefix', default='', help='Prefix output filenames to retain multiple phases')
    a = p.parse_args()
    plan = json.loads((a.plan or a.results/'matrix/plan.json').read_text())
    prefix = a.prefix + '-' if a.prefix else ''
    rows, groups = [], {}
    for job in plan['jobs']:
        root = a.results/'runs'/job['label']
        if not (root/'config.json').exists():
            continue
        config = json.loads((root/'config.json').read_text())
        summary = json.loads((root/'trace/summary.json').read_text()) if (root/'trace/summary.json').exists() else {}
        quality = json.loads((root/'quality/summary.json').read_text()) if (root/'quality/summary.json').exists() else {}
        complete = config.get('complete') is True and summary.get('complete') is True
        row = {'label': job['label'], 'engine': job['engine'], 'workload': job['workload_kind'],
               'replicate': job['replicate'], 'complete': complete,
               **{key: summary.get(key) for key in [
                   'wall_seconds', 'completion_tokens', 'generation_tokens_per_second', 'prompt_tokens',
                   'cached_fraction', 'successful_requests', 'failed_requests', 'missing_requests',
                   'length_capped_requests', 'graded_correct', 'graded_total', 'ttft_p50_seconds',
                   'ttft_p95_seconds', 'latency_p50_seconds', 'latency_p95_seconds']},
               'short_quality_correct': quality.get('quality_correct'),
               'short_quality_total': quality.get('quality_total')}
        resources = root/'resources.csv'
        if resources.exists():
            with resources.open() as f:
                samples = list(csv.DictReader(f))
            row['max_process_rss_sum_mb'] = max((float(s['rss_sum_mb']) for s in samples), default=None)
        else:
            row['max_process_rss_sum_mb'] = None
        rows.append(row)
        if complete:
            groups.setdefault((job['engine'], job['workload_kind']), []).append(row)
    if rows:
        with (a.results/(prefix+'comparison.csv')).open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    aggregates = []
    for (engine, workload), group in sorted(groups.items()):
        aggregates.append({'engine': engine, 'workload': workload, 'repeats': len(group),
                           'wall_seconds_mean': statistics.mean(x['wall_seconds'] for x in group),
                           'wall_seconds_min': min(x['wall_seconds'] for x in group),
                           'wall_seconds_max': max(x['wall_seconds'] for x in group),
                           'generation_tok_per_second_mean': statistics.mean(x['generation_tokens_per_second'] for x in group),
                           'completion_tokens_each': [x['completion_tokens'] for x in group],
                           'graded_correct_each': [x['graded_correct'] for x in group],
                           'graded_total_each': [x['graded_total'] for x in group],
                           'all_transport_complete': True,
                           'note': 'Capped review output and different quantizations do not establish equal coding quality.'})
    (a.results/(prefix+'aggregates.json')).write_text(json.dumps(aggregates, indent=2)+'\n')
    print(json.dumps({'recorded_trials': len(rows), 'complete_trials': sum(x['complete'] for x in rows),
                      'planned_trials': len(plan['jobs']), 'groups': aggregates}, indent=2))


if __name__ == '__main__':
    main()
