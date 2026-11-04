#!/usr/bin/env python3
"""Summarize durable throughput runs without counting incomplete work as gains."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    rows = []
    for run in sorted((args.results / 'runs').iterdir()):
        config_file = run / 'config.json'
        summary_file = run / 'trace/summary.json'
        if not config_file.exists():
            continue
        config = json.loads(config_file.read_text())
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        quality_file = run / 'quality/summary.json'
        quality = json.loads(quality_file.read_text()) if quality_file.exists() else {}
        rows.append({
            'label': run.name,
            'complete': config.get('complete') is True and summary.get('complete') is True,
            'wall_seconds': summary.get('wall_seconds'),
            'completion_tokens': summary.get('completion_tokens'),
            'generation_tokens_per_second': summary.get('generation_tokens_per_second'),
            'cached_fraction': summary.get('cached_fraction'),
            'expected_requests': summary.get('expected_requests'),
            'successful_requests': summary.get('successful_requests'),
            'failed_requests': summary.get('failed_requests'),
            'missing_requests': summary.get('missing_requests'),
            'length_capped_requests': summary.get('length_capped_requests'),
            'graded_correct': summary.get('graded_correct'),
            'graded_total': summary.get('graded_total'),
            'short_quality_correct': quality.get('quality_correct'),
            'short_quality_total': quality.get('quality_total'),
            'ttft_p50_seconds': summary.get('ttft_p50_seconds'),
            'ttft_p95_seconds': summary.get('ttft_p95_seconds'),
            'binary_sha256': config.get('binary_sha256'),
            'workload_sha256': summary.get('workload_sha256'),
        })
    if rows:
        with (args.results / 'comparison.csv').open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    by_name = {row['label']: row for row in rows}
    pairs = [
        ('control-f16-8-a', 'final-all-f16-8-b'),
        ('control-f16-8-b', 'final-all-f16-8-c'),
        ('qa-control-f16-8', 'qa-all-f16-8'),
        ('cap256-control-f16-8', 'cap256-all-f16-8'),
        ('control-kv8-16', 'all-kv8-16'),
        ('control-kv8-32', 'all-kv8-32'),
    ]
    comparisons = []
    for control, optimized in pairs:
        a, b = by_name.get(control), by_name.get(optimized)
        if not a or not b or not a['complete'] or not b['complete']:
            continue
        def replies(label):
            return {entry['id']: (entry.get('content'), entry.get('reasoning'))
                    for line in (args.results / 'runs' / label / 'trace/calls.jsonl').read_text().splitlines()
                    if (entry := json.loads(line)) and not entry.get('error')}
        ar, br = replies(control), replies(optimized)
        comparisons.append({
            'control': control, 'optimized': optimized,
            'wall_speedup': a['wall_seconds'] / b['wall_seconds'],
            'same_generated_tokens': a['completion_tokens'] == b['completion_tokens'],
            'identical_content_and_reasoning': sum(key in br and value == br[key] for key, value in ar.items()),
            'compared_requests': len(ar),
            'note': 'Exact text matching is an audit, not a general quality grade. Capped traces are not completed tasks.',
        })
    (args.results / 'paired-comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    print(json.dumps({'runs': len(rows), 'complete_runs': sum(row['complete'] for row in rows),
                      'complete_pairs': len(comparisons)}))


if __name__ == '__main__':
    main()
