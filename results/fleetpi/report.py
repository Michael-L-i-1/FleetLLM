#!/usr/bin/env python3
"""Derive comparison metrics from frozen raw trials; never edits benchmark workspaces."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
MODES=['fleet8','vllm8','vllm1']
LABELS={'fleet8':'FleetPi + Fleet, 8 agents','vllm8':'Pi + default vLLM, 8 agents','vllm1':'Pi + default vLLM, 1 agent'}
def load(path):return json.loads(path.read_text())
def peak(path):
    rows=list(csv.DictReader(path.open()))
    vals=[float(x['physical_footprint_sum_mb'])/1024 for x in rows if x['physical_footprint_sum_mb']]
    return max(vals) if vals else None

rows=[]
for mode in MODES:
    base=HERE/'comparison'/mode
    result=load(base/'result.json')
    calls=[json.loads(x) for x in (base/'run/calls.jsonl').read_text().splitlines()]
    events=[json.loads(x) for x in (base/'run/events.jsonl').read_text().splitlines()]
    totals=result['totals']
    checker=[load(x) for x in sorted((base/'run/checks').glob('*/outcome.json'))]
    final=result['quality'][-1] if result['quality'] else {}
    timing=load(base/'timing.json')
    frames=[json.loads(x) for x in (base/'capture/frames.jsonl').read_text().splitlines()]
    assert not result.get('infrastructureError'), (mode,result.get('infrastructureError'))
    assert not result['cancelled'],mode
    assert not any(e['event']=='run_resume' for e in events),mode
    assert len([x for x in frames if x.get('reason')=='initial' and x.get('captured')])==1,mode
    assert any(x.get('reason')=='final' and x.get('captured') for x in frames),mode
    assert sum(c['output'] for c in calls)==totals['outputTokens'],mode
    # A shared check is notified to every agent; count its actual immutable outcome once.
    # Frozen initial harness totals over-count shared notifications, so derive this centrally.
    row={'mode':mode,'label':LABELS[mode],'passed':result['passed'] and all(r['checksPassed']==r['checksTotal'] for r in result['quality']),'final_passed':result['passed'],'timed_out':result['timedOut'],
      'checks_passed':final.get('checksPassed',0),'checks_total':final.get('checksTotal',0),
      'task_seconds':totals['wallSeconds'],'output_tokens':totals['outputTokens'],
      'workflow_output_tokens_per_second':totals['outputTokens']/totals['wallSeconds'],
      'engine_peak_footprint_gib':peak(base/'engine-memory.csv'),
      'whole_run_peak_footprint_gib':peak(base/'system-memory.csv'),
      'calls':len(calls),'tool_calls':totals['toolCalls'],'writes':sum(e['event']=='file_write' for e in events),
      'compactions':totals['compactions'],'retries':totals['retries'],
      'length_limited_calls':sum(c['stopReason']=='length' for c in calls),
      'error_calls':sum(c['stopReason'] in ['error','aborted'] for c in calls),
      'median_request_seconds':statistics.median((c['endUnixMs']-c['startUnixMs'])/1000 for c in calls),
      'actual_checker_executions':len(checker),'actual_checker_execution_seconds':sum(c['seconds'] for c in checker),
      'invalid_stream_rate_excluded':True,'reasoning_token_subcount':None,
      'rounds':result['quality'],
      'capture_seconds':timing['finishedUnix']-timing['captureStartUnix'],
      'initial_capture_seconds':load(base/'capture/capture-ready.json')['elapsed'],
      'captured_frames':sum(bool(f.get('captured')) for f in frames),
      'capture_errors':sum(not f.get('captured') for f in frames)}
    # Official final check must describe the actual delivered page bytes.
    last_round=result['quality'][-1]['round']
    official=load(base/f'run/official/round-{last_round}.json')
    for file,digest in official['pageHashes'].items():
        assert hashlib.sha256((base/'run/workspace'/file).read_bytes()).hexdigest()==digest,(mode,file)
    rows.append(row)
fleet=rows[0]
for row in rows:
    row['fleet_task_speedup_vs_mode']=row['task_seconds']/fleet['task_seconds'] if fleet['passed'] and row['passed'] and not row['timed_out'] else None
notes=['One trial per mode; no confidence interval or statistical significance claim.',
 'Exact same local Muse Glimmer30B affine4bit checkpoint. Default vLLM performance features remain enabled, including prefix caching and batching.',
 'Workflow output tokens/second includes actual Pi tools, checks and model calls; it is not GPU decode throughput. Output includes reasoning when counted by the provider; reasoning-specific token counts were not supplied.',
 'Frozen harness stream-rate fields are excluded: buffered native tool responses make first-delta-to-end division invalid. Full task wall and API token counts remain valid.',
 'Checker execution metrics are reconstructed from unique immutable outcome files, correcting repeated per-agent notifications in the frozen harness.',
 'Success requires every staged round to pass, not just the final cumulative score. A single-agent run that misses early milestones is not a comparable speed-to-success result. No cancellation or resume was exercised. Any time limit is reported explicitly; speedups are withheld for unfinished/timed-out modes. Subsequent durability/accounting fixes are tested separately and do not change frozen measurements.',
 'Darwin physical-footprint sums sampled once/second; engine-only and whole owned process-tree peaks include charged GPU memory. Peaks are approximate.',
 'Video uses actual sampled browser screenshots, equal40xplayback and a shared real-time axis; capture clock includes the initial screenshot hook, task wall excludes it.']
(HERE/'summary.json').write_text(json.dumps({'rows':rows,'notes':notes},indent=2)+'\n')
columns=['mode','checks_passed','checks_total','task_seconds','output_tokens','workflow_output_tokens_per_second','engine_peak_footprint_gib','whole_run_peak_footprint_gib','calls','tool_calls','actual_checker_executions','compactions','retries','fleet_task_speedup_vs_mode']
with (HERE/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
config={'title':'Glimmer four-bit · real Pi agents · eight-page website, three rounds','speed':40,'sample_seconds':10,'fps':12,
 'lanes':[{'label':row['label'],'capture_dir':str(HERE/'comparison'/row['mode']/'capture'),'duration_seconds':row['capture_seconds'],
 'status':f"{'passed' if row['passed'] else 'early rounds failed; final'} {row['checks_passed']}/{row['checks_total']}",
 'metrics':f"Task {row['task_seconds']:.1f}s · {row['workflow_output_tokens_per_second']:.1f} output tok/s · {row['engine_peak_footprint_gib']:.2f} GiB engine"} for row in rows]}
(HERE/'video-config.json').write_text(json.dumps(config,indent=2)+'\n')
print(json.dumps(rows,indent=2))
