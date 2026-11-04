#!/usr/bin/env python3
"""Summarize scored real-task trials without treating failures as speedups."""
import argparse
import csv
import json
import math
from pathlib import Path


def load(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def jsonlines(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass  # a concurrently appended final line may be incomplete
    return result


def quantile(values,q):
    values=sorted(x for x in values if x is not None)
    return values[max(0,math.ceil(len(values)*q)-1)] if values else None


def resource(path,start,end):
    if not path.exists():return {}
    with path.open() as f:rows=list(csv.DictReader(f))
    filtered=[]
    for row in rows:
        try:
            if start<=float(row['unix_seconds'])<=end:filtered.append(row)
        except (ValueError,TypeError):pass
    result={}
    for key in ['rss_sum_mb','physical_footprint_sum_mb']:
        values=[]
        for row in filtered:
            try:values.append(float(row[key])/1024)
            except (ValueError,TypeError,KeyError):pass
        result[key.replace('_mb','_gib')]=max(values) if values else None
    return result


def score(details):
    if type(details.get('passed')) is int:return details['passed'],details.get('total')
    items=details.get('checks',[])
    if items and isinstance(items[0],dict):
        return sum(bool(x.get('passed')) for x in items),len(items)
    return None,len(items) if items else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--prefix',default='primary')
    a=p.parse_args();plan=load(a.plan or a.results/'corrected-matrix/plan.json');rows=[]
    for job in plan['jobs']:
        cmd=job['command'];root=Path(cmd[cmd.index('--output')+1]);task=root/'task'
        cfg,s=load(root/'config.json'),load(task/'summary.json')
        complete=cfg.get('complete') is True and bool(s) and not s.get('infrastructure_error')
        passed=complete and s.get('passed') is True
        status=('passed' if passed else 'failed' if complete else
                'infrastructure_error' if s.get('infrastructure_error') or cfg.get('complete') is False else
                'running' if root.exists() else 'pending')
        calls=jsonlines(task/'calls.jsonl');events=jsonlines(task/'events.jsonl')
        starts={};intervals=[];active=set();peak=0
        for event in events:
            if event['event']=='request':
                starts[event['id']]=event['elapsed'];active.add(event['id']);peak=max(peak,len(active))
            elif event['event']=='response':
                active.discard(event['id'])
                if event['id'] in starts:intervals.append((starts[event['id']],event['elapsed']))
        merged=[]
        for lo,hi in sorted(intervals):
            if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
            else:merged.append([lo,hi])
        model_wall=sum(hi-lo for lo,hi in merged)
        good=[c for c in calls if not c.get('error')]
        tokens=sum(c.get('usage',{}).get('completion_tokens',0) for c in good)
        prompts=sum(c.get('usage',{}).get('prompt_tokens',0) for c in good)
        cache=[c.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens') for c in good]
        hit=sum(x for x in cache if x is not None)/prompts if prompts and cache and all(x is not None for x in cache) else None
        correct,total=score(s.get('final_check',{}).get('details',{}))
        check_scores=[score(e.get('result',{}).get('details',{})) for e in events if e['event']=='check']
        best=max((n for n,t in check_scores if n is not None and t==total),default=correct)
        begin=s.get('started_unix',0);end=s.get('finished_unix',float('inf'))
        mr,rr=resource(root/'model-resources.csv',begin,end),resource(root/'run-resources.csv',begin,end)
        row={'label':job['label'],'task':job.get('task'),'mode':job.get('engine_mode'),
             'agents':job.get('agents',s.get('agents')),'status':status,
             'checks_passed':correct,'checks_total':total,'time_to_pass_seconds':s.get('time_to_first_passing_check_seconds') if passed else None,
             'best_checks_passed':best,'run_error':s.get('error'),
             'task_wall_seconds':s.get('wall_seconds'),'output_tokens':tokens,'prompt_tokens':prompts,
             'task_tokens_per_second':s.get('completion_tokens_per_wall_second'),
             'serving_tokens_per_second':tokens/model_wall if model_wall else None,
             'serving_wall_seconds':model_wall,'calls':len(calls),'failed_calls':sum(bool(c.get('error')) for c in calls),
             'tool_errors':s.get('tool_errors'),'length_capped_calls':s.get('length_capped_calls'),
             'peak_inflight_requests':peak,'cache_hit_fraction':hit,
             'call_seconds_p50':quantile([c['seconds'] for c in good],.5),
             'call_seconds_p95':quantile([c['seconds'] for c in good],.95),
             'ttft_seconds_p50':quantile([c.get('ttft_seconds') for c in good],.5),
             'ttft_seconds_p95':quantile([c.get('ttft_seconds') for c in good],.95),
             'peak_model_physical_footprint_gib':mr.get('physical_footprint_sum_gib'),
             'peak_run_physical_footprint_gib':rr.get('physical_footprint_sum_gib'),
             'peak_model_rss_gib':mr.get('rss_sum_gib'),'capture_exitcode':s.get('capture_exitcode')}
        rows.append(row)
    (a.results/(a.prefix+'-summary.json')).write_text(json.dumps(rows,indent=2)+'\n')
    if rows:
        with (a.results/(a.prefix+'-summary.csv')).open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
