#!/usr/bin/env python3
"""Print durable progress without changing a running trial."""
from pathlib import Path
import collections,json,time
base=Path(__file__).resolve().parent/'comparison'
for mode in ['fleet8','vllm8','vllm1']:
 p=base/mode/'run/events.jsonl'
 if not p.exists():print(mode,'not started');continue
 events=[]
 for line in p.read_text().splitlines():
  try:events.append(json.loads(line))
  except json.JSONDecodeError:pass
 if not events:continue
 counts=collections.Counter(e['event'] for e in events)
 rounds=[f"r{e['round']} {e['checksPassed']}/{e['checksTotal']}" for e in events if e['event']=='round_complete']
 calls=[e for e in events if e['event']=='call_end']
 elapsed=events[-1]['elapsed']
 if not any(e['event']=='run_end' for e in events):elapsed+=time.time()-events[-1]['t_unix']
 print(mode,f'{elapsed:.1f}s',', '.join(rounds) or 'initial build',f"calls={len(calls)} output={sum(e.get('output',0) for e in calls)} writes={counts['file_write']}")
 if mode=='fleet8':
  stats=next((e for e in reversed(events) if e['event']=='engine_stats'),{})
  print('  engine', {k:stats.get(k) for k in ['active','queued','kvFreeFraction','sharedPrefixTokens','prefillTokens','generatedTokens']})
 issues=[e for e in events if e['event'] in ['agent_error','infrastructure_error']]
 if issues:print('  errors',issues[-2:])
