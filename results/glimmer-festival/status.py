import json,time
from pathlib import Path
r=Path('results/glimmer-festival');p=json.loads((r/'matrix/progress.json').read_text());label=p.get('active_label');print(p['status'],len(p['completed']),'/',p['total'],label or '')
if label:
 task=r/'runs'/label/'task';events=[]
 if (task/'events.jsonl').exists():
  for line in (task/'events.jsonl').read_text().splitlines():
   try:events.append(json.loads(line))
   except ValueError:pass
 if events:
  last=events[-1];current=last.get('round');stage=[x for x in events if x.get('round')==current]
  writes={x['path'] for x in stage if x['event']=='file_write'}
  start=events[0]['unix_seconds']-events[0]['elapsed'];print('Round',current,last.get('round_title'),'phase',last.get('phase'),'elapsed',round(time.time()-start,1),'s','page files written in round',len(writes))
  rounds=[x for x in events if x['event']=='round_complete']
  for x in rounds:print('Completed round',x['round'],'passed',x['passed'],'wall',round(x['wall_seconds'],1),'s','tokens',x['completion_tokens'])
  active={x['id'] for x in stage if x['event']=='request'}-{x['id'] for x in stage if x['event']=='response'};print('Active requests',len(active))
