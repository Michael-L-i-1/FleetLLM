#!/usr/bin/env python3
import hashlib,json,subprocess,time
from pathlib import Path
root=Path('results/real-agent-tasks');m=root/'ablation';plan=json.loads((m/'plan.json').read_text());completed=[]
for job in plan['jobs']:
 cmd=job['command'];cfg=Path(cmd[cmd.index('--config')+1]);task=Path(cmd[cmd.index('--task')+1])
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==job['config_sha256']
 for name,sha in job['task_file_hashes'].items():assert hashlib.sha256((task/name).read_bytes()).hexdigest()==sha,name
 (m/'progress.json').write_text(json.dumps({'status':'running','active_label':job['label'],'completed':completed,'total':len(plan['jobs'])},indent=2)+'\n')
 t=time.monotonic()
 with (m/(job['label']+'.log')).open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 out=Path(cmd[cmd.index('--output')+1]);record={'label':job['label'],'driver_returncode':rc,'driver_wall_seconds':time.monotonic()-t}
 if (out/'task/summary.json').exists():
  s=json.loads((out/'task/summary.json').read_text());record.update({k:s.get(k) for k in ['passed','wall_seconds','time_to_first_passing_check_seconds','completion_tokens','completion_tokens_per_wall_second','calls','failed_calls','tool_errors']})
 completed.append(record);print(json.dumps(record),flush=True)
 (m/'progress.json').write_text(json.dumps({'status':'running','active_label':None,'completed':completed,'total':len(plan['jobs'])},indent=2)+'\n')
(m/'progress.json').write_text(json.dumps({'status':'complete','active_label':None,'completed':completed,'total':len(plan['jobs'])},indent=2)+'\n')
