import hashlib,json,subprocess,time
from pathlib import Path
r=Path('results/engine-comparison/tuning');plan=json.loads((r/'plan.json').read_text());done=[]
for job in plan['jobs']:
 cmd=job['command'];cfg=Path(cmd[cmd.index('--server-config')+1]);assert hashlib.sha256(cfg.read_bytes()).hexdigest()==job['config_sha256']
 (r/'progress.json').write_text(json.dumps({'status':'running','active_label':job['label'],'completed':done,'total':len(plan['jobs'])},indent=2)+'\n')
 t=time.monotonic()
 with (r/(job['label']+'.log')).open('w') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 done.append({'label':job['label'],'returncode':code,'driver_wall_seconds':time.monotonic()-t});print(done[-1],flush=True)
(r/'progress.json').write_text(json.dumps({'status':'complete','active_label':None,'completed':done,'total':len(plan['jobs'])},indent=2)+'\n')
