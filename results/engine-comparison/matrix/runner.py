import hashlib,json,subprocess,time
from pathlib import Path
r=Path('results/engine-comparison/matrix');plan=json.loads((r/'plan.json').read_text());done=[]
for job in plan['jobs']:
 cmd=job['command']; cfg=Path(cmd[cmd.index('--server-config')+1]);work=Path(cmd[cmd.index('--workload')+1])
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==job['config_sha256']
 assert hashlib.sha256(work.read_bytes()).hexdigest()==job['workload_file_sha256']
 (r/'progress.json').write_text(json.dumps({'status':'running','active_label':job['label'],'completed':done,'total':len(plan['jobs'])},indent=2)+'\n')
 start=time.monotonic()
 with (r/(job['label']+'.log')).open('w') as log: code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 done.append({'label':job['label'],'returncode':code,'driver_wall_seconds':time.monotonic()-start})
 (r/'progress.json').write_text(json.dumps({'status':'running','active_label':None,'completed':done,'total':len(plan['jobs'])},indent=2)+'\n')
 print(done[-1],flush=True)
(r/'progress.json').write_text(json.dumps({'status':'complete','active_label':None,'completed':done,'total':len(plan['jobs']),'all_drivers_passed':all(j['returncode']==0 for j in done)},indent=2)+'\n')
