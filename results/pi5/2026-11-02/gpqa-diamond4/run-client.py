import json,subprocess,time,urllib.request
from pathlib import Path
root=Path.cwd();out=root/'results/pi5/2026-11-02/gpqa-diamond4';remote='/home/mli5/fleetllm/results/pi5/2026-11-02/gpqa-diamond4/stop'
tunnel_command=['ssh','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-N','-L','127.0.0.1:18096:127.0.0.1:18096','mli5@100.85.175.6']
command=[str(root/'evals/.venv/bin/python'),'-u','evals/gpqa.py','--base-url','http://127.0.0.1:18096/v1','--model','pi-gpqa-glimmer','--workers','4','--num-examples','4','--repeats','1','--max-tokens','1024','--temperature','0','--stream','--reasoning-effort','low','--request-timeout','5400','--label','pilot','--out-dir',str(out)]
manifest={'command':command,'tunnel_command':tunnel_command,'model':'Muse-Glimmer-30B-Q2_K.gguf','production_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'harness_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'evals/third_party/simple_evals',text=True).strip(),'note':'Four deterministic GPQA Diamond examples from the pinned standard evaluator; no question or grader changes. Streaming wrapper records final and reasoning channels separately. Current wrapper working-tree source saved with run.'}
(out/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
start=time.monotonic();proc=None
with (out/'tunnel.log').open('w') as tunnel_log,(out/'client.log').open('w') as client_log:
 tunnel=subprocess.Popen(tunnel_command,stdin=subprocess.DEVNULL,stdout=tunnel_log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+180
  while True:
   if tunnel.poll() is not None:raise RuntimeError('SSH tunnel exited')
   try:
    with urllib.request.urlopen('http://127.0.0.1:18096/health',timeout=2):break
   except Exception:
    if time.monotonic()>deadline:raise RuntimeError('server health timeout')
    time.sleep(1)
  print('Pi server healthy; starting four-question reasoning-enabled pilot',flush=True)
  proc=subprocess.Popen(command,cwd=root,stdout=client_log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   elapsed=time.monotonic()-start
   if elapsed>5300:raise RuntimeError('pilot total wall timeout')
   time.sleep(1)
  if proc.returncode:raise RuntimeError(f'GPQA runner exited {proc.returncode}; see client.log')
  print((out/'pilot.json').read_text(),flush=True)
 finally:
  if proc is not None and proc.poll() is None:
   proc.terminate()
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
  subprocess.run(['ssh','-o','BatchMode=yes','mli5@100.85.175.6','touch',remote],timeout=30,check=False)
  tunnel.terminate()
  try:tunnel.wait(timeout=15)
  except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
  (out/'client-run.json').write_text(json.dumps({'seconds':time.monotonic()-start,'returncode':proc.returncode if proc else None},indent=2)+'\n')
