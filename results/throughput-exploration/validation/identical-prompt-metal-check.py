import json, os, pathlib, subprocess, time, urllib.request
root=pathlib.Path.cwd(); out=root/'results/throughput-exploration/validation'; base='http://127.0.0.1:8090'
try: urllib.request.urlopen(base+'/health',timeout=1)
except OSError:pass
else:raise RuntimeError('Refusing occupied port')
model=pathlib.Path.home()/'.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-4bit/snapshots/0e7ffd5c629ef7719d4cbc04069232580bfa9d9c'
cmd=['build/fleet-serve','--model',str(model),'--device','metal','--name','qwen35-4b-mlx4','--budget-gb','16','--kv-gb','4','--max-active','8','--admit','--row-budget','128','--prompt-checkpoints','--recurrent-prefix-dedup','--recurrent-state-move','--recurrent-cache-priority','--port','8090']
results=[]
with (out/'identical-prompt-metal-server.txt').open('w') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,FLEET_Q35_NAX_ATTN='1',FLEET_TRACE_STATS='1'))
 try:
  start=time.monotonic()
  while True:
   if p.poll() is not None:raise RuntimeError('Server exited')
   try:
    urllib.request.urlopen(base+'/health',timeout=1);break
   except OSError:
    if time.monotonic()-start>60:raise
    time.sleep(.2)
  body={'model':'qwen35-4b-mlx4','messages':[{'role':'system','content':'Answer exactly with the requested word.'},{'role':'user','content':'Reference notes:\n'+''.join(f'Entry {i}: a harmless cache regression fixture.\n' for i in range(200))+'\nThe secret marker is PURPLE. Reply with only the secret marker.'}],'max_tokens':16,'temperature':0,'reasoning_effort':'none'}
  for i in range(4):
   req=urllib.request.Request(base+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
   d=json.load(urllib.request.urlopen(req,timeout=60)); results.append(d)
  (out/'identical-prompt-metal.json').write_text(json.dumps({'command':cmd,'responses':results},indent=2)+'\n')
  for i,d in enumerate(results):
   c=d['choices'][0];cached=d['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)
   assert c['message']['content'].strip()=='PURPLE',c
   assert c['finish_reason']=='stop',c
   assert i==0 or cached>0,(i,cached)
  print('PASS: four identical prompts; correct complete answers; all three repeat requests reuse cached prompt state')
  print([d['usage'] for d in results])
 finally:
  p.terminate();p.wait(timeout=10)
