"""One cold, one-agent llama.cpp request, same formatted prompt as Fleet generate."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
h=Path.home(); out=h/'glimmer-q2k-llama'; out.mkdir(exist_ok=True)
text='Explain how a rainbow forms, step by step. Discuss sunlight, refraction, internal reflection, and why the colors separate. Give a detailed explanation in plain English.'
prompt=('<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n\nReasoning strength: high.\n\n# Valid recipients: "self", "user".<|eot|><|start|>user<|message|>'+text+'<|eot|><|start|>assistant to=user<|message|>')
cmd=[str(h/'llama.cpp/build/bin/llama-server'),'-m',str(h/'models/Muse-Glimmer-30B-Q2_K.gguf'),'--host','127.0.0.1','--port','18092','-t','4','-tb','4','-np','1','-c','512','-ngl','0','-cram','0','--no-warmup']
with (out/'server.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+90
  while True:
   if proc.poll() is not None: raise RuntimeError('server exited')
   try:
    with urllib.request.urlopen('http://127.0.0.1:18092/health',timeout=1): break
   except Exception:
    if time.monotonic()>deadline: raise RuntimeError('startup timeout')
    time.sleep(.2)
  body=dict(model='glimmer',prompt=prompt,max_tokens=32,temperature=0,seed=42,stream=False,cache_prompt=False)
  req=urllib.request.Request('http://127.0.0.1:18092/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
  start=time.monotonic()
  with urllib.request.urlopen(req,timeout=600) as r: response=json.load(r)
  result=dict(command=cmd,request=body,seconds=time.monotonic()-start,response=response)
  (out/'result.json').write_text(json.dumps(result,indent=2))
  print(json.dumps(result,indent=2),flush=True)
 finally:
  proc.terminate()
  try: proc.wait(timeout=15)
  except subprocess.TimeoutExpired: proc.kill();proc.wait()
