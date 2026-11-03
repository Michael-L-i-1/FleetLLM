import fcntl,json,os,signal,subprocess,time
from pathlib import Path
h=Path.home(); root=h/'fleetllm'; out=root/'results/pi5/2026-11-02/gpqa-diamond4'
out.mkdir(parents=True,exist_ok=True)
stop=out/'stop'
if stop.exists():stop.unlink()
lock=(h/'.cache/fleetllm-bench.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cmd=[str(root/'build/fleet'),'serve','--model',str(h/'models/Muse-Glimmer-30B-Q2_K.gguf'),'--backend','cpu','--threads','4','--budget-gb','14.8','--kv-gb','1','--row-budget','64','--no-pin','--ngram','--spec-k','15','--spec-rows','16','--host','127.0.0.1','--port','18096','--name','pi-gpqa-glimmer','--reasoning','low','--max-active','4','--admit','--max-tokens','1024']
def memory():
 d={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
 return d['MemAvailable'],d['SwapTotal']-d['SwapFree']
start=time.monotonic();initial_swap=memory()[1];ready_swap=None;peak=loading_growth=timed_growth=low=0;minimum=10**30;reason=None;report=-30;sample=-5
logpath=out/'server.log'
with logpath.open('w') as log,(out/'server-metrics.jsonl').open('w') as metrics:
 proc=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  while proc.poll() is None:
   available,swap=memory();minimum=min(minimum,available);elapsed=time.monotonic()-start
   rss=0
   try:
    for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines():
     if line.startswith('VmRSS:'):rss=int(line.split()[1])*1024
     if line.startswith('VmHWM:'):peak=max(peak,int(line.split()[1])*1024)
   except FileNotFoundError:pass
   if ready_swap is None and '[serve] listening' in logpath.read_text():ready_swap=swap
   if ready_swap is None:loading_growth=max(loading_growth,swap-initial_swap)
   else:timed_growth=max(timed_growth,swap-ready_swap)
   if elapsed-sample>=5:
    metrics.write(json.dumps(dict(timestamp=time.time(),elapsed=elapsed,available_bytes=available,swap_used_bytes=swap,rss_bytes=rss))+'\n');metrics.flush();sample=elapsed
   if elapsed-report>=30:
    print(f'{elapsed:.0f}s, ready={ready_swap is not None}, available={available/1e9:.2f}GB, timed swap growth={timed_growth/1e6:.1f}MB',flush=True);report=elapsed
   low=low+1 if available<400*1024**2 else 0
   if stop.exists():reason='pilot finished';break
   if low>=6 or timed_growth>128*1024**2 or elapsed>5400:
    reason='memory pressure or timeout';break
   time.sleep(.5)
 finally:
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  result=dict(command=cmd,seconds=time.monotonic()-start,returncode=proc.returncode,stop_reason=reason,peak_rss_bytes=peak,minimum_available_bytes=minimum,loading_peak_swap_growth_bytes=loading_growth,timed_peak_swap_growth_bytes=timed_growth,throttle=subprocess.check_output(['vcgencmd','get_throttled'],text=True).strip())
  (out/'server-run.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result),flush=True)
