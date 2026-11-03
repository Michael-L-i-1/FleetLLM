"""One-agent Glimmer Q2_K CPU tuning; run on the Pi. Logs timings and memory."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cases', nargs='+', default=['plain-t4', 'plain-t3', 'plain-t2', 'draft-k1', 'draft-k3'])
p.add_argument('--tokens', type=int, default=32)
p.add_argument('--prompt', default='Explain how a rainbow forms, step by step. Discuss sunlight, refraction, internal reflection, and why the colors separate. Give a detailed explanation in plain English.')
p.add_argument('--output', type=Path, default=Path.home() / 'glimmer-q2k-tests')
a = p.parse_args()
a.output.mkdir(exist_ok=True, parents=True)
home = Path.home()
prompt = a.prompt
base = [str(home / 'fleetllm/build/fleet-generate'), '--model', str(home / 'models/Muse-Glimmer-30B-Q2_K.gguf'), '--prompt', prompt, '--n', str(a.tokens), '--temp', '0', '--budget-gb', '14.8', '--kv-gb', '.25', '--row-budget', '64', '--no-pin']
def mem():
    d = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    return d['MemAvailable'], d['SwapTotal'] - d['SwapFree']
results = json.loads((a.output / 'run.json').read_text()) if (a.output / 'run.json').exists() else {}
for case in a.cases:
    threads = case[-1] if case.startswith('plain-t') else '4'
    extra = ['--threads', threads]
    if case.startswith('draft-k'):
        k = int(case.split('k')[-1])
        extra += ['--draft', str(home / 'models/glimmer-dflash'), '--draft-kv-gb', '.064', '--spec-k', str(k), '--spec-rows', str(k+1)]
    if case == 'int8': extra += ['--kv-int8']
    if case == 'reference': extra += ['--backend', 'cpu-ref']
    command = base + extra
    start = time.monotonic(); before = mem()[1]; peak = growth = 0; minimum = 10**30; low = 0; report = -30; reason = None
    with (a.output / (case + '.txt')).open('w') as out, (a.output / (case + '.log')).open('w') as err:
        proc = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True, env=dict(os.environ, FLEET_PROFILE='1'))
        while proc.poll() is None:
            available, swap = mem(); minimum = min(minimum, available); growth = max(growth, swap-before)
            try:
                for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines():
                    if line.startswith('VmHWM:'): peak = max(peak, int(line.split()[1])*1024)
            except FileNotFoundError: pass
            elapsed = time.monotonic()-start
            if elapsed-report >= 30:
                print(f'{case}: {elapsed:.0f}s; available {available/1e9:.2f}GB; swap growth {growth/1e6:.1f}MB', flush=True); report=elapsed
            low = low+1 if available < 400*1024**2 else 0
            if low >= 6 or growth > 128*1024**2 or elapsed > 900:
                reason='memory pressure or timeout'; os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(.5)
        status=proc.wait()
    results[case] = dict(command=command, seconds=time.monotonic()-start, returncode=status, stopped=reason, peak_rss_bytes=peak, minimum_available_bytes=minimum, peak_swap_delta_bytes=growth)
    (a.output / 'run.json').write_text(json.dumps(results, indent=2))
    print((a.output / (case+'.log')).read_text(), flush=True)
    if status or reason: raise SystemExit(1)
