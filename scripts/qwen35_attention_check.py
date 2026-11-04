#!/usr/bin/env python3
"""Compare optional HD256 NAX prefill against Fleet's existing vector attention.

Reports last-token logits, not model-task accuracy. No new weights or Python
inference runtime is used. NumPy is a development-only comparison dependency.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--lengths', default='31,32,33,63,64,65,4095,4096,4097')
a = p.parse_args()
records = []
with tempfile.TemporaryDirectory(prefix='fleet-attn256-check-') as td:
    for words in map(int, a.lengths.split(',')):
        prompt = ' '.join(['x']*words)
        for kv8 in (False, True):
            values = []
            for nax in (False, True):
                file = Path(td)/'logits.bin'
                cmd = [str(ROOT/'build/fleet-model-probe'), '--model', 'qwen35-4b-mlx4', '--device', 'metal',
                       '--budget-gb', '10', '--kv-gb', '4', '--chunk', '128', '--prompt', prompt,
                       '--last-only', '--output', str(file)]
                if kv8:
                    cmd.append('--kv-int8')
                env = dict(os.environ, FLEET_Q35_NAX_ATTN='1' if nax else '0', FLEET_NAX_SPAN='64' if words<100 else '4096')
                run = subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,check=True)
                info = json.loads(run.stdout)
                logits = np.fromfile(file,np.float32)
                assert np.isfinite(logits).all()
                values.append((info,logits,run.stderr))
            assert values[0][0]['tokens'] == values[1][0]['tokens']
            old,new = values[0][1],values[1][1]
            def logprob(x):
                z = x.astype(np.float64)-x.max()
                return z-np.log(np.exp(z).sum())
            lp,lq = logprob(old),logprob(new)
            kl = float((np.exp(lp)*(lp-lq)).sum())
            row = {'words':words,'tokens':len(values[0][0]['tokens']),'kv_int8':kv8,
                   'mean_abs_error':float(np.abs(old-new).mean()),'max_abs_error':float(np.abs(old-new).max()),
                   'kl_old_to_new':kl,'argmax_same':int(old.argmax())==int(new.argmax()),
                   'vec_log':values[0][2],'nax_log':values[1][2]}
            row['passed'] = row['mean_abs_error']<.10 and kl<.01
            records.append(row)
            print(words,kv8,'mae',row['mean_abs_error'],'KL',kl,'argmax',row['argmax_same'],flush=True)
result = {'passed':all(x['passed'] for x in records),'cases':records,
          'note':'Same attention semantics, different floating-point rounding; no general coding-quality claim.'}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
raise SystemExit(0 if result['passed'] else 1)
