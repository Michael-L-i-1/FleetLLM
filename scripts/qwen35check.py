#!/usr/bin/env python3
"""Compare native Fleet Qwen3.5 logits to MLX-LM on the SAME pinned checkpoint.

Development-only dependencies: mlx-lm and numpy. Inference remains native C/Metal.
Run with a Python environment containing those packages after make fleet-model-probe.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import tempfile

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference-f32', action='store_true', help='Match Fleet activation precision instead of checkpoint bf16')
a = p.parse_args()
model, tokenizer = load(str(a.model))
if a.reference_f32:
    model.set_dtype(mx.float32)
cases = [
    'The capital of France is',
    '<|im_start|>user\nWhat is 2 + 2? Reply with only the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n4',
    'def clamp(value, low, high):\n    """Return value constrained to the inclusive bounds."""\n    return max(low, min(value, high))\n\nassert clamp(9, 0, 5) == 5\n',
    'A page contains search filters and a signup form. Preserve the hidden attribute and accessible labels. ' * 5,
]
report = {'reference_f32': a.reference_f32, 'model': str(a.model), 'cases': []}
with tempfile.TemporaryDirectory(prefix='fleet-qwen35-check-') as td:
    for index, prompt in enumerate(cases):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        ref = model(mx.array([ids]))
        mx.eval(ref)
        ref = np.asarray(ref[0].astype(mx.float32))
        case = {'prompt': prompt, 'tokens': ids, 'checks': []}
        for device, chunk, kv8 in [('cpu-ref', 128, False), ('cpu', 128, False), ('metal', 128, False), ('metal', 1, False), ('metal', 17, False), ('metal', 17, True)]:
            out = Path(td) / 'logits.bin'
            cmd = [str(ROOT/'build/fleet-model-probe'), '--model', str(a.model), '--device', device,
                   '--budget-gb', '8', '--kv-gb', '.5', '--prompt', prompt, '--chunk', str(chunk), '--output', str(out)]
            if kv8:
                cmd.append('--kv-int8')
            run = subprocess.run(cmd, capture_output=True, text=True, check=True)
            meta = json.loads(run.stdout)
            assert meta['tokens'] == ids, 'Tokenizer differs'
            actual = np.fromfile(out, np.float32).reshape(ref.shape)
            assert np.isfinite(actual).all(), 'Non-finite logits'
            delta = np.abs(ref - actual)
            match = float((ref.argmax(-1) == actual.argmax(-1)).mean())
            def logsoftmax(x):
                z=x.astype(np.float64)-x.max(-1,keepdims=True)
                return z-np.log(np.exp(z).sum(-1,keepdims=True))
            lp,la=logsoftmax(ref),logsoftmax(actual)
            kl=(np.exp(lp)*(lp-la)).sum(-1)
            item = {'kl_mean': float(kl.mean()), 'kl_max': float(kl.max()), 'device': device, 'chunk': chunk, 'kv_int8': kv8, 'mae': float(delta.mean()),
                    'max_abs': float(delta.max()), 'argmax_match': match, 'stderr': run.stderr}
            # bf16 MLX activations differ from Fleet f32/f16; compare numerical error AND choices.
            item['strict_logit_passed'] = item['mae'] < .16 and item['max_abs'] < 2.0 and match >= .85
            item['passed'] = item['strict_logit_passed']
            if device == 'cpu':
                # Optimized CPU quantizes activations to q8; bound distribution error explicitly.
                item['passed'] = item['mae'] < .16 and item['kl_mean'] < .005 and item['kl_max'] < .05 and match >= .95
            case['checks'].append(item)
            print(index, device, chunk, kv8, item['mae'], match, flush=True)
        report['cases'].append(case)
report['passed'] = all(x['passed'] for c in report['cases'] for x in c['checks'])
report['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()}
                   for p in a.model.iterdir() if p.is_file()}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
raise SystemExit(0 if report['passed'] else 1)
