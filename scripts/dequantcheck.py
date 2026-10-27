"""Check build/dequant_dump against gguf-py's reference dequantization, for one
tensor of every quant type present in a model.

    python scripts/dequantcheck.py MODEL.gguf [rows]
"""
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

path = sys.argv[1]
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 8
reader = GGUFReader(path)
seen = {}
for t in reader.tensors:
    if t.tensor_type.name not in seen and len(t.shape) == 2:
        seen[t.tensor_type.name] = t
bad = 0
for name, t in sorted(seen.items()):
    cols = int(t.shape[0])
    want = dequantize(t.data, t.tensor_type).reshape(-1, cols)[:rows].astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        subprocess.run(["build/dequant_dump", path, t.name, str(rows), f.name], check=True, capture_output=True)
        got = np.fromfile(f.name, dtype=np.float32).reshape(rows, cols)
    err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)
    ok = err < 1e-6
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name:6s} {t.name:32s} max rel err {err:.2e}")
print("all exact" if not bad else f"{bad} mismatches")
