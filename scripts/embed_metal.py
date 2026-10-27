"""Flatten a .metal file's quoted includes and emit it as a C byte array.

    python3 scripts/embed_metal.py src/metal/kernels.metal NAME > build/NAME.h

Quoted includes resolve relative to the including file, except
"mlx/backend/metal/kernels/..." which resolves to src/metal/mlx/ (vendored MLX
headers). Each file is inlined once (headers use #pragma once).
"""
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "metal")
MLX_PREFIX = "mlx/backend/metal/kernels/"
seen = set()


def resolve(name, cur_dir):
    if name.startswith(MLX_PREFIX):
        return os.path.normpath(os.path.join(ROOT, "mlx", name[len(MLX_PREFIX):]))
    return os.path.normpath(os.path.join(cur_dir, name))


def flatten(path):
    path = os.path.normpath(path)
    if path in seen:
        return ""
    seen.add(path)
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r'\s*#\s*include\s+"([^"]+)"', line)
        if m:
            out.append(flatten(resolve(m.group(1), os.path.dirname(path))))
        elif re.match(r"\s*#\s*pragma\s+once", line):
            continue
        else:
            out.append(line)
    return "".join(out)


src, name = sys.argv[1], sys.argv[2]
data = flatten(src).encode()
print(f"static const unsigned char {name}[] = {{")
for i in range(0, len(data), 16):
    print("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 16]) + ",")
print("};")
print(f"static const unsigned int {name}_len = {len(data)};")
