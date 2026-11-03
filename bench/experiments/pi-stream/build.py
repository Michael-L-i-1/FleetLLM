"""Build isolated, compile-time packed-weight prefetch variants for a Pi experiment.

No production source or inference default is changed. Artifacts go under build/.
"""

import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
build = root / "build" / "stream"
build.mkdir(parents=True, exist_ok=True)
source = (root / "src/cpuq.c").read_text()
marker = """            for (int g = 0; g < G; g++) {
                if (sparse) {"""
assert source.count(marker) == 1
public = [
    "cq_kind_name",
    "cq_from_qmat",
    "cq_free",
    "cq_bytes",
    "cq_check",
    "cq_quantize",
    "cq_quantize_group",
    "cq_isa",
    "cq_matmul_rows",
]
variants = [("baseline", 0, 3)] + [
    (f"pf{distance}l{locality}", distance, locality)
    for locality in (0, 3)
    for distance in (4, 8, 16, 32)
]
objects = []
for name, distance, locality in variants:
    injection = """            for (int g = 0; g < G; g++) {
                if (!sparse && a->n == 1 && (g & 1) == 0 && g + DISTANCE < G) {
                    size_t ahead = ((size_t)rg * G + g + DISTANCE) * 4;
                    __builtin_prefetch(w->codes + ahead * code_bytes(kind), 0, LOCALITY);
                    if (kind == CQ2K || kind == CQ3K || kind == CQ4K || kind == CQ6K)
                        __builtin_prefetch(w->sub + ahead * 2, 0, LOCALITY);
                }
                if (sparse) {""".replace("DISTANCE", str(distance)).replace(
        "LOCALITY", str(locality)
    )
    text = source if distance == 0 else source.replace(marker, injection)
    path = build / f"{name}.c"
    path.write_text(text)
    obj = build / f"{name}.o"
    cmd = ["cc", "-O3", "-std=c11", "-mcpu=native", "-D_GNU_SOURCE", "-Isrc"]
    cmd += [f"-D{symbol}={name}_{symbol}" for symbol in public]
    cmd += ["-c", str(path), "-o", str(obj)]
    subprocess.run(cmd, cwd=root, check=True)
    objects.append(str(obj))
subprocess.run(["make", "NO_METAL=1", "-j4"], cwd=root, check=True)
subprocess.run(
    [
        "cc",
        "-O3",
        "-std=c11",
        "-mcpu=native",
        "-D_GNU_SOURCE",
        "-Isrc",
        "bench/experiments/pi-stream/measure.c",
        *objects,
        "build/libfleetllm.a",
        "-lpthread",
        "-lm",
        "-o",
        str(build / "measure"),
    ],
    cwd=root,
    check=True,
)
(build / "variants.json").write_text(json.dumps(variants, indent=2) + "\n")
