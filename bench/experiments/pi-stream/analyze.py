"""Summarize rotated packed-weight prefetch timings, without pooling unlike tensors."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
groups = {}
for row in csv.DictReader(args.csv.open()):
    assert row["bitwise_equal"] == "1", row
    key = (row["tensor"], row["type"], int(row["batch"]))
    groups.setdefault(key, {}).setdefault(row["variant"], []).append(
        float(row["milliseconds"])
    )
result = []
for (tensor, kind, batch), variants in groups.items():
    baseline = median(variants["baseline"])
    result.append(
        dict(
            tensor=tensor,
            kind=kind,
            batch=batch,
            baseline_ms=baseline,
            variants={
                name: dict(
                    median_ms=median(values),
                    samples=len(values),
                    speedup=baseline / median(values),
                )
                for name, values in variants.items()
            },
        )
    )
text = json.dumps(result, indent=2) + "\n"
if args.output:
    args.output.write_text(text)
else:
    print(text, end="")
