"""Run isolated Pi CPU experiments with a lock, memory guard and JSONL outputs.

No model downloads or machine configuration changes. Each configuration loads
one model and evaluates fresh sequences so prefix caching cannot bias timings.
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

PROFILES = {
    "exact": [],
    "group256": ["--activation-group", "256"],
    "group256-ngram7": [
        "--activation-group",
        "256",
        "--ngram",
        "--spec-k",
        "7",
        "--spec-rows",
        "8",
    ],
    "group256-draft2": [
        "--activation-group",
        "256",
        "--draft",
        str(Path.home() / "models/glimmer-dflash"),
        "--draft-kv-gb",
        ".064",
        "--spec-k",
        "2",
        "--spec-rows",
        "3",
    ],
    "ngram15": ["--ngram", "--spec-k", "15", "--spec-rows", "16"],
    "ngram7": ["--ngram", "--spec-k", "7", "--spec-rows", "8"],
    "draft2": [
        "--draft",
        str(Path.home() / "models/glimmer-dflash"),
        "--draft-kv-gb",
        ".064",
        "--spec-k",
        "2",
        "--spec-rows",
        "3",
    ],
    "sparse25": ["--ffn-sparsity", ".25"],
    "sparse50": ["--ffn-sparsity", ".5"],
    "skip4": ["--skip-layers", "8,20,32,44"],
    "skip8": ["--skip-layers", "5,11,17,23,29,35,41,47"],
    "skip-half-diagnostic": ["--skip-layers", ",".join(map(str, range(1, 51, 2)))],
}


def memory():
    d = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    return d["MemAvailable"], d["SwapTotal"] - d["SwapFree"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--binary",
        type=Path,
        default=root / "build/fleet",
        help="Fleet dispatcher from this checkout or an isolated comparison build",
    )
    p.add_argument("--profiles", nargs="+", choices=PROFILES, default=["exact"])
    p.add_argument("--tasks", type=Path, default=root / "bench/pi-quality.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model", type=Path, default=Path.home() / "models/Muse-Glimmer-30B-Q2_K.gguf"
    )
    p.add_argument("--ppl", action="store_true")
    p.add_argument("--ppl-tokens", type=int, default=192)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--row-budget", type=int, default=64)
    p.add_argument("--kv-gb", type=float, default=0.25)
    args = p.parse_args()
    if (
        args.threads < 1
        or args.row_budget < 1
        or not math.isfinite(args.kv_gb)
        or args.kv_gb <= 0
    ):
        p.error("threads, row budget and KV budget must be positive")
    args.binary = args.binary.expanduser().resolve()
    args.tasks = args.tasks.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lockpath = Path.home() / ".cache/fleetllm-bench.lock"
    lockpath.parent.mkdir(exist_ok=True)
    with lockpath.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base = [
            "--model",
            str(args.model),
            "--backend",
            "cpu",
            "--threads",
            str(args.threads),
            "--budget-gb",
            "14.8",
            "--kv-gb",
            str(args.kv_gb),
            "--row-budget",
            str(args.row_budget),
            "--no-pin",
        ]
        results = {}
        for name in args.profiles:
            command = [
                str(args.binary),
                "ppl" if args.ppl else "eval",
                *base,
                *PROFILES[name],
            ]
            command += (
                [
                    "--file",
                    str(root / "bench/pi-ppl.txt"),
                    "--max-tokens",
                    str(args.ppl_tokens),
                    "--ctx",
                    "256",
                    "--chunk",
                    "32",
                ]
                if args.ppl
                else ["--tasks", str(args.tasks)]
            )
            start = time.monotonic()
            before = memory()[1]
            peak = growth = low = 0
            minimum = 10**30
            report = -30
            reason = None
            with (
                (
                    args.output / (name + ".jsonl" if not args.ppl else name + ".txt")
                ).open("w") as out,
                (args.output / (name + ".log")).open("w") as err,
            ):
                proc = subprocess.Popen(
                    command, stdout=out, stderr=err, start_new_session=True
                )
                while proc.poll() is None:
                    avail, swap = memory()
                    minimum = min(minimum, avail)
                    growth = max(growth, swap - before)
                    try:
                        for line in (
                            Path(f"/proc/{proc.pid}/status").read_text().splitlines()
                        ):
                            if line.startswith("VmHWM:"):
                                peak = max(peak, int(line.split()[1]) * 1024)
                    except FileNotFoundError:
                        pass
                    elapsed = time.monotonic() - start
                    if elapsed - report >= 30:
                        print(
                            f"{name}: {elapsed:.0f}s, available {avail / 1e9:.2f}GB, swap growth {growth / 1e6:.1f}MB",
                            flush=True,
                        )
                        report = elapsed
                    low = low + 1 if avail < 400 * 1024**2 else 0
                    if low >= 6 or growth > 128 * 1024**2 or elapsed > 3600:
                        reason = "memory pressure or timeout"
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                        break
                    time.sleep(0.5)
                status = proc.wait()
            results[name] = dict(
                command=command,
                seconds=time.monotonic() - start,
                returncode=status,
                stopped=reason,
                peak_rss_bytes=peak,
                minimum_available_bytes=minimum,
                peak_swap_delta_bytes=growth,
            )
            (args.output / "run.json").write_text(json.dumps(results, indent=2))
            print((args.output / (name + ".log")).read_text(), flush=True)
            if status or reason:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
