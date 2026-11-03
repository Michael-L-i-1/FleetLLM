"""Run isolated Pi CPU experiments with a lock, memory guard and JSONL outputs.

No model downloads or machine configuration changes. Each configuration loads
one model and evaluates fresh sequences so prefix caching cannot bias timings.
"""

import argparse
import fcntl
import json
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
    p.add_argument("--profiles", nargs="+", choices=PROFILES, default=["exact"])
    p.add_argument(
        "--tasks", type=Path, default=Path.home() / "fleetllm/bench/pi-quality.jsonl"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model", type=Path, default=Path.home() / "models/Muse-Glimmer-30B-Q2_K.gguf"
    )
    p.add_argument("--ppl", action="store_true")
    p.add_argument("--ppl-tokens", type=int, default=192)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lockpath = Path.home() / ".cache/fleetllm-bench.lock"
    lockpath.parent.mkdir(exist_ok=True)
    with lockpath.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path.home() / "fleetllm"
        base = [
            "--model",
            str(args.model),
            "--backend",
            "cpu",
            "--threads",
            "4",
            "--budget-gb",
            "14.8",
            "--kv-gb",
            ".25",
            "--row-budget",
            "64",
            "--no-pin",
        ]
        results = {}
        for name in args.profiles:
            command = [
                str(root / "build/fleet"),
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
