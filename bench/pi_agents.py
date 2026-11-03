"""Guarded first concurrent-request trial on the provisioned Pi (not coordination)."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def memory():
    values = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    return values["MemAvailable"], values["SwapTotal"] - values["SwapFree"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--source", type=Path)
    p.add_argument("--tasks", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model", type=Path, default=Path.home() / "models/Muse-Glimmer-30B-Q2_K.gguf"
    )
    p.add_argument("--gen", type=int, default=32)
    a = p.parse_args()
    a.root = a.root.expanduser().resolve()
    a.source = (a.source or a.root / "bench/agent_trial.c").resolve()
    a.tasks = (a.tasks or a.root / "bench/pi-agents.jsonl").resolve()
    if not 2 <= a.gen <= 4096:
        p.error("--gen must be 2..4096")
    a.output.mkdir(parents=True, exist_ok=True)
    lockpath = Path.home() / ".cache/fleetllm-bench.lock"
    lockpath.parent.mkdir(exist_ok=True)
    with lockpath.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binary = a.root / "build/agent-trial"
        compile_command = [
            "cc",
            "-O3",
            "-mcpu=native",
            "-D_GNU_SOURCE",
            "-Itools",
            "-Isrc",
            str(a.source),
            "build/libfleetllm.a",
            "-lpthread",
            "-lm",
            "-o",
            str(binary),
        ]
        subprocess.run(compile_command, cwd=a.root, check=True)
        command = [
            str(binary),
            "--tasks",
            str(a.tasks),
            "--model",
            str(a.model),
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
            "--ngram",
            "--spec-k",
            "15",
            "--spec-rows",
            "16",
            "--gen",
            str(a.gen),
        ]
        start = time.monotonic()
        initial_swap = memory()[1]
        peak = loading_growth = timed_growth = low = 0
        minimum = 10**30
        ready_swap = ready_at = None
        report = -30
        reason = None
        logfile = a.output / "trial.log"
        with (
            (a.output / "results.jsonl").open("w") as stdout,
            logfile.open("w") as stderr,
        ):
            proc = subprocess.Popen(
                command,
                cwd=a.root,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                while proc.poll() is None:
                    available, swap = memory()
                    minimum = min(minimum, available)
                    try:
                        for line in (
                            Path(f"/proc/{proc.pid}/status").read_text().splitlines()
                        ):
                            if line.startswith("VmHWM:"):
                                peak = max(peak, int(line.split()[1]) * 1024)
                    except FileNotFoundError:
                        pass
                    elapsed = time.monotonic() - start
                    if ready_swap is None and "[trial-ready]" in logfile.read_text():
                        ready_swap, ready_at = swap, elapsed
                    if ready_swap is None:
                        loading_growth = max(loading_growth, swap - initial_swap)
                    else:
                        timed_growth = max(timed_growth, swap - ready_swap)
                    if elapsed - report >= 30:
                        print(
                            f"{elapsed:.0f}s, {'loading/warmup' if ready_swap is None else 'timed trial'}, available {available / 1e9:.2f}GB, timed swap growth {timed_growth / 1e6:.1f}MB",
                            flush=True,
                        )
                        report = elapsed
                    low = low + 1 if available < 400 * 1024**2 else 0
                    if low >= 6 or timed_growth > 128 * 1024**2 or elapsed > 900:
                        reason = "memory pressure or timeout"
                        raise RuntimeError(reason)
                    time.sleep(0.5)
                if proc.returncode:
                    raise RuntimeError(f"trial failed: {proc.returncode}")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                metadata = dict(
                    command=command,
                    compile=compile_command,
                    seconds=time.monotonic() - start,
                    returncode=proc.returncode,
                    stopped=reason,
                    peak_rss_bytes=peak,
                    minimum_available_bytes=minimum,
                    loading_peak_swap_growth_bytes=loading_growth,
                    timed_peak_swap_growth_bytes=timed_growth,
                    ready_seconds=ready_at,
                    source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                    tasks_sha256=hashlib.sha256(a.tasks.read_bytes()).hexdigest(),
                    protocol="1/4/8 independent concurrent requests; distinct prompts; fixed token cap ignoring EOS; default target precision; no DFlash/pruning; shared prefix and verified ngram15; no retained KV across levels; one four-token warmup before timing",
                )
                (a.output / "run.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
        print(logfile.read_text(), flush=True)
        for row in map(
            json.loads, (a.output / "results.jsonl").read_text().splitlines()
        ):
            print(
                {key: value for key, value in row.items() if key != "outputs"},
                flush=True,
            )


if __name__ == "__main__":
    main()
