import json
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

home = Path.home()
out = home / "glimmer-cpu-dflash-smoke"
out.mkdir(exist_ok=True)
base = [str(home / "fleetllm/build/fleet-generate"), "--model",
        str(home / "models/Muse-Glimmer-30B-UD-Q2_K_XL.gguf"),
        "--prompt", "What is the capital of France? Answer in one sentence.",
        "--n", "16", "--temp", "0", "--threads", "4", "--budget-gb", "14.8",
        "--kv-gb", "0.25", "--row-budget", "8", "--no-pin"]

def memory():
    d = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        d[key] = int(value.strip().split()[0]) * 1024
    return d["MemAvailable"], d["SwapTotal"] - d["SwapFree"]

parser = argparse.ArgumentParser()
parser.add_argument("--only-dflash", action="store_true")
args = parser.parse_args()
metadata = json.loads((out / "run.json").read_text()) if args.only_dflash else {}
runs = [("plain", []), ("dflash", ["--draft", str(home / "models/glimmer-dflash"),
                        "--draft-kv-gb", "0.064", "--spec-k", "3", "--spec-rows", "4"])]
for label, extra in runs[1:] if args.only_dflash else runs:
    command = [*base, *extra]
    before_swap = memory()[1]
    start = time.monotonic()
    minimum, low, last_report, peak_rss, peak_swap = 10**20, 0, -30, 0, 0
    with (out / f"{label}.txt").open("w") as stdout, (out / f"{label}.log").open("w") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        reason = None
        while proc.poll() is None:
            available, swap = memory()
            minimum = min(minimum, available)
            peak_swap = max(peak_swap, swap - before_swap)
            try:
                for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        peak_rss = max(peak_rss, int(line.split()[1]) * 1024)
            except FileNotFoundError:
                pass
            elapsed = time.monotonic() - start
            low = low + 1 if available < 400 * 1024**2 else 0
            if elapsed - last_report >= 30:
                print(f"{label}: {elapsed:.0f}s, available {available/1e9:.2f} GB, swap delta {(swap-before_swap)/1e6:.0f} MB", flush=True)
                last_report = elapsed
            if low >= 6 or swap - before_swap > 128 * 1024**2 or elapsed > 900:
                reason = "memory pressure or timeout"
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                break
            time.sleep(.5)
        status = proc.wait()
    metadata[label] = dict(command=command, seconds=time.monotonic()-start,
                           minimum_available_bytes=minimum, peak_rss_bytes=peak_rss, peak_swap_delta_bytes=peak_swap,
                           returncode=status, stopped=reason)
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    print(label, "exit", status, flush=True)
    print((out / f"{label}.txt").read_text(), flush=True)
    print((out / f"{label}.log").read_text(), flush=True)
    if status or reason:
        raise SystemExit(1)
same = (out / "plain.txt").read_bytes() == (out / "dflash.txt").read_bytes()
metadata["identical_text"] = same
(out / "run.json").write_text(json.dumps(metadata, indent=2))
print("Greedy output text identical:", same, flush=True)
raise SystemExit(0 if same else 2)
