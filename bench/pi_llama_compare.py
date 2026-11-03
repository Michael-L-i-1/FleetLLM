"""Compare a saved fleet-eval request to llama.cpp using identical input token IDs.

Run on the Pi. Uses llama.cpp's native /completion API to avoid BOS/template
ambiguity. The shared library/server is only a benchmark dependency.
"""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--task", default="rainbow")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    row = next(
        r
        for r in map(json.loads, a.record.read_text().splitlines())
        if r["id"] == a.task
    )
    a.output.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(Path.home() / "llama.cpp/build/bin/llama-server"),
        "-m",
        str(a.model),
        "--host",
        "127.0.0.1",
        "--port",
        "18094",
        "-t",
        "4",
        "-tb",
        "4",
        "-np",
        "1",
        "-c",
        "512",
        "-ngl",
        "0",
        "-cram",
        "0",
        "--no-warmup",
    ]
    lockpath = Path.home() / ".cache/fleetllm-bench.lock"
    lockpath.parent.mkdir(exist_ok=True)
    with lockpath.open("w") as lock, (a.output / "server.log").open("w") as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("llama.cpp server exited")
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18094/health", timeout=1
                    ):
                        break
                except Exception:
                    if time.monotonic() > deadline:
                        raise RuntimeError("startup timeout")
                    time.sleep(0.2)
            body = dict(
                prompt=row["input_tokens"],
                n_predict=row["generated_tokens"],
                temperature=0,
                seed=42,
                stream=False,
                cache_prompt=False,
                return_tokens=True,
            )
            req = urllib.request.Request(
                "http://127.0.0.1:18094/completion",
                json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            )
            start = time.monotonic()
            with urllib.request.urlopen(req, timeout=600) as response:
                result = json.load(response)
            elapsed = time.monotonic() - start
            assert result["timings"]["prompt_n"] == len(row["input_tokens"])
            assert result["timings"]["predicted_n"] == row["generated_tokens"]
            record = dict(
                command=cmd,
                request=body,
                request_seconds=elapsed,
                response=result,
                fleet=row,
                same_text=result["content"] == row["text"],
            )
            (a.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
            print(
                json.dumps(
                    dict(
                        seconds=elapsed,
                        timings=result["timings"],
                        same_text=record["same_text"],
                    )
                ),
                flush=True,
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
