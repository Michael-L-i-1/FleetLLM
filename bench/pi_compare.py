"""Run matched HTTP completion workloads on an already provisioned Raspberry Pi.

Uses Python's standard library. Starts one localhost-only engine at a time and
stops it on exit. Run on the Pi from ~/fleetllm; raw results go to --output.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request


def request(port, prompt, limit, automatic_bos=False):
    if automatic_bos:
        prompt = prompt.removeprefix("<|begin_of_text|>")
    body = dict(model="llama1b", prompt=prompt, max_tokens=limit,
                temperature=0, seed=42, stream=False, cache_prompt=False)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def prompt(task, repetition, index):
    # Distinct early request labels limit reuse across requests and repetitions.
    return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Trial {repetition}, request {index}.\n{task.strip()}\n\n"
            "Give a very detailed explanation of at least 2000 words. "
            "Continue with worked examples and explanations.\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def batch(port, tasks, concurrency, repetition, limit, automatic_bos=False):
    barrier = threading.Barrier(concurrency + 1)

    def one(index):
        text = prompt(tasks[index % len(tasks)], repetition, index)
        barrier.wait()
        start = time.perf_counter()
        response = request(port, text, limit, automatic_bos)
        usage = response["usage"]
        if usage["completion_tokens"] != limit:
            raise RuntimeError(f"Short generation: {usage}; response={response}")
        return dict(index=index, seconds=time.perf_counter() - start,
                    usage=usage, text=response["choices"][0]["text"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, i) for i in range(concurrency)]
        start = time.perf_counter()
        barrier.wait()
        responses = [f.result() for f in futures]
        seconds = time.perf_counter() - start
    return dict(concurrency=concurrency, repetition=repetition, seconds=seconds,
                tokens=sum(x["usage"]["completion_tokens"] for x in responses),
                tokens_per_second=concurrency * limit / seconds, responses=responses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "fleetllm")
    parser.add_argument("--model", type=Path, default=Path.home() / "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--engines", nargs="+", choices=("fleetllm", "llama.cpp"),
                        default=["fleetllm", "llama.cpp"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = (args.root / "bench/tasks.txt").read_text().split("\n=====\n")
    port = 18091
    commands = {
        "fleetllm": [str(args.root / "build/fleet-serve"), "--model", str(args.model),
                     "--host", "127.0.0.1", "--port", str(port), "--name", "llama1b",
                     "--threads", "4", "--budget-gb", "8", "--kv-gb", "2"],
        "llama.cpp": [str(Path.home() / "llama.cpp/build/bin/llama-server"),
                      "-m", str(args.model), "--alias", "llama1b", "--host", "127.0.0.1",
                      "--port", str(port), "-t", "4", "-tb", "4", "-np", "16",
                      "-c", "16384", "-kvu", "-ngl", "0", "-cram", "0",
                      "--cache-reuse", "0"],
    }
    all_results = {}
    for engine, command in commands.items():
        if engine not in args.engines:
            all_results[engine] = json.loads((args.output / f"{engine}.json").read_text())["results"]
            continue
        with (args.output / f"{engine}.server.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=args.root)
            try:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{engine} exited; see server log")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                            break
                    except Exception:
                        if time.monotonic() > deadline:
                            raise RuntimeError(f"{engine} startup timed out")
                        time.sleep(.2)
                automatic_bos = engine == "llama.cpp"
                request(port, prompt("Explain arithmetic in detail.", -1, 0), 4, automatic_bos)
                results = []
                for concurrency in (1, 4, 16):
                    for repetition in range(args.reps):
                        result = batch(port, tasks, concurrency, concurrency * 10 + repetition, 128, automatic_bos)
                        results.append(result)
                        print(engine, concurrency, repetition,
                              f'{result["tokens_per_second"]:.2f} tok/s', flush=True)
                        (args.output / f"{engine}.json").write_text(json.dumps(
                            dict(command=command, results=results), indent=2))
                all_results[engine] = results
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    for fleet, llama in zip(all_results["fleetllm"], all_results["llama.cpp"]):
        fp = [r["usage"]["prompt_tokens"] for r in fleet["responses"]]
        lp = [r["usage"]["prompt_tokens"] for r in llama["responses"]]
        if fp != lp:
            raise RuntimeError(f"Prompt token counts differ: Fleet={fp}, llama={lp}")
    summary = []
    for n in (1, 4, 16):
        row = dict(concurrency=n)
        for engine, results in all_results.items():
            row[engine] = statistics.median(r["tokens_per_second"] for r in results if r["concurrency"] == n)
        row["fleet_over_llama"] = row["fleetllm"] / row["llama.cpp"]
        summary.append(row)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
