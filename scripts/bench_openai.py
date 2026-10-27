"""Same workload as build/bench, against an OpenAI-compatible server (e.g. vLLM).

N concurrent agents, agent i gets task i % 8 from bench/tasks.txt, all sent at
t=0, greedy, exactly --gen output tokens (ignore_eos). Reports the best of
--reps runs per level: wall time, mean time to first token, aggregate tok/s.

    python3 scripts/bench_openai.py --base http://localhost:8077/v1 --model glimmer --levels 1,2,4,8 --gen 128
"""
import argparse
import json
import threading
import time
import urllib.request


def one(base, model, prompt, gen, out, i, t0):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    first, n = None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("choices") and ev["choices"][0]["delta"].get("content") and first is None:
                first = time.perf_counter() - t0
            if ev.get("usage"):
                n = ev["usage"]["completion_tokens"]
    out[i] = (first or 0.0, n)


def run(base, model, tasks, k, gen):
    out = [None] * k
    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(base, model, tasks[i % len(tasks)], gen, out, i, t0)) for i in range(k)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    return wall, sum(o[0] for o in out) / k, sum(o[1] for o in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8077/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="bench/tasks.txt")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    tasks = open(a.tasks).read().split("\n=====\n")
    run(a.base, a.model, tasks, 2, 4)  # warm-up
    print(f"engine=openai base={a.base} model={a.model} gen={a.gen}")
    print(f"{'N':>4} {'wall s':>8} {'TTFT ms':>9} {'agg tok/s':>10} {'per-agent':>10} {'tokens':>7}")
    for k in map(int, a.levels.split(",")):
        best = None
        for _ in range(a.reps):
            r = run(a.base, a.model, tasks, k, a.gen)
            if best is None or r[0] < best[0]:
                best = r
        wall, ttft, toks = best
        print(f"{k:4d} {wall:8.2f} {ttft * 1e3:9.0f} {toks / wall:10.1f} {toks / wall / k:10.1f} {toks:7d}", flush=True)


if __name__ == "__main__":
    main()
