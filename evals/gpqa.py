"""GPQA Diamond through OpenAI's simple-evals, against any OpenAI-compatible server.

The harness is used unchanged (its prompt, answer extraction, grading); this
script only points its chat sampler at --base-url, runs --workers questions
concurrently (each worker is one "agent" in flight), and records throughput
from the API's usage fields alongside the score.

    sh evals/setup.sh
    ./build/serve --model llama8b-mlx4 --budget-gb 8 --kv-gb 2.5 &
    evals/.venv/bin/python evals/gpqa.py --model llama8b-mlx4 --workers 16 --label cinfer
"""
import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", required=True, help="model id sent in requests")
    ap.add_argument("--workers", type=int, default=16, help="questions in flight at once")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (deterministic)")
    ap.add_argument("--repeats", type=int, default=1, help="passes over the 198 questions (simple-evals default is 4)")
    ap.add_argument("--num-examples", type=int, default=None, help="random subset, for a smoke test")
    ap.add_argument("--label", default="run", help="name of this run in results/gpqa/")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "results", "gpqa"))
    a = ap.parse_args()

    os.environ["OPENAI_BASE_URL"] = a.base_url
    os.environ.setdefault("OPENAI_API_KEY", "none")
    from simple_evals import common
    from simple_evals.gpqa_eval import GPQAEval
    from simple_evals.sampler.chat_completion_sampler import ChatCompletionSampler

    # One request = one agent call; record what the server reports.
    calls, lock = [], threading.Lock()

    class TimedSampler(ChatCompletionSampler):
        def __call__(self, message_list):
            t0 = time.time()
            r = super().__call__(message_list)
            u = r.response_metadata.get("usage")
            with lock:
                calls.append({"secs": time.time() - t0, "prompt_tokens": getattr(u, "prompt_tokens", 0), "completion_tokens": getattr(u, "completion_tokens", 0)})
            return r

    sampler = TimedSampler(model=a.model, temperature=a.temperature, max_tokens=a.max_tokens)
    orig_map = common.map_with_progress
    common.map_with_progress = lambda f, xs, num_threads=a.workers, pbar=True: orig_map(f, xs, num_threads=a.workers, pbar=pbar)

    ev = GPQAEval(n_repeats=a.repeats, num_examples=a.num_examples)
    t0 = time.time()
    result = ev(sampler)
    wall = time.time() - t0

    gen = sum(c["completion_tokens"] for c in calls)
    prompt = sum(c["prompt_tokens"] for c in calls)
    summary = {
        "label": a.label,
        "model": a.model,
        "base_url": a.base_url,
        "harness": "openai/simple-evals gpqa_eval (diamond)",
        "questions": len(calls),
        "workers": a.workers,
        "temperature": a.temperature,
        "max_tokens": a.max_tokens,
        "score": result.score,
        "wall_secs": round(wall, 1),
        "questions_per_min": round(len(calls) / wall * 60, 2),
        "completion_tokens": gen,
        "prompt_tokens": prompt,
        "gen_tok_per_sec": round(gen / wall, 1),
        "mean_completion_tokens": round(gen / max(1, len(calls)), 1),
        "mean_request_secs": round(sum(c["secs"] for c in calls) / max(1, len(calls)), 2),
        "truncated": sum(1 for c in calls if c["completion_tokens"] >= a.max_tokens),
        "metrics": result.metrics,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, a.label)
    json.dump(summary, open(base + ".json", "w"), indent=2)
    open(base + ".html", "w").write(common.make_report(result))
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
