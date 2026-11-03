"""GPQA Diamond through OpenAI's simple-evals, against any OpenAI-compatible server.

The harness is used unchanged (its prompt, answer extraction, grading); this
script only points its chat sampler at --base-url, runs --workers questions
concurrently (each worker is one "agent" in flight), and records throughput
from the API's usage fields alongside the score.

    sh evals/setup.sh
    ./build/fleet serve --model llama8b-mlx4 --budget-gb 8 --kv-gb 2.5 &
    evals/.venv/bin/python evals/gpqa.py --model llama8b-mlx4 --workers 16 --label fleet
"""

import argparse
import hashlib
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
    ap.add_argument(
        "--workers", type=int, default=16, help="questions in flight at once"
    )
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="0 = greedy (deterministic)"
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="passes over the 198 questions (simple-evals default is 4)",
    )
    ap.add_argument(
        "--num-examples", type=int, default=None, help="random subset, for a smoke test"
    )
    ap.add_argument("--label", default="run", help="name of this run in results/gpqa/")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "results", "gpqa"))
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Record streaming progress and separate reasoning/answer channels",
    )
    ap.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"))
    ap.add_argument(
        "--request-timeout",
        type=float,
        default=3600,
        help="Seconds per HTTP request; Pi reasoning can take many minutes",
    )
    a = ap.parse_args()
    if a.workers < 1 or a.max_tokens < 1 or a.request_timeout <= 0:
        ap.error("workers, max tokens and request timeout must be positive")
    if a.reasoning_effort and not a.stream:
        ap.error("--reasoning-effort currently requires --stream")
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, a.label)

    os.environ["OPENAI_BASE_URL"] = a.base_url
    os.environ.setdefault("OPENAI_API_KEY", "none")
    from simple_evals import common
    from simple_evals.gpqa_eval import GPQAEval
    from simple_evals.sampler.chat_completion_sampler import ChatCompletionSampler
    from simple_evals.types import SamplerResponse
    from openai import OpenAI

    # One request = one agent call; record what the server reports.
    calls, lock = [], threading.Lock()
    context = threading.local()
    events = open(base + ".events.jsonl", "w")

    def event(kind, **data):
        row = dict(
            event=kind,
            timestamp=time.time(),
            question_id=getattr(context, "question_id", None),
            **data,
        )
        with lock:
            events.write(json.dumps(row) + "\n")
            events.flush()

    class TimedSampler(ChatCompletionSampler):
        def __call__(self, message_list):
            t0 = time.time()
            event("request_started")
            if a.stream:
                # The pinned evaluator still supplies the exact prompts and
                # grades only final content. Streaming changes transport only;
                # no automatic retry can silently duplicate a slow Pi request.
                actual = (
                    [self._pack_message("system", self.system_message)]
                    if self.system_message
                    else []
                ) + message_list
                content, reasoning = [], []
                usage, finish = None, None
                first = first_answer = None
                last_progress = t0
                extra = (
                    {"reasoning_effort": a.reasoning_effort}
                    if a.reasoning_effort
                    else {}
                )
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=actual,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=extra,
                )
                try:
                    for chunk in stream:
                        if chunk.usage is not None:
                            usage = chunk.usage
                        for choice in chunk.choices:
                            if choice.finish_reason:
                                finish = choice.finish_reason
                            text = choice.delta.content or ""
                            thought = (
                                getattr(choice.delta, "reasoning_content", None) or ""
                            )
                            if (text or thought) and first is None:
                                first = time.time() - t0
                                event("first_output", seconds=first)
                            if text and first_answer is None:
                                first_answer = time.time() - t0
                                event("first_answer_output", seconds=first_answer)
                            if text:
                                content.append(text)
                            if thought:
                                reasoning.append(thought)
                        if time.time() - last_progress >= 5:
                            event(
                                "request_progress",
                                seconds=time.time() - t0,
                                answer_characters=sum(map(len, content)),
                                reasoning_characters=sum(map(len, reasoning)),
                            )
                            last_progress = time.time()
                finally:
                    stream.close()
                record = dict(
                    seconds=time.time() - t0,
                    first_output_seconds=first,
                    first_answer_seconds=first_answer,
                    finish_reason=finish,
                    usage=usage.model_dump() if usage else None,
                    content="".join(content),
                    reasoning_content="".join(reasoning),
                    messages=actual,
                )
                event("request_completed", **record)
                r = SamplerResponse(
                    response_text=record["content"],
                    response_metadata={"usage": usage},
                    actual_queried_message_list=actual,
                )
            else:
                r = super().__call__(message_list)
                event(
                    "request_completed",
                    seconds=time.time() - t0,
                    content=r.response_text,
                )
            u = r.response_metadata.get("usage")
            with lock:
                calls.append(
                    {
                        "secs": time.time() - t0,
                        "prompt_tokens": getattr(u, "prompt_tokens", 0),
                        "completion_tokens": getattr(u, "completion_tokens", 0),
                    }
                )
            return r

    sampler = TimedSampler(
        model=a.model, temperature=a.temperature, max_tokens=a.max_tokens
    )
    sampler.client = OpenAI(
        base_url=a.base_url, api_key="none", timeout=a.request_timeout, max_retries=0
    )
    orig_map = common.map_with_progress

    def observed_map(f, xs, num_threads=a.workers, pbar=True):
        def observed(row):
            context.question_id = hashlib.sha256(row["Question"].encode()).hexdigest()[
                :12
            ]
            event("question_started")
            try:
                result = f(row)
                event("question_graded", score=result.score)
                print(f"[gpqa] {context.question_id}: score={result.score}", flush=True)
                return result
            except Exception as exc:
                event("question_failed", error=str(exc))
                raise

        return orig_map(observed, xs, num_threads=a.workers, pbar=pbar)

    common.map_with_progress = observed_map

    ev = GPQAEval(n_repeats=a.repeats, num_examples=a.num_examples)
    with open(base + ".manifest.json", "w") as manifest:
        json.dump(
            dict(
                model=a.model,
                base_url=a.base_url,
                workers=a.workers,
                temperature=a.temperature,
                max_tokens=a.max_tokens,
                reasoning_effort=a.reasoning_effort,
                stream=a.stream,
                request_timeout=a.request_timeout,
                repeats=a.repeats,
                questions=[
                    dict(
                        id=hashlib.sha256(row["Question"].encode()).hexdigest()[:12],
                        permutation=row["permutation"],
                    )
                    for row in ev.examples
                ],
            ),
            manifest,
            indent=2,
        )
    event("run_started", questions=len(ev.examples), workers=a.workers)
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
        "reasoning_effort": a.reasoning_effort,
        "stream": a.stream,
        "score": result.score,
        "wall_secs": round(wall, 1),
        "questions_per_min": round(len(calls) / wall * 60, 2),
        "completion_tokens": gen,
        "prompt_tokens": prompt,
        "gen_tok_per_sec": round(gen / wall, 1),
        "mean_completion_tokens": round(gen / max(1, len(calls)), 1),
        "mean_request_secs": round(
            sum(c["secs"] for c in calls) / max(1, len(calls)), 2
        ),
        "truncated": sum(1 for c in calls if c["completion_tokens"] >= a.max_tokens),
        "metrics": result.metrics,
    }
    json.dump(summary, open(base + ".json", "w"), indent=2)
    open(base + ".html", "w").write(common.make_report(result))
    event("run_completed", summary=summary)
    events.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
