"""Multi-turn agent sessions against an OpenAI-compatible server.

Models the shape of coding-agent traffic (TraceLab: long autonomous loops,
growing contexts with short outputs, tool results between model calls):

  * N agents start together from one shared context (system prompt with tool
    descriptions + project files), like subagents fanned out by an orchestrator
  * each turn the agent answers (short output), a tool runs (simulated latency),
    and its result (a real file from this repository; agents read overlapping
    files) is appended; the next request resends the whole conversation
  * --keep-tools N models harnesses that edit their context (drop or condense
    old tool outputs): only the last N tool results are sent verbatim, older
    ones become a one-line stub, so everything after the edit point is the same
    text at shifted positions
  * everything is deterministic (fixed seed, greedy) so engines see identical work

Reports wall time, turn latency, generated tok/s and how many prompt tokens the
server reused from cache (usage.prompt_tokens_details.cached_tokens, if given).

    python3 bench/agents.py --base http://127.0.0.1:8090/v1 --model llama8b-mlx4 --agents 8 --turns 6
"""
import argparse
import glob
import json
import os
import random
import statistics
import threading
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def corpus():
    files = sorted(glob.glob(os.path.join(ROOT, "src", "*.c")) + glob.glob(os.path.join(ROOT, "src", "*.h")) + glob.glob(os.path.join(ROOT, "docs", "*.md")))
    return [(os.path.relpath(f, ROOT), open(f, encoding="utf-8", errors="replace").read()) for f in files]


def chat(base, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    u = r.get("usage") or {}
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return r["choices"][0]["message"]["content"] or "", time.time() - t0, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=160, help="output per turn (agents write short steps)")
    ap.add_argument("--shared-chars", type=int, default=12000, help="shared project context in the system prompt")
    ap.add_argument("--tool-chars", type=int, default=2400, help="size of each tool result")
    ap.add_argument("--tool-secs", type=float, default=1.5, help="mean simulated tool latency (exponential, capped at 4x)")
    ap.add_argument("--keep-tools", type=int, default=0, help="send only the last N tool results verbatim (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None, help="write results/agents/<label>.json")
    a = ap.parse_args()

    files = corpus()
    shared = "".join(f"\n### {name}\n{text[:2000]}" for name, text in files[:12])[: a.shared_chars]
    system = (
        "You are a coding agent working in a C inference-engine repository. Tools: read_file(path) returns a file; "
        "you think briefly, then state the single next step and any risk you see. Keep every reply under 120 words.\n"
        "Project overview and key files:" + shared
    )
    rng = random.Random(a.seed)
    plans = []
    for i in range(a.agents):  # each agent reads a different but overlapping sequence of files
        order = [(i + 3 * t) % len(files) for t in range(a.turns)]
        delays = [min(4 * a.tool_secs, rng.expovariate(1 / a.tool_secs)) for _ in range(a.turns)]
        plans.append((order, delays))

    turns, lock = [], threading.Lock()

    def agent(i):
        order, delays = plans[i]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Task {i}: audit how {files[order[0]][0]} interacts with the rest of the engine and propose one concrete improvement."}]
        t_start = time.time()
        for t in range(a.turns):
            sent = msgs
            if a.keep_tools:
                tools = [k for k, m in enumerate(msgs) if m.get("tool")]
                old = set(tools[:-a.keep_tools])
                sent = [{"role": m["role"], "content": m["tool"] + " returned: [output omitted]\nContinue."} if k in old else m for k, m in enumerate(msgs)]
            sent = [{"role": m["role"], "content": m["content"]} for m in sent]
            text, secs, p, c, cached = chat(a.base, a.model, sent, a.max_tokens)
            with lock:
                turns.append({"agent": i, "turn": t, "secs": secs, "prompt": p, "completion": c, "cached": cached})
            msgs.append({"role": "assistant", "content": text})
            if t + 1 < a.turns:
                time.sleep(delays[t])  # the tool runs
                name, body = files[order[t + 1]]
                msgs.append({"role": "user", "content": f"read_file({name}) returned:\n{body[: a.tool_chars]}\nContinue.", "tool": f"read_file({name})"})
        with lock:
            done.append(time.time() - t_start)

    done = []
    t0 = time.time()
    th = [threading.Thread(target=agent, args=(i,)) for i in range(a.agents)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    wall = time.time() - t0
    lat = sorted(x["secs"] for x in turns)
    gen = sum(x["completion"] for x in turns)
    prompt = sum(x["prompt"] for x in turns)
    cached = sum(x["cached"] for x in turns)
    out = {
        "base": a.base, "model": a.model, "agents": a.agents, "turns": a.turns, "max_tokens": a.max_tokens, "keep_tools": a.keep_tools,
        "wall_secs": round(wall, 1), "mean_agent_secs": round(statistics.mean(done), 1),
        "turn_p50_secs": round(lat[len(lat) // 2], 2), "turn_p95_secs": round(lat[int(len(lat) * 0.95) - 1], 2),
        "gen_tokens": gen, "gen_tok_per_sec": round(gen / wall, 1),
        "prompt_tokens": prompt, "cached_prompt_tokens": cached,
        "cache_hit": round(cached / prompt, 3) if prompt else 0,
        "mean_final_context": round(statistics.mean(x["prompt"] for x in turns if x["turn"] == a.turns - 1)),
    }
    print(json.dumps(out, indent=2))
    if a.label:
        d = os.path.join(ROOT, "results", "agents")
        os.makedirs(d, exist_ok=True)
        json.dump(out, open(os.path.join(d, a.label + ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
