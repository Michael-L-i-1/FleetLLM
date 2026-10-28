"""Tool-using agents over this repository, through the OpenAI tool-calling API.

Unlike agents.py (scripted turns, identical work on every engine), the model
drives here: each agent gets a question about the repository and three tools
(list_dir, read_file, grep, executed for real), and loops until it answers
without calling a tool or hits --max-steps. Engines take different paths, so
compare throughput (model calls, tool calls, tokens per second) and how many
agents finished, not exact work.

    python3 bench/tool_agents.py --base http://127.0.0.1:8090/v1 --model llama8b-mlx4 --agents 16
"""
import argparse
import json
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

TASKS = [
    "Which function in src/kv.c takes a page from the free list? Give its name.",
    "What is the engine's default row budget (rows per forward step)?",
    "Which command-line flag enables 8-bit KV, and where is it handled?",
    "What does span_find in src/spans.c return when no run is long enough?",
    "How does the server decide whether to admit a queued request?",
    "What does kv_move do to K that it does not do to V?",
    "Which chat template families does src/chat.c support?",
    "What is NG_N in src/ngram.h and what does it control?",
    "Where is the swap file for a sequence written, and which pages go into it?",
    "What does the attn_reduce kernel skip when slot[row] is all ones?",
    "Which tool measures perplexity, and what is its default context length?",
    "What does engine_retire do to a sequence?",
    "How many tokens are in a KV page (KV_PAGE)?",
    "What happens in engine_create when a matching sequence is swapped out?",
    "Which file implements the OpenAI-compatible HTTP server?",
    "What does tok_chat_template return?",
]

TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List the entries of a directory of the repository.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "directory, relative to the repository root"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read lines of a file of the repository (numbered).",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer", "description": "first line (1-based)"},
                                                                                      "lines": {"type": "integer", "description": "how many lines (default 80)"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository's source and docs for a regular expression; returns file:line: text.",
                                      "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
]

SYSTEM = ("You are a coding agent working in a C inference-engine repository (src/ engine, tools/ command-line tools, docs/, bench/). "
          "Answer the user's question about the code. Use the tools to look things up instead of guessing: grep to find where something is, "
          "read_file to read the relevant lines. Call one tool at a time. When you know the answer, reply in plain text with a short answer "
          "that names the file and function involved.")


def safe(path):
    p = os.path.normpath(os.path.join(ROOT, path or "."))
    return p if p == ROOT or p.startswith(ROOT + os.sep) else None


def run_tool(name, args):
    try:
        if name == "list_dir":
            p = safe(args.get("path", "."))
            return "\n".join(sorted(e for e in os.listdir(p) if not e.startswith("."))) if p and os.path.isdir(p) else "error: no such directory"
        if name == "read_file":
            p = safe(args.get("path", ""))
            if not p or not os.path.isfile(p):
                return "error: no such file"
            lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
            s = max(1, int(args.get("start", 1) or 1))
            n = min(200, max(1, int(args.get("lines", 80) or 80)))
            return "\n".join(f"{i}: {lines[i - 1]}" for i in range(s, min(len(lines), s + n - 1) + 1))[:6000]
        if name == "grep":
            rx = re.compile(args.get("pattern", ""))
            out = []
            for d in ("src", "tools", "docs"):
                for dp, _, fs in os.walk(os.path.join(ROOT, d)):
                    for f in sorted(fs):
                        if not f.endswith((".c", ".h", ".m", ".metal", ".md", ".py")):
                            continue
                        fp = os.path.join(dp, f)
                        for i, l in enumerate(open(fp, encoding="utf-8", errors="replace"), 1):
                            if rx.search(l):
                                out.append(f"{os.path.relpath(fp, ROOT)}:{i}: {l.strip()[:160]}")
                                if len(out) >= 30:
                                    return "\n".join(out)
            return "\n".join(out) or "no matches"
    except Exception as e:  # bad regex, bad arguments
        return f"error: {e}"
    return f"error: unknown tool {name}"


def chat(base, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "tools": TOOLS, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return r["choices"][0], r.get("usage") or {}, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--agents", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default=None, help="write results/agents/<label>.json")
    a = ap.parse_args()

    lock = threading.Lock()
    stats = {"calls": 0, "tool_calls": 0, "finished": 0, "failed": 0, "prompt": 0, "completion": 0, "cached": 0, "secs": []}
    answers = {}

    def agent(i):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASKS[i % len(TASKS)]}]
        for _ in range(a.max_steps):
            try:
                choice, u, secs = chat(a.base, a.model, msgs, a.max_tokens)
            except urllib.error.HTTPError:  # the server rejected the conversation (e.g. its template)
                with lock:
                    stats["failed"] += 1
                return
            m = choice["message"]
            calls = m.get("tool_calls") or []
            with lock:
                stats["calls"] += 1
                stats["tool_calls"] += len(calls)
                stats["prompt"] += u.get("prompt_tokens", 0)
                stats["completion"] += u.get("completion_tokens", 0)
                stats["cached"] += ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
                stats["secs"].append(secs)
            if not calls:
                with lock:
                    stats["finished"] += 1
                    answers[i] = (m.get("content") or "")[:300]
                return
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": run_tool(c["function"]["name"], args if isinstance(args, dict) else {})})

    t0 = time.time()
    th = [threading.Thread(target=agent, args=(i,)) for i in range(a.agents)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    wall = time.time() - t0
    out = {
        "base": a.base, "model": a.model, "agents": a.agents, "max_steps": a.max_steps,
        "wall_secs": round(wall, 1), "model_calls": stats["calls"], "tool_calls": stats["tool_calls"],
        "finished": stats["finished"], "failed_requests": stats["failed"], "calls_per_min": round(stats["calls"] / wall * 60, 1),
        "call_p50_secs": round(statistics.median(stats["secs"]), 2), "gen_tokens": stats["completion"],
        "gen_tok_per_sec": round(stats["completion"] / wall, 1), "prompt_tokens": stats["prompt"],
        "prompt_tok_per_sec": round(stats["prompt"] / wall, 1), "cached_prompt_tokens": stats["cached"],
        "answers": {str(k): v for k, v in sorted(answers.items())},
    }
    print(json.dumps({k: v for k, v in out.items() if k != "answers"}, indent=2))
    if a.label:
        d = os.path.join(ROOT, "results", "agents")
        os.makedirs(d, exist_ok=True)
        json.dump(out, open(os.path.join(d, a.label + ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
