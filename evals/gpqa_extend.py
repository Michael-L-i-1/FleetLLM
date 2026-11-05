"""Give GPQA questions that hit the output cap (or never finished) another run with a
larger cap, then combine the runs into one table that records where each answer came from.

    python evals/gpqa_extend.py select RUN/checkpoint NEW/checkpoint/examples.json [--controls 2] [--unfinished]
    python evals/gpqa_extend.py merge OUT_DIR FIRST/checkpoint SECOND/checkpoint [THIRD/checkpoint ...]

`select` freezes the questions a run left capped (and, with --unfinished, those it never
finished) as a new checkpoint, optionally with a few questions that stopped on their own as
controls. `merge` starts from the first run and, phase by phase, replaces any answer that is
not final (capped or unfinished) with the later run's answer. Each row keeps the phase, the
host (from the run directory's run_info.json) and the token cap that produced it.

A combined table equals one run at the largest cap only if decoding is reproducible across
the runs; the controls check that, and the summary's "method" text says which case applies.
"""

import argparse, csv, hashlib, json
from pathlib import Path

FINAL = ("correct", "wrong", "capped")


def key(example):  # same identity as checkpoint.py ids, without the index prefix
    return hashlib.sha256(json.dumps(example, sort_keys=True).encode()).hexdigest()[:12]


def load(ck):
    ck = Path(ck)
    examples = json.loads((ck / "examples.json").read_text())
    rows = list(csv.DictReader((ck / "questions.csv").open(newline="")))
    config = json.loads((ck / "config.json").read_text())
    info_file = ck.parent / "run_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    info.setdefault("name", ck.parent.name)
    info.setdefault("host", "unknown")
    return ck, examples, rows, config, info


def result(ck, qid):
    p = Path(ck) / "responses" / f"{qid}.result.json"
    return json.loads(p.read_text()) if p.exists() else None


def control_keys(ck):
    sel = Path(ck) / "selection.json"
    if not sel.exists():
        return set()
    s = json.loads(sel.read_text())
    return {s["keys"][str(i)] for i in s.get("controls", [])}


def select(a):
    ck, examples, rows, _, _ = load(a.checkpoint)
    skip = control_keys(ck)  # a previous phase's controls are not re-run
    want = lambda r: r["status"] == "capped" or (a.unfinished and r["status"] not in ("correct", "wrong"))
    picked = [int(r["index"]) for r, e in zip(rows, examples) if want(r) and key(e) not in skip]
    capped = [i for i in picked if rows[i - 1]["status"] == "capped"]
    controls = []
    for status in ("correct", "wrong"):  # one of each when possible
        for r, e in zip(rows, examples):
            if len(controls) >= a.controls: break
            if r["status"] == status and r["finish_reason"] == "stop" and key(e) not in skip:
                controls.append(int(r["index"])); break
    for r, e in zip(rows, examples):  # top up if needed
        if len(controls) >= a.controls: break
        if r["status"] in ("correct", "wrong") and r["finish_reason"] == "stop" and int(r["index"]) not in controls and key(e) not in skip:
            controls.append(int(r["index"]))
    chosen = picked + controls
    out = Path(a.examples_out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([examples[i - 1] for i in chosen], indent=2) + "\n")
    (out.parent / "selection.json").write_text(json.dumps(
        {"source": str(ck), "capped": capped, "unfinished": [i for i in picked if i not in capped],
         "controls": controls, "keys": {str(i): key(examples[i - 1]) for i in chosen}}, indent=2) + "\n")
    print(f"selected {len(capped)} capped + {len(picked) - len(capped)} unfinished + {len(controls)} control questions -> {out}")


def combine(checkpoints):
    """Combine runs in order; returns (rows, summary)."""
    runs = [load(c) for c in checkpoints]
    ck1, ex1, rows1, cfg1, info1 = runs[0]
    later = [(ck, {key(e): r for e, r in zip(ex, rows)}, cfg, info) for ck, ex, rows, cfg, info in runs[1:]]
    out_rows, checks = [], []
    for e, r in zip(ex1, rows1):
        k = key(e)
        row, ck, cfg, info, phase = r, ck1, cfg1, info1, 1
        for n, (ckn, by_key, cfgn, infon) in enumerate(later, start=2):
            rn = by_key.get(k)
            if rn is None:
                continue
            if k in control_keys(ckn) and row["status"] in ("correct", "wrong") and rn["status"] in FINAL:
                a1, a2 = result(ck1, r["id"]), result(ckn, rn["id"])
                same = bool(a1 and a2) and all(a1["response"].get(f) == a2["response"].get(f) for f in ("content", "reasoning_content"))
                checks.append({"index": int(r["index"]), "phase": n, "identical": same})
                continue
            if row["status"] not in ("correct", "wrong") and rn["status"] in FINAL:
                row, ck, cfg, info, phase = rn, ckn, cfgn, infon, n
        out_rows.append({"index": int(r["index"]), "key": k, "status": row["status"] if row["status"] in FINAL else "unfinished",
                         "score": row["score"], "completion_tokens": row["completion_tokens"], "seconds": row["seconds"],
                         "finish_reason": row["finish_reason"], "phase": phase, "run": info["name"],
                         "host": info["host"], "max_tokens": cfg["max_tokens"]})
    n = {s: sum(r["status"] == s for r in out_rows) for s in ("correct", "wrong", "capped", "unfinished")}
    phases = []
    for i, (ck, ex, rows, cfg, info) in enumerate(runs, start=1):
        used = [r for r in out_rows if r["phase"] == i]
        phases.append({"phase": i, "run": info["name"], "host": info["host"], "max_tokens": cfg["max_tokens"],
                       "workers": info.get("workers"), "questions_in_run": len(rows), "answers_used": len(used),
                       "correct_used": sum(r["status"] == "correct" for r in used)})
    identical = all(c["identical"] for c in checks) if checks else None
    summary = {"questions": len(out_rows), **n, "score": n["correct"] / len(out_rows) if out_rows else 0,
               "first_run": {"run": info1["name"], "host": info1["host"], "max_tokens": cfg1["max_tokens"],
                             **{s: sum(r["status"] == s for r in rows1) for s in ("correct", "wrong", "capped")}},
               "phases": phases, "controls": checks, "controls_identical": identical}
    hosts = sorted({p["host"] for p in phases if p["answers_used"]})
    caps = " -> ".join(str(p["max_tokens"]) for p in phases)
    summary["method"] = (
        f"Phased run (token caps {caps}; hosts: {', '.join(hosts)}). Each question keeps its first final answer; "
        f"questions that hit a cap or did not finish were run again in the next phase with a larger cap. "
        + ("Greedy outputs reproduced exactly on the controls, so this equals one run at the largest cap."
           if identical else
           "Decoding did not reproduce token for token across phases"
           + (f" (controls identical: {[c['identical'] for c in checks]})" if checks else "")
           + f", so this is an extended-budget estimate, not one uniform run; the uniform result is phase 1 "
             f"({summary['first_run']['correct']}/{len(rows1)} at {cfg1['max_tokens']} tokens)."))
    return out_rows, summary


def merge(a):
    out_rows, summary = combine(a.checkpoints)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "questions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("checkpoint"); s.add_argument("examples_out")
    s.add_argument("--controls", type=int, default=0); s.add_argument("--unfinished", action="store_true")
    m = sub.add_parser("merge"); m.add_argument("out_dir"); m.add_argument("checkpoints", nargs="+")
    a = ap.parse_args()
    select(a) if a.cmd == "select" else merge(a)
