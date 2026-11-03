# Evals

Accuracy and throughput on real benchmarks, through an unmodified public
harness talking to the engine over the OpenAI API (the same way an agent
framework would).

## GPQA Diamond (OpenAI simple-evals)

198 graduate-level science questions, 4 options each, the harness's standard
prompt ("Think step by step ... last line 'Answer: $LETTER'") and regex
grader. `gpqa.py` only points simple-evals' chat sampler at a server, runs
`--workers` questions concurrently (each worker = one agent in flight) and
records the usage the server reports.

```bash
sh evals/setup.sh                       # venv + simple-evals pinned to a fixed commit (third_party/, gitignored)
./build/fleet serve --model llama8b-mlx4 --budget-gb 8 --kv-gb 2.5 --port 8090 &
evals/.venv/bin/python evals/gpqa.py --model llama8b-mlx4 --workers 16 --label my-run
evals/.venv/bin/python evals/gpqa.py --model llama8b-mlx4 --workers 1 --num-examples 16 --label single   # one agent at a time
```

Each run writes `results/gpqa/<label>.json` (score, wall time, tokens,
throughput, truncations) and `<label>.html` (every question, response, and
grade). Any OpenAI-compatible endpoint works with `--base-url` (vLLM, Ollama,
...), so other engines can be measured with the identical harness.

Notes:
- `--temperature 0` is greedy and deterministic; small models tend to loop
  until `--max-tokens` under greedy decoding, which costs them answers.
  simple-evals' own sampler default is 0.5.
- The data comes from OpenAI's public copy of GPQA Diamond
  (`openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv`).

## Observable Pi pilots

Run the evaluation client on the Mac and connect it to the Pi's localhost server
through SSH forwarding. Use explicit concurrency, reasoning and KV settings;
the default Pi demo server is direct-answer and allows only one active request.
For example, with the Pi server forwarded to local port 18096:

```sh
evals/.venv/bin/python evals/gpqa.py \
  --base-url http://127.0.0.1:18096/v1 --model pi-gpqa-glimmer \
  --workers 4 --num-examples 4 --repeats 1 --temperature 0 \
  --max-tokens 1024 --stream --reasoning-effort low \
  --request-timeout 5400 --label pi-pilot
```

The Pi pilot server uses the installed Muse-Glimmer-30B Q2_K model, four CPU
threads, FP16 KV 1 GB, default activation precision, prefix sharing and verified
prompt lookup. No pruning or learned drafter is required. Four concurrent
questions are independent evaluation requests, not a multi-agent debate.

In addition to the existing JSON/HTML report, the wrapper writes
`<label>.manifest.json` and append-only `<label>.events.jsonl`. Events identify
questions and record starts, first visible output, first answer output,
five-second character-count progress, completed responses/usage/finish reasons,
and grades. Progress characters and stream chunks are **not token counts**.
Final token totals come from the server's usage report, including generated
reasoning. Only final answer content is passed to the pinned evaluator's grader.
The prompts and grading code in `third_party/simple_evals` remain unchanged.

`--reasoning-effort` requires `--stream`. Streaming uses a finite HTTP timeout
and no automatic request retries, avoiding accidental duplicate long Pi jobs.
The existing non-streaming sampler remains available. The event stream can feed
a future viewer tab; this change does not alter `scripts/pimon.py`.

Validate the streaming adapter without a model or dataset download:

```sh
evals/.venv/bin/python scripts/gpqa_streamcheck.py
```

The local fixture checks four concurrent responses, reasoning/final separation,
official final-only grading, usage, truncation and stable question IDs.
