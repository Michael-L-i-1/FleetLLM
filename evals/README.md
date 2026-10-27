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
./build/serve --model llama8b-mlx4 --budget-gb 8 --kv-gb 2.5 --port 8090 &
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
