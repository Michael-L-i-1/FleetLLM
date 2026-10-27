# cinfer

A C inference engine for running **many concurrent agents** (subagents, long
tool loops, eval harnesses) through one copy of a model on **limited
hardware**: a Raspberry Pi 5 with 16 GB is the final target; an Apple M5 Pro
(Metal) is the development machine.

It follows vLLM's core design (paged KV, continuous batching, chunked
prefill, prefix sharing) under a hard memory budget, plus agent-oriented
pieces: deduplicating agents that fan out at the same moment, recompute
preemption under a fixed KV budget, speculative decoding that backs off as
more agents decode, and an OpenAI-compatible server so real harnesses drive it.

Pure C11 plus one Objective-C file for Metal. No runtime dependencies; ported
kernel code is credited in place (ggml/llama.cpp, MIT; MLX, MIT, vendored
headers in `src/metal/mlx/`). Models load from GGUF or Hugging Face / MLX
safetensors (Llama, Qwen2/3, Muse Glimmer).

## Highlights

**GPQA Diamond, Llama 3.1 8B (MLX 4-bit), OpenAI simple-evals, 16 agents at once**
(4.2 GB weights + 2.5 GB KV cap; full table in [docs/results.md](docs/results.md)):

| Run | Agents at once | Temp | Score | Wall time | Generated tok/s | Questions/min | Hit 2048-token cap |
|---|---|---|---|---|---|---|---|
| all 198 questions | 16 | 0 (greedy) | 25.3% | 613 s | 282 | 19.4 | 55 |
| all 198 questions | 16 | 0.5 | 27.3% | 546 s | 277 | 21.8 | 37 |
| same 16 questions | **1** | 0 | 31.2% | 238 s | 52 | 4.0 | |
| same 16 questions | **16** | 0 | 31.2% | 59 s | 211 | 16.2 | |

16 agents finish the same work **4x faster** than one at a time (5.4x sustained over the
full run). Meta reports 30.4% for Llama 3.1 8B Instruct on GPQA (0-shot CoT, bf16); with
198 questions the standard error is about 3 points, and most of the gap is truncation from
repetition loops at greedy / low temperature. Greedy output is token-identical to MLX's
reference implementation on the same weights (checked on a GPQA prompt, 200 tokens).

**Muse Glimmer 30B (same 4-bit file as vLLM), aggregate tok/s:**

| Agents | vLLM | cinfer | cinfer + sharing | cinfer + DFlash spec + sharing |
|---|---|---|---|---|
| 1 | 17.3 | 17.2 | 18.2 | 90.8 |
| 8 | 57.6 | 98.4 | 118.0 | 158.7 |
| 32 | 113.2 | 210.0 | 413.6 | 408.9 |

## Quick start

```bash
make                                   # Metal on macOS; make NO_METAL=1 for CPU-only (Pi)
./build/serve --model llama8b-mlx4 --budget-gb 8 --kv-gb 2.5 --port 8090   # OpenAI API on :8090/v1
sh evals/setup.sh && evals/.venv/bin/python evals/gpqa.py --model llama8b-mlx4 --workers 16
```

More: `./build/generate` (one prompt), `./build/bench` (concurrent-agent
throughput, `--share`, `--draft`, `--check`, `--check-spec`), `./build/ppl`
(perplexity, for lossy options). `--model` takes a GGUF file, a safetensors
directory, or an alias (`llama8b-mlx4`, `llama1b-mlx4`, `glimmer`,
`glimmer-mlx4`, `glimmer-draft`, `qwen3`) resolved in the Hugging Face cache.

## Repository

```
src/          engine: model loading, tokenizer, chat templates, paged KV, scheduler, server
src/metal/    Metal backend and kernels (+ vendored MLX headers)
tools/        CLIs: serve, generate, bench, ppl, tokcheck, dequant_dump
bench/        concurrent-agent throughput workload + OpenAI-API client (for vLLM etc.)
evals/        GPQA Diamond via OpenAI simple-evals (setup.sh, gpqa.py)
results/      raw outputs: throughput/, gpqa/
docs/         design.md, results.md
scripts/      verification against HF/gguf-py, build helpers
```

## Status

| Piece | State |
|---|---|
| GGUF loader (mmap, zero-copy) | done |
| Tokenizer from GGUF metadata (byte-level BPE; llama3 / qwen2 / llama4 pre-tokenizers) | matches HF `tokenizers` 10/10 on all three families |
| Dequantization F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K..Q6_K | bit-exact vs gguf-py |
| Generic decoder: llama, qwen2, qwen3, muse-glimmer | correct output on all |
| CPU reference backend | correct, slow by design (Pi NEON kernels are next) |
| Metal backend | correct; GGUF: fused mat-vec / fused tile GEMM / dequant-once + dense GEMM; MLX 4-bit: tiled NAX GEMM + tiled mat-vec; flash-decoding attention |
| MLX / HF safetensors loading (config, shards, tokenizer.json, affine weights) | Glimmer, Llama verified on CPU and GPU; tokenizer matches HF 10/10 |
| In-flight prefix dedup (agents fanned out together share one prefill) | done |
| Speculative decoding with the DFlash drafter (batched, adaptive draft length) | done; outputs identical to plain greedy |
| OpenAI-compatible server (streaming, reasoning split, KV cap + recompute preemption) | done; drives simple-evals GPQA at 16 workers |
| Paged copy-on-write KV (full + sliding-window pools), continuous batching, automatic prefix sharing | done; batched greedy outputs identical to sequential (8/8) |

## Roadmap

1. ~~Server + real harness~~ (done: OpenAI-compatible `serve`, GPQA Diamond via simple-evals).
   Next for this phase: multi-turn agent sessions (long contexts, tool waits, unique prompts)
   as a benchmark, and the same harness against vLLM / Ollama for comparison.
2. **One tiled weight format for every quant type + startup auto-tuning** (crossovers,
   split sizes, draft budget measured per device instead of tuned for the M5); the 2-bit
   Glimmer GGUF then gets the fast kernels and fits 16 GB.
3. **Agent memory**: session API (KV kept across turns), explicit fork for subagents, 8-bit KV,
   tool-wait KV offload to SSD, multi-row prefill attention for long contexts.
4. **Free speculation**: copy-from-context drafts, schema-forced tokens for tool calls; the
   DFlash drafter switched on only when free compute and memory allow.
5. **Pi 5**: NEON kernels on the same tiled format.
