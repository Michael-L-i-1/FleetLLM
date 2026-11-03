# FleetLLM

High-throughput inference for fleets of agents on one machine.

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

**GPQA Diamond, Llama 3.1 8B (MLX 4-bit), OpenAI simple-evals, vs vLLM on the same file**
(4.2 GB weights + 2.5 GB KV cap; full table in [docs/results.md](docs/results.md)):

| Run | Agents at once | **FleetLLM** wall | vLLM wall | Speedup | FleetLLM tok/s | vLLM tok/s | Score FleetLLM / vLLM |
|---|---|---|---|---|---|---|---|
| all 198, greedy | 16 | **616 s** | 873 s | **1.42x** | 281 | 204 | 25.3% / 21.7% |
| all 198, T=0.5 | 16 | **547 s** | 878 s | **1.61x** | 276 | 171 | 27.3% / 24.2% |
| same 16 questions | 1 | **241 s** | 344 s | **1.43x** | 51 | 40 | 31.2% / 25.0% |
| same 16 questions | 16 | **59 s** | 80 s | **1.37x** | 213 | 150 | 31.2% / 31.2% |

Same model file (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`), same harness and settings,
matched KV (FleetLLM `--kv-gb 2.5` = 19,072 tokens; vllm-metal 0.32 `--gpu-memory-utilization
0.193` = 18,416 tokens, since it ignores `--kv-cache-memory-bytes`), prefix caching on in both,
AC power, run back to back (`evals/gpqa_suite.sh`). 16 agents finish the same work 4.1x faster
than one at a time on FleetLLM (4.3x on vLLM). Score differences are noise (about +/-3 points at
n=198) plus greedy paths diverging; both engines lose ~30% of questions to repetition loops that
hit the 2048-token cap. GPQA prompts share almost nothing, so this measures raw batching and
kernels, not the agent features (sharing, sessions). Greedy FleetLLM output is token-identical to
MLX's reference on the same weights (GPQA prompt, 200 tokens).

**Multi-turn agent sessions** (8 agents x 6 turns from a shared context,
tool results appended, ~8k-token final contexts, 2.5 GB KV cap, same 4-bit
Llama 3.1 8B as vLLM): **52 s vs vLLM's 158 s (3.0x)**, p95 turn 9.3 s vs
41.9 s. From 975 s that morning: repeated-text speculation shared across agents,
8-bit KV, attention for input rows on the M5 neural accelerators, and a
latency-shaped decode attention kernel, with kernel choice per row so outputs
stay batch-invariant. The same 16 GPQA questions take 121-130 s with 1 agent
and 32-42 s with 16 (vLLM: 344 s / 80 s). With **16 agents** (~130k tokens of
session state, 38k tokens of KV) sessions park on SSD between turns
(share-aware swap + memory-aware admission): **85 s vs vLLM's 315 s (3.7x)**,
p95 turn 17 s vs 77 s; 32 agents **163 s vs 595 s** (p95 31 s vs 162 s). Opt-in **span reuse** (`--reuse-spans`: KV of text another
agent already processed, e.g. the same file read by several subagents, is moved
instead of recomputed; +0.36% perplexity on the reused text) brings 8 / 16 / 32
agents to **43 / 64 / 111 s** (3.7x / 4.9x / 5.4x vLLM), and a harness that drops old tool
outputs (history re-sent at shifted positions) to **68 s vs vLLM's 161 s**.
Agents that drive themselves through OpenAI tool calling (`bench/tool_agents.py`:
real `grep` / `read_file` tools on this repository) make **406 model calls per
minute vs vLLM's 219** at 16 agents (663 vs 438 at 32). See [docs/results.md](docs/results.md).

**Muse Glimmer 30B (same 4-bit file as vLLM), aggregate tok/s:**

| Agents | vLLM | FleetLLM | FleetLLM + sharing | FleetLLM + DFlash spec + sharing |
|---|---|---|---|---|
| 1 | 17.3 | 17.2 | 18.2 | 90.8 |
| 8 | 57.6 | 98.4 | 118.0 | 158.7 |
| 32 | 113.2 | 210.0 | 413.6 | 408.9 |

## Quick start

```bash
make                                   # Metal on macOS; make NO_METAL=1 for CPU-only (Pi)
./build/fleet                          # list subcommands
./build/fleet serve --model llama8b-mlx4 --budget-gb 8 --kv-gb 2.5 --port 8090   # OpenAI API on :8090/v1
sh evals/setup.sh && evals/.venv/bin/python evals/gpqa.py --model llama8b-mlx4 --workers 16
```

Agent-oriented flags (any model): `--ngram` (drafts from repeated text in every
agent's context, exact at any temperature), `--kv-int8` (2x tokens per GB),
`--swap-gb N --swap-dir D` (evicted KV goes to SSD instead of being recomputed;
experimental). `python3 bench/agents.py` runs multi-turn agent sessions against
any OpenAI-compatible server.

More: `./build/fleet generate` (one prompt), `./build/fleet bench` (concurrent-agent
throughput, `--share`, `--draft`, `--check`, `--check-spec`), `./build/fleet ppl`
(perplexity, for lossy options); `./build/fleet` lists them all. `--model` takes a
GGUF file, a safetensors directory, or an alias (`llama8b-mlx4`, `llama1b-mlx4`,
`glimmer`, `glimmer-mlx4`, `glimmer-draft`, `qwen3`) resolved in the Hugging Face cache.

## Repository

```
src/          engine: model loading, tokenizer, chat templates, paged KV, scheduler, server
src/metal/    Metal backend and kernels (+ vendored MLX headers)
tools/        `fleet` subcommands: serve, generate, bench, ppl, reuse-eval, tokcheck, dequant-dump
              (built as build/fleet-<name>; tools/fleet/ is the `fleet` dispatcher)
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
| CPU reference backend | correct, slow by design |
| Optimized CPU backend | NEON integer kernels for Q2_K/Q3_K/Q4_K/Q6_K, Q4_0/Q4_1/Q8_0 and affine 4-bit; [Pi Glimmer Q2_K tests](results/pi5/2026-11-02/glimmer-q2k/README.md) |
| Metal backend | correct; GGUF: fused mat-vec / fused tile GEMM / dequant-once + dense GEMM; MLX 4-bit: tiled NAX GEMM + tiled mat-vec; attention: NAX (input rows), latency-shaped vector kernel (decode / verify), 8x8-MMA flash kernel (GPUs without NAX) |
| MLX / HF safetensors loading (config, shards, tokenizer.json, affine weights) | Glimmer, Llama verified on CPU and GPU; tokenizer matches HF 10/10 |
| In-flight prefix dedup (agents fanned out together share one prefill) | done |
| Speculative decoding with the DFlash drafter (batched, adaptive draft length) | Metal and CPU; [first Pi Glimmer test](results/pi5/2026-11-02/glimmer-dflash/README.md), with greedy-output validation |
| OpenAI-compatible server (streaming, reasoning split, tool calling, KV cap, swap + memory-aware admission) | done; drives simple-evals GPQA and tool-calling agents (official `openai` client) |
| Paged copy-on-write KV (full + sliding-window pools), continuous batching, automatic prefix sharing | done; batched greedy outputs identical to sequential (8/8) |

## Raspberry Pi tuning

The [Pi optimization sweep](results/pi5/2026-11-02/optimization-sweep/README.md)
compares normal decoding, verified prompt lookup, DFlash, activation precision,
and rejected pruning experiments. [Quality controls](docs/pi-tuning.md) are
explicit and composable: default CPU arithmetic keeps all layers/features;
pruning and coarser activations are opt-in. The optional Qwen3-8B launcher is a
separate model choice, while Glimmer remains available unchanged.
The [isolated follow-up experiments](results/pi5/2026-11-02/followup/README.md)
record additional kernel/drafter trials and direct-answer quality checks.

## Roadmap

1. ~~Server + real harness~~ (done: OpenAI-compatible `fleet serve`, GPQA Diamond via simple-evals).
   Next for this phase: multi-turn agent sessions (long contexts, tool waits, unique prompts)
   as a benchmark, and the same harness against vLLM / Ollama for comparison.
2. **One tiled weight format for every quant type + startup auto-tuning** (crossovers,
   split sizes, draft budget measured per device instead of tuned for the M5).
   CPU Q2_K/Q3_K tiles are implemented and Glimmer fits 16 GB; Metal low-bit
   kernel coverage and startup auto-tuning remain.
3. **Agent memory**: session API (KV kept across turns), explicit fork for subagents, 8-bit KV,
   tool-wait KV offload to SSD, multi-row prefill attention for long contexts.
4. **Free speculation**: copy-from-context drafts, schema-forced tokens for tool calls; the
   DFlash drafter switched on only when free compute and memory allow.
5. **Pi 5**: NEON kernels deployed; continue tuning and measuring agent workloads.
