# cinfer

A C inference engine for running **many concurrent agents** (subagents, long
tool loops, Terminal-Bench-style runs) through one copy of a model on
constrained hardware: a Raspberry Pi 5 with 16 GB is the final target; an Apple
M5 Pro (Metal) is the development machine. It follows vLLM's core design
(paged KV, continuous batching, chunked prefill, prefix sharing) under a hard
memory budget, with agent-specific features to come.

Pure C11 plus one Objective-C file for the Metal host code. No runtime
dependencies. Kernel code ported from other projects is credited in place
(ggml/llama.cpp, MIT; MLX, MIT: vendored headers in `src/metal/mlx/`).

Models load from a GGUF file or a Hugging Face / MLX safetensors directory
(config.json + sharded safetensors + tokenizer.json, affine 4/8-bit weights).

## Head to head with vLLM on the same 4-bit file

`mlx-community/Muse-Glimmer-30B-4bit` (the checkpoint vllm-metal serves), M5 Pro
on battery, both engines back to back. Workload: N concurrent agents, agent i
gets task i % 8 from `bench/tasks.txt` (8 different tasks), all sent at t=0,
greedy, exactly 128 output tokens each. Aggregate output tok/s
(`build/bench` vs `scripts/bench_openai.py` against vllm-metal 0.32 on :8077).

| Agents | vLLM, prefix cache off | cinfer, sharing off | vLLM, prefix cache on (default) | cinfer `--share` |
|---|---|---|---|---|
| 1 | 17.3 | 16.6 | 18.2 | 18.1 |
| 2 | 31.0 | 31.2 | 34.1 | 34.8 |
| 4 | 52.9 | 57.5 | 56.0 | 64.7 |
| 8 | 57.6 | **96.1** | 63.5 | **111.8** |
| 16 | 71.2 | **147.5** | 87.5 | **219.3** |
| 32 | 113.2 | **194.1** | 150.4 | **360.3** |

Mean time to first token at 32 agents: vLLM 9.5 s / 1.0 s (cache off / on),
cinfer 5.3 s / 0.08 s. Raw output in `results/`. With sharing, cinfer also
deduplicates prompts that arrive at the same moment (a new agent waits for a
still-prefilling agent with the same prefix and forks its KV); vLLM's prefix
cache only reuses blocks that finished computing.

Why it is faster at 4+ agents (all measured with the kernel microbenchmark):
MLX's kernels re-read the weights for every 5 decode rows (`qmv_wide`) and its
4-bit GEMM streams 32-byte pieces of 64 rows at once, so 8-32 rows cost 1.5-3x
one row. cinfer repacks 4-bit weights at load into 64x64 tiles with the scales
inline (each K step is one contiguous read) and runs them on the M5 neural
accelerators with split-K, so 1-16 rows cost within ~10% of one row.

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
| Paged copy-on-write KV (full + sliding-window pools), continuous batching, automatic prefix sharing | done; batched greedy outputs identical to sequential (8/8) |

## Build and run

```bash
make                     # Metal on macOS; `make NO_METAL=1` for CPU-only (e.g. Pi)
./build/generate --model glimmer --prompt "What is the capital of France?"
./build/bench --model llama1b --budget-gb 4 --levels 1,2,4,8,16,32 --gen 128
./build/bench --model llama1b --budget-gb 4 --check      # batched == sequential
./build/bench --model glimmer-mlx4 --budget-gb 18 --levels 1,8,32 --gen 128 --share
CINFER_TMV_MAX=0 ./build/bench --model glimmer-mlx4 --budget-gb 18 --check   # exact: one kernel path for all batch sizes
./build/ppl --model glimmer-mlx4 --budget-gb 18 --file text.txt              # perplexity (for lossy options)
python3 scripts/bench_openai.py --base http://localhost:8077/v1 --model glimmer --levels 1,8,32 --gen 128
python3 scripts/tokcheck.py MODEL.gguf tokenizer.json   # tokenizer vs HF
python3 scripts/dequantcheck.py MODEL.gguf              # dequant vs gguf-py
```

`--model` takes `glimmer`, `llama1b`, `qwen3`, `glimmer-mlx4`, `llama1b-mlx4` (resolved in the Hugging Face cache) or a path
(GGUF file or safetensors directory). The MLX 4-bit Glimmer needs ~15 GB of weights, so its runs use `--budget-gb 18`.
`--backend cpu|metal`, `--budget-gb N` (weights + KV + scratch must fit), `--row-budget N` (prefill chunk).

## Layout

```
src/gguf.*        GGUF reader
src/safetensors.* safetensors reader (sharded); src/json.* minimal JSON parser
src/quant.*       reference dequantization (ggml port)
src/tokenizer.*   BPE + hand-written pre-tokenizers; src/unicode_data.c is generated
src/chat.*        chat formats (Llama 3, ChatML/Qwen, Glimmer channels)
src/model.*       generic decoder description (flags per architecture)
src/kv.*          paged ref-counted copy-on-write KV
src/engine.*      sequences, scheduler, prefix sharing, sampling
src/cpu.c         CPU reference backend
src/metal/        Metal backend (metal.m) and kernels (kernels.metal; kernels_nax.metal for the
                  M5 neural accelerators; vendored MLX headers in mlx/), embedded at build
tools/            generate, bench, ppl, tokcheck, dequant_dump
scripts/          verification scripts, Unicode table generator
bench/tasks.txt   8 heterogeneous agent tasks used by bench
```

## Metal matmul dispatch

| Rows in the step | Kernel |
|---|---|
| 1-8 | fused mat-vec on the packed K-quant blocks |
| 9-64 | fused GEMM: weight tiles decoded into threadgroup memory, simdgroup MMA |
| > 64 (prefill) | dequantize once into an f16 scratch matrix (256 MB, row-chunked), dense GEMM |

Crossovers are env-tunable: `CINFER_MV_MAX`, `CINFER_FUSED_MAX`.

MLX affine 4-bit weights (repacked at load into 64-row x 64-K blocks, scales/biases inline):

| Rows in the step | Kernel |
|---|---|
| 1-2 | tiled mat-vec (8 rows per simdgroup, 4 K tiles of loads in flight), split-K |
| 3-4 | same, with the 16-value partial dots in f16 (codes centered at 8; perplexity unchanged) |
| 5-16 / 17+ | NAX GEMM, one simdgroup per 32 weight rows x 16 / 32 tokens, split-K |

q/k/v/gate and ffn gate/up are fused into single matrices at repack time; the
norm, attention and SiLU kernels write f16 matmul inputs directly; the residual
add, post-norm and next pre-norm run as one kernel.

## Baseline results (M5 Pro, AC power, 8 different tasks, 128 tokens each, greedy, no prefix sharing)

| Agents | Llama 3.2 1B Q4_K_M tok/s | Glimmer 30B UD-Q2_K_XL tok/s | Glimmer TTFT |
|---|---|---|---|
| 1 | 167 | 11.9 | 2.4 s |
| 2 | 266 | 17.5 | 3.5 s |
| 4 | 401 | 23.0 | 5.1 s |
| 8 | 448 | 23.6 | 8.1 s |
| 16 | 612 | 37.1 | 14.4 s |
| 32 | 722 | 41.2 | 27.9 s |

For reference, vllm-metal on the same Llama file measured 210 / 370 / 609 / 664 tok/s at
1/2/4/8 agents (no prefix cache), and on Glimmer (its MLX 4-bit build, 20 GB) 17.9 / 66.8 tok/s
at 1/8 agents and 148.5 at 32.

## Next

1. **GGUF K-quants on the new path**: repack Q4_K/Q5_K/Q6_K (and the 2-bit GGUF's Q2_K/Q3_K)
   into the same tiled layout so the 2-bit Glimmer (12 GB, fits 16 GB) gets the NAX kernels.
2. **Prefill**: now NAX-bound (~30 TFLOP/s); left: multi-row causal attention for prefill
   chunks, larger chunks when no agent is decoding.
3. **Agent features**: session API (KV kept across turns), fork for subagents, tool-wait KV
   offload to SSD, disk prompt cache, group-aware scheduling, grammar-constrained tool calls
   with jump-forward, n-gram / DFlash speculation, int8 KV (long contexts).
4. **Server**: OpenAI-compatible HTTP with streaming, so agent harnesses (and
   `scripts/bench_openai.py`) talk to it directly.
5. **Pi 5**: NEON sdot kernels (port ggml's), repacked weights, cross-compile and run.
