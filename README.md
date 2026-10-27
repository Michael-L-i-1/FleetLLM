# cinfer

A C inference engine for running **many concurrent agents** (subagents, long
tool loops, Terminal-Bench-style runs) through one copy of a model on
constrained hardware: a Raspberry Pi 5 with 16 GB is the final target; an Apple
M5 Pro (Metal) is the development machine. It follows vLLM's core design
(paged KV, continuous batching, chunked prefill, prefix sharing) under a hard
memory budget, with agent-specific features to come.

Pure C11 plus one Objective-C file for the Metal host code. No runtime
dependencies. Kernel code ported from other projects is credited in place
(ggml/llama.cpp, MIT).

## Status (phases 1-2 complete)

| Piece | State |
|---|---|
| GGUF loader (mmap, zero-copy) | done |
| Tokenizer from GGUF metadata (byte-level BPE; llama3 / qwen2 / llama4 pre-tokenizers) | matches HF `tokenizers` 10/10 on all three families |
| Dequantization F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K..Q6_K | bit-exact vs gguf-py |
| Generic decoder: llama, qwen2, qwen3, muse-glimmer | correct output on all |
| CPU reference backend | correct, slow by design (Pi NEON kernels are next) |
| Metal backend | correct; fused mat-vec / fused tile GEMM / dequant-once + dense GEMM; flash-decoding attention |
| Paged copy-on-write KV (full + sliding-window pools), continuous batching, automatic prefix sharing | done; batched greedy outputs identical to sequential (8/8) |

## Build and run

```bash
make                     # Metal on macOS; `make NO_METAL=1` for CPU-only (e.g. Pi)
./build/generate --model glimmer --prompt "What is the capital of France?"
./build/bench --model llama1b --budget-gb 4 --levels 1,2,4,8,16,32 --gen 128
./build/bench --model llama1b --budget-gb 4 --check      # batched == sequential
python3 scripts/tokcheck.py MODEL.gguf tokenizer.json   # tokenizer vs HF
python3 scripts/dequantcheck.py MODEL.gguf              # dequant vs gguf-py
```

`--model` takes `glimmer`, `llama1b`, `qwen3` (resolved in the Hugging Face cache) or a path.
`--backend cpu|metal`, `--budget-gb N` (weights + KV + scratch must fit), `--row-budget N` (prefill chunk).

## Layout

```
src/gguf.*        GGUF reader
src/quant.*       reference dequantization (ggml port)
src/tokenizer.*   BPE + hand-written pre-tokenizers; src/unicode_data.c is generated
src/chat.*        chat formats (Llama 3, ChatML/Qwen, Glimmer channels)
src/model.*       generic decoder description (flags per architecture)
src/kv.*          paged ref-counted copy-on-write KV
src/engine.*      sequences, scheduler, prefix sharing, sampling
src/cpu.c         CPU reference backend
src/metal/        Metal backend (metal.m) and kernels (kernels.metal, embedded at build)
tools/            generate, bench, tokcheck, dequant_dump
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

## Next (optimization phase)

1. **Prefill** (biggest gap: Glimmer TTFT 2.4 s for ~225 tokens). Larger prefill chunks
   (256-512 rows) so dequantize-once amortizes; a faster dense GEMM (MLX steel-style tiles,
   M5 tensor units); multi-row causal attention for prefill chunks.
2. **Small/medium-batch decode**: cut the fused kernels' fixed cost (repack Q4_K/Q5_K at load
   into a simple affine layout as vllm-metal does; register-blocked tiles) so 2-32 agents cost
   close to one.
3. **Agent features**: session API (KV kept across turns), fork for subagents, tool-wait KV
   offload to SSD, disk prompt cache shared across a benchmark run, group-aware scheduling,
   grammar-constrained tool calls with jump-forward, n-gram / DFlash speculation, int8 KV.
4. **Server**: OpenAI-compatible HTTP (vendored JSON/HTTP helpers) so the vLLM benchmark client
   runs unchanged.
5. **Pi 5**: NEON sdot kernels (port ggml's), repacked weights, cross-compile and run.
