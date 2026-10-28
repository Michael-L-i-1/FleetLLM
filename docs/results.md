# Results

All runs on an Apple M5 Pro (48 GB). Raw output is in `results/`; how to
reproduce each table is given under it.

## GPQA Diamond through a real harness (Llama 3.1 8B)

`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` served by `./build/serve --budget-gb 8
--kv-gb 2.5` (4.22 GB weights + a 2.5 GB KV cap; 3-7 sequences were preempted and
recomputed when the cap was hit), OpenAI simple-evals GPQA Diamond (commit 652c89d),
`--max-tokens 2048`, on battery. Per-question transcripts in `results/gpqa/*.html`.

| Run | Agents at once | **cinfer** wall | vLLM wall | Speedup | cinfer tok/s | vLLM tok/s | Score cinfer / vLLM |
|---|---|---|---|---|---|---|---|
| all 198, greedy | 16 | **616 s** | 873 s | **1.42x** | 281 | 204 | 25.3% / 21.7% |
| all 198, T=0.5 | 16 | **547 s** | 878 s | **1.61x** | 276 | 171 | 27.3% / 24.2% |
| same 16 questions | 1 | **241 s** | 344 s | **1.43x** | 51 | 40 | 31.2% / 25.0% |
| same 16 questions | 16 | **59 s** | 80 s | **1.37x** | 213 | 150 | 31.2% / 31.2% |

Same model file (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`), same harness and settings,
matched KV (cinfer `--kv-gb 2.5` = 19,072 tokens; vllm-metal 0.32 `--gpu-memory-utilization
0.193` = 18,416 tokens, since it ignores `--kv-cache-memory-bytes`), prefix caching on in both,
AC power, run back to back (`evals/gpqa_suite.sh`). 16 agents finish the same work 4.1x faster
than one at a time on cinfer (4.3x on vLLM). Score differences are noise (about +/-3 points at
n=198) plus greedy paths diverging; both engines lose ~30% of questions to repetition loops that
hit the 2048-token cap. GPQA prompts share almost nothing, so this measures raw batching and
kernels, not the agent features (sharing, sessions). Greedy cinfer output is token-identical to
MLX's reference on the same weights (GPQA prompt, 200 tokens).

Reproduce: see [evals/README.md](../evals/README.md).

## Agent sessions (bench/agents.py, Llama 3.1 8B, 2.5 GB KV budget)

8 agents fanned out from one shared context (system prompt + ~3k tokens of
project files), 6 turns each: answer (<= 160 tokens), simulated tool call
(1.5 s mean), tool result (a real file, ~600 tokens) appended; each request
resends the whole conversation (final context ~8.2k tokens). Needs ~40k tokens
of KV against 19k (16-bit) / 38k (8-bit) available, so memory pressure is real.

| Engine / version | Wall | Median turn | p95 turn | Preemptions | Prompt tokens from cache |
|---|---|---|---|---|---|
| cinfer, morning (row-at-a-time attention) | 975 s | 91 s | 373 s | many | 77% |
| + row-grouped attention | 446 s | 30 s | 167 s | 16 | 79% |
| + 8-bit KV (`--kv-int8`) | **174 s** | 27 s | 36 s | 0 | 89% |
| + 512/1024-row prefill chunks | 171 s | 25 s | 37 s | 0 | 89% |
| 16-bit KV + SSD swap (`--swap-gb 8`) | 470 s | 33 s | 145 s | 0 (swap instead) | |
| + input rows' attention on the neural accelerators (`attn_nax`) | 152 s | 21 s | 32 s | 0 | 89% |
| + decode attention kernel (`attn_vec`, one row per threadgroup) | 111 s | 15 s | 23 s | 0 | 89% |
| + 4 positions per lane group, unrolled heads | **92 s** | **12.4 s** | **18.6 s** | 0 | 89% |
| **vLLM** (vllm-metal 0.32, 18.4k-token KV) | 158 s | 18 s | 42 s | | n/a |

cinfer now finishes the workload **1.72x faster than vLLM** with a 2.2x lower
p95 turn latency, on the same model file and a matched KV budget (8-bit KV
holds 2x the tokens in the same 2.5 GB, so nothing is preempted).

What closed the gap (8B, 8-bit KV; per-step GPU time from `CINFER_PROFILE_EACH=1`):

- **Prefill attention on the M5 neural accelerators.** A 512-row chunk at 4.6k
  context took 1330 ms, about 1.2 TFLOPS of attention (the 8x8 simdgroup
  matrix ops were near their ceiling, so better loads and splits changed
  nothing), against ~26 TFLOPS for the NAX weight GEMM. `attn_nax` runs QK^T
  and PV as MPP 16x32x16 matmuls, 64 rows x one head per threadgroup, K/V
  loaded from the pages straight into registers (ported from vllm-metal's
  `paged_attention_nax` / MLX `steel_attention_nax`). The same chunk now takes
  488 ms; prefill 539 -> 1178 tok/s at 4.8k tokens and 235 -> 807 tok/s at 16.7k.
- **Decode attention shaped for latency, not matrix units.** One decode row at
  10k context spent 41 ms/step in attention, though its KV is only ~0.7 GB;
  f16 KV (2x the bytes) and a 4-page working set ran equally fast, so the
  kernel was issue/latency bound. `attn_vec` (one row per threadgroup, lane
  groups on 4 consecutive positions, loads issued before the math, unrolled
  heads) brings it to 5.3 ms: decode at 10k context 17 -> 44 tok/s.
- **Kernel choice per row, not per batch.** Input rows (prompt, appended text)
  always take `attn_nax`, sampled and drafted rows always take `attn_vec`, and
  both split context at row-local boundaries, so batched outputs stay identical
  to sequential (check 8/8) and speculative outputs to plain greedy (8/8).
  Perplexity unchanged (8B: 19.6854 -> 19.6841; Glimmer, sliding window: 8.2507
  -> 8.2499). Attention partials get compacted per-kernel slots: a 512-row chunk
  at 100k context no longer sizes a rows x max-splits buffer (3.4 GB before).

SSD swap works (IO is cheap: 277 swaps, 13 s) but with 2-3x oversubscription
it only replaces recompute with churn; it needs session-level admission (park
agents in tool waits).

## GPQA subset with the agent-oriented features (16 questions, Llama 3.1 8B)

| Version | 1 agent | 16 agents | Tokens generated |
|---|---|---|---|
| vLLM | 344 s | 80 s | |
| cinfer baseline | 241 s | 59 s | 12.4k |
| + repeated-text drafts (`--ngram`, output identical) | 120 s | 36 s | 12.4k |
| + 8-bit KV + grouped attention | **121 s** | **32 s** | 16.4k (8-bit KV changes some outputs) |
| + NAX prefill / vec decode attention | 130 s | 36-42 s | 15.5-15.9k |

This subset does not separate the latest kernels: its prompts are ~200 tokens
(attention is a small share of each step), different kernels change some greedy
paths (token counts and repeated-text acceptance move with them), and the
16-worker wall time is set by the slowest truncated question decoding alone
(three runs of one build: 36 s, 42 s, and 36 s). The deterministic `bench`
(16 agents, `--ngram`, 512 tokens each) measures the same build at 510 tok/s
vs 368 with the previous decode kernel.

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

## Speculative decoding with Meta's DFlash drafter

`meta-models/Muse-Glimmer-30B-assistant` (5-layer block-diffusion drafter, bf16,
quantized to 4-bit at load: 1.44 GB). Each step the drafter reads the target's
hidden states after layers 1/13/25/37/49, denoises a block (the last token + mask
tokens) in one pass, and the target verifies all drafts in one batched forward;
the longest matching prefix plus one bonus token is kept and rejected KV is
truncated. Greedy outputs are identical to plain decoding
(`bench --check-spec`: 8/8 tasks).

Drafts per agent shrink with concurrency so a step stays near `--spec-rows`
(32) rows, the batch size the 4-bit kernels run at nearly flat cost; the
drafter's block shrinks with it (anchor + drafts instead of all 16 positions:
~10% fewer accepted tokens, half the drafter time).

Same workload as above (MLX 4-bit Glimmer, 128 tokens per agent, battery),
aggregate tok/s; "no spec" is the same build without `--draft`:

| Agents | no spec | **spec** | no spec, `--share` | **spec, `--share`** | tokens per verify (spec) |
|---|---|---|---|---|---|
| 1 | 17.2 | **73.0** | 18.2 | **90.8** | 7.0 |
| 2 | 31.8 | **112.5** | 35.2 | **162.1** | 8.5 |
| 4 | 59.4 | **82.8** | 67.2 | **101.0** | 3.9 |
| 8 | 98.4 | **129.0** | 118.0 | **158.7** | 2.9 |
| 16 | 155.9 | **168.7** | 239.4 | **301.5** | 1.9 |
| 32 | 210.0 | 198.6 | 413.6 | 408.9 | (few drafts) |

Agent i runs task i % 8, so the 1- and 2-agent rows are the code-edit and JSON
tasks (very predictable); from 4 agents on, the story and math tasks accept far
fewer drafts and, since every agent must produce 128 tokens, set the wall time.
Meta reports 1.8x for this drafter on an M5 Max (26.6 -> 50.2 tok/s, one stream,
diverse prompts). The "no spec" numbers above are higher than in the vLLM table
because greedy sampling now runs as a GPU argmax (no logits copy).

```bash
./build/bench --model glimmer-mlx4 --draft glimmer-draft --budget-gb 21 --levels 1,8,32 --gen 128 [--share]
./build/bench --model glimmer-mlx4 --draft glimmer-draft --budget-gb 21 --check-spec   # == plain greedy
```

## Early baseline (GGUF kernels, before the tiled path) (M5 Pro, AC power, 8 different tasks, 128 tokens each, greedy, no prefix sharing)

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

Commands:

```bash
./build/bench --model glimmer-mlx4 --budget-gb 18 --levels 1,2,4,8,16,32 --gen 128 [--share]
./build/bench --model glimmer-mlx4 --draft glimmer-draft --budget-gb 21 --levels 1,8,32 --gen 128 [--share]
python3 bench/bench_openai.py --base http://localhost:8077/v1 --model glimmer --levels 1,8,32 --gen 128   # vLLM
```
