# Results

All runs on an Apple M5 Pro (48 GB). Raw output is in `results/`; how to
reproduce each table is given under it.

## GPQA Diamond through a real harness (Llama 3.1 8B)

`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` served by `./build/serve --budget-gb 8
--kv-gb 2.5` (4.22 GB weights + a 2.5 GB KV cap; 3-7 sequences were preempted and
recomputed when the cap was hit), OpenAI simple-evals GPQA Diamond (commit 652c89d),
`--max-tokens 2048`, on battery. Per-question transcripts in `results/gpqa/*.html`.

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

Reproduce: see [evals/README.md](../evals/README.md).

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
