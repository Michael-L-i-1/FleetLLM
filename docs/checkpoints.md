# Checkpoints

## agents-v1 (2026-10-28, commit a376156)

Multi-turn agent sessions (`bench/agents.py`, 6 turns, shared context, tool
results appended, ~8k-token contexts), Llama 3.1 8B MLX 4-bit, 2.5 GB KV cap,
`--ngram --kv-int8 --swap-gb 24`, M5 Pro, vs vllm-metal 0.32 on the same file:

| Agents | FleetLLM | vLLM | Speedup |
|---|---|---|---|
| 8 | 57 s | 158 s | 2.76x |
| 16 | 93 s | 315 s | 3.40x |
| 32 | 182 s | 595 s | 3.27x |

Ideas this checkpoint rests on:

- **KV is agent state, not a cache.** Sessions keep their KV across turns
  (prefix forks, copy-on-write pages); under memory pressure only a session's
  exclusive pages go to SSD (shared prefixes stay resident), and the server
  admits a turn only when it fits, so nothing is recomputed or thrashed.
- **Agents share work.** One prefill for agents that fan out together
  (in-flight dedup); one global repeated-text index drafts tokens for every
  agent from what any agent has read or written (`--ngram`, exact).
- **Each row picks the kernel that fits its role** (input text vs sampled /
  drafted), so results stay identical however requests are batched:
  `attn_nax` (prompt / tool-result rows, neural-accelerator matmuls, one
  dequantized K/V tile per threadgroup), `attn_naxd` (a decode row, its drafts
  and its GQA heads share matmul tiles, so verifying drafts is nearly free).
- **8-bit KV** (per-token, per-head scales): 2x the sessions per GB, +0.04%
  perplexity.

Measured dead ends: more KV memory at 32 agents (compute-bound); 4-bit KV
(+0.7% perplexity at ~6 bits/value for no throughput gain); split-size tuning
and reduce-scatter scoring in decode attention.
