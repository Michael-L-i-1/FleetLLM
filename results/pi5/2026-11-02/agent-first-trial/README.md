# First Muse Glimmer concurrent-request trial

The installed model is **Muse-Glimmer-30B Q2_K**, not Qwen or Muse Spark.
This is a first throughput probe on the Pi 5 with one, four and eight independent
requests. It does not test tool use, debate, GPQA accuracy or agent coordination.

| Requests | Aggregate decode tok/s | Average per request | Relative to one-request baseline | End-to-end tok/s | Batch wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.948 | 0.948 | 1.00x | 0.589 | 54.33 |
| 4 | 2.176 | 0.544 | 2.30x | 1.304 | 98.15 |
| 8 | 2.623 | 0.328 | 2.77x | 1.567 | 163.37 |

Four requests balance aggregate throughput and individual speed; eight give the
highest aggregate throughput in this short trial. These ratios normalize to the
single rainbow request, not a matched sequential run of all eight different tasks.

## Protocol

- Same model loaded once, four CPU threads, budget 14.8 GB, FP16 KV 250 MB,
  row budget 64, source pinning off, default 32-value activation scales.
- Existing native Q2/Q3/Q4/Q6 kernels, continuous batching, automatic exact-prefix
  sharing and verified prompt lookup (`--ngram --spec-k 15 --spec-rows 16`).
- No learned DFlash drafter, layer skipping, FFN sparsity or coarser activations.
- Eight distinct short prompts; greedy, fixed 32 output tokens per request,
  ignoring EOS as in the existing throughput benchmark. Outputs are truncated
  speed samples, not complete-answer quality assessments.
- Four-token warmup before measurements. KV is released between levels; prefix
  sharing operates within each level, without retaining prior-level cached KV.
- Decode timing starts after **every request has emitted its first token**.
  Tokens emitted earlier are excluded from its numerator. The window includes
  the tail as requests finish, so per-request is aggregate divided by N.
- End-to-end includes prompt processing and generation, excluding model load
  and warmup. First-token averages were21.62s /40.80s /55.80s for1/4/8.
- Shared-prefix credits were0/130/304 tokens; actual prefill rows66/131/219.
  Prompt lookup accepted0/4/7 proposals out of0/9/15; most throughput gain here
  is batching, not a copy-heavy speculative-decoding showcase.
- First request output token IDs matched its single-request baseline at 4/8.
  Other requests were not compared to separate sequential runs in this first trial.

## Memory and limitations

No timed swap growth or KV preemptions. Peak process RSS across loading/warmup
and measurements was 11.94 GB; minimum system available RAM 5.41 GB. The 250 MB
target KV pool fits these short contexts; this does not establish capacity for
eight long reasoning histories. One sample per level, no statistical intervals.

The first attempt was stopped by a 128 MiB swap-growth guard during model loading,
before any output. Its metadata is preserved in [loader-stopped](loader-stopped/).
The successful run separated loading/warmup from the timed guard; loading swap
growth was 0.82 MB, timed growth zero. No OS settings or inference defaults changed.

## Reproduce

```sh
cd ~/fleetllm
python3 bench/pi_agents.py --output ~/my-agent-trial
```

The wrapper takes the shared Pi benchmark lock for compilation and execution,
monitors memory, limits timed swap growth, and stops the child on failure/timeout.
It compiles the standalone probe against the existing CPU library. Results and
commands are in [warmed](warmed/); [summary.json](summary.json) includes source
hashes and timing definitions. Model identity matches the pinned standard Q2_K
file in [the single-agent setup](../glimmer-q2k/model.json).

No server is left running. The normal single-agent launchers are unchanged.
