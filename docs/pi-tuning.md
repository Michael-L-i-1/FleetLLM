# Raspberry Pi tuning and response quality

Keep hardware tuning, target precision and speculative decoding separate. The
shared engine still owns sequences, paged KV, prefix sharing and verification;
the Pi-specific launchers and experiment settings live outside that engine.

## Defaults and explicit choices

The CPU default uses the supplied checkpoint, int8 activations scaled over 32
values, FP16 KV, every decoder layer and every FFN feature. It does not prune
layers or activations. This is the comparison target, not full-precision HF
inference: the chosen weight file and activation arithmetic are quantized.

| Option | Effect on responses | Intended use |
| --- | --- | --- |
| `--ngram --spec-k 7 --spec-rows 8` | Greedy proposals are verified by the selected target | Copying/editing file contents or repeated tool text; little benefit for fresh prose |
| `--draft PATH --draft-kv-gb .064 --spec-k 2 --spec-rows 3` | Greedy proposals are verified by the selected target | Learned DFlash drafting; measure acceptance and drafting cost |
| `--activation-group 256` | Changes activation rounding; uses integer K-quant superblock accumulation | Opt-in CPU precision/performance experiment; weights and layers unchanged |
| `--ffn-sparsity .25` | Changes outputs: zero the lowest-energy 25% of 32-feature blocks **only at the FFN down-projection input** | Experimental block-sparsity baseline, not the full TEAL method; not selected for the Pi demo |
| `--skip-layers 8,20,32,44` | Removes complete decoder blocks; changes outputs | Explicit pruning experiment; model-specific indices, validated against its layer count |
| `--kv-int8` | Changes stored history precision | About half the FP16 KV storage, with scale overhead; evaluate long-context quality separately |

`--activation-group 32` is the default. Widths not divisible by 256 retain
32-value scaling even when 256 is requested. Configure CPU options before any
forward; changing target arithmetic requires discarding old KV. Unsupported
backend/invalid approximation settings fail instead of being silently applied.

Verification preserves the **selected target's** greedy sequence: enabling
speculation alongside approximate target arithmetic does not recover the
original target's answers. No setting enables pruning automatically.

`--spec-k` now applies to both prompt lookup and DFlash. Zero selects the mode's
default; it does not disable a mode enabled by `--ngram` or `--draft`.
`--spec-rows` limits total verification positions across active agents, including
the anchor. Batching and speculation share this budget.

## Current Pi choices

- **Glimmer Q2_K:** retain the existing model and use verified prompt lookup for
  copy-heavy editing. Lookup depth 15 reached 2.31 tok/s with exactly the same output
  tokens as ordinary decoding; fresh prose remained about 0.96 tok/s.
- **Optional Qwen3-8B Q4_K_M:** about 2.25 tok/s without speculation or pruning on
  the tested requests. This changes the model; it is not an engine-only speedup.
  Six objective task checks passed, but this small suite does not establish
  general coding, reasoning or long-running-agent quality.
  Verified prompt lookup reached 7.27 tok/s on the file edit, with all seven
  tested output sequences identical to ordinary Qwen decoding.
- **Experimental Glimmer activation group 256 + prompt lookup:** one edit reached
  2.39 tok/s. Calibration perplexity rose from 11.1266 to 11.2235 over 158 scored
  tokens (+0.87%). Keep this option explicit; the sample is too small to promise
  a universal quality bound.

The layer/FFN pruning trials did not justify a demo preset. In particular,
skipping eight layers introduced a false explanation; skipping nearly half
produced unusable text. The FFN block-pruning prototype changed responses with
almost no measured speed gain. Their controls remain experimental and off.

Full commands, raw answers and rejected trials are in the
[optimization sweep](../results/pi5/2026-11-02/optimization-sweep/README.md).

## Reproduce and extend

Run on the provisioned Pi, from `~/fleetllm`:

```sh
python3 bench/pi_sweep.py --profiles exact --output ~/my-baseline
python3 bench/pi_sweep.py --profiles ngram7 --tasks bench/pi-edit.jsonl --output ~/my-edit-test
python3 bench/pi_sweep.py --profiles exact group256 --ppl --output ~/my-ppl-test
python3 bench/check_pi_quality.py ~/my-baseline ~/my-edit-test
# Compare a worktree build while keeping the same model, tasks and runtime settings:
python3 bench/pi_sweep.py --binary ~/fleetllm-experiments/example/build/fleet --profiles exact --output ~/my-worktree-test
```

`fleet eval --tasks tasks.jsonl` loads a model once, uses fresh sequences, and
records token IDs plus decode-only draft/target times. Task definitions include
expected JSON, exact answers or Python ASTs; generated code is not executed.
The truncated rainbow task measures speed and is not a complete prose-quality
assessment. Use longer responses and task-specific evaluations before adopting
an approximation for important work.

`bench/pi_sweep.py` is Linux/Pi orchestration: named experiments, a benchmark
lock, memory/swap guard and saved commands. The underlying `fleet eval` command
is backend/model independent. The default inference binary has no new runtime
package dependencies.

The sweep uses tasks relative to its own checkout; `--binary` can select an
isolated build. `--threads`, `--row-budget` and `--kv-gb` make hardware settings
explicit in saved commands. Serialize compilation as well as inference during
timing runs, so work in another worktree cannot compete for the Pi's cores.
`bench/pi-reasoning.jsonl` adds three self-written science/probability questions
with short explanations. Its `json_fields` checks grade the answer field only;
the explanation still needs review. This is a smoke test, not GPQA.

`bench/cpu_kernels.c` optionally loads the existing ggml CPU shared library to
compare identical weight matrices. This is a development benchmark dependency
only. Its multi-row ggml baseline repeats row-dot calls; it is **not** a
comparison against every optimized llama.cpp batched kernel. Microbenchmarks
exclude model loading and activation preparation; full task timings include
per-token work. Prefer full-model results when choosing a preset.
The original largest-matrix selector can choose `token_embd.weight` for Q2_K;
that tests a dense operation on real weights, whereas inference uses embedding
row lookups. Follow-up kernel experiments must also select streamed decoder
matrices and confirm any improvement end to end.

T-MAC lookup kernels, trained early-exit models and full TEAL-style sparsity
remain separate research directions. The present experiment does not implement
or rule out those systems, and their published gains are not independent
multipliers over the current baseline.
