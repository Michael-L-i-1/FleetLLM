# Mac coding-agent measurement notes

Measurement notes; read README.md / analysis.json for the results.

- M5 Pro, 48 GB unified memory, 20 GPU cores, AC power. No thermal warning reported.
- One coordinator creates exactly four workers with exclusive file ownership. At most four
  model requests are active while the coordinator waits; integration is serial.
- Same site fixture, system instructions, max 4096 generated tokens/call (including reasoning),
  low reasoning, greedy sampling, 20-call bound per agent. Fresh server per measured run,
  identical unrelated READY warmup. Task timing excludes startup and fixture preflight.
- Engine-selected paths and generated edits can differ; task speed is time to functional checks,
  not proof of equal aesthetic quality or equal amount of reasoning. Compare saved screenshots.
- Completion token rates include reasoning and template/control tokens as counted by the API.
  Aggregate task rate is completion tokens divided by whole task wall time; server log windows
  and per-request rates are different measurements and must be labeled separately.
- Longest-output turn from each of the first Fleet run's four worker transcripts forms a fixed
  four-request replay. 256-token cap per request is a throughput probe, not completed edits.
  Replays start a fresh server and use the same unrelated warmup to avoid task-cache contamination.
- First two Fleet runs and first patched-vLLM task ran with a background GGUF download. No other
  inference engine or compiler ran concurrently with their timed task windows. This is exploratory
  single-machine evidence, not a controlled many-repeat performance claim.
- Fleet first target: pinned MLX affine four-bit, group 64. FP16 KV, 4 GB pool; DFlash (15 proposed
  tokens), prompt lookup, prefix sharing and admission/batching. First span run supplied the approximate span-reuse flag, but the engine disables span processing when DFlash is attached; it was inactive, with zero moved tokens. No layer skipping or activation sparsity.
- vLLM Metal installed 0.32.0 failed to serve Glimmer both with and without n-gram speculation.
  The isolated overlay copies installed vllm_metal and backports upstream Glimmer support from
  7f95a63fee91153cf4f0def3927119aa3e28704b and selective logits from 14554be (full hash in source
  metadata when finalized). No system installation was modified. Exact applied diff is retained.
  Upstream fix: https://github.com/vllm-project/vllm-metal/pull/1120
  Selective logits: https://github.com/vllm-project/vllm-metal/pull/1220
- vLLM uses identical pinned MLX weights, NAX/paged Metal attention, prefix caching, four sequences,
  max model length 16384, and n-gram speculation (5 tokens, lookup 2–5). Explicit template kwarg
  reasoning_strength=low is required; request reasoning_effort alone does not set it.
  Its KV allocation is 5.31 GiB / 107120 tokens; neither engine is capacity constrained here.
  Glimmer's assistant is not supported by this vLLM Metal DFlash implementation.
- First attempt at upstream tests inadvertently imported the source checkout instead of the
  overlay: 8 missing-native-extension errors, 4 pass. Those are setup failures, not numeric parity
  failures. A standalone copy of the test is used for the actual overlay test result.
- llama.cpp is built with Metal from the revision in llama-revision.txt. Embedded UI gzip was
  disabled to avoid a /tmp vs /private/tmp asset-path build failure. Inference source is unchanged.
- llama.cpp target is Q4_K_M; do not call it identical to MLX affine four-bit. A same-Q4_K_M Fleet
  replay provides an engine comparison; MLX vs GGUF remains a different-format comparison.
- Resource CSVs sample server-process-tree RSS/CPU each second. RSS does not capture total Metal
  allocation reliably; vLLM process-tree sums can double-count shared pages. Use startup allocation
  logs for GPU/KV sizes, and do not rank memory efficiency solely from these RSS samples.

## Superseding comparator policy

The user requested standard serving with Fleet-specific optimizations only. The primary
vLLM and llama.cpp commands now omit added speculation, prompt lookup and cache-reuse
settings; llama.cpp also uses its default batch/microbatch and attention settings. Native
GPU acceleration, ordinary four-request batching and built-in defaults remain enabled.
vLLM standard resolves prefix caching to true by default (observed in its startup config),
which is retained. No llama.cpp DFlash run was performed; its converted assistant is unused.
The completed tuned-vLLM experiment is supplementary. See baseline-policy.json.
The isolated compatibility-overlay tests pass all 12 upstream Glimmer checks; the earlier
source-checkout missing-extension failures are superseded by vllm-compat-overlay-tests.txt.
