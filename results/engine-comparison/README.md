# Fleet / llama.cpp / vLLM-Metal serving comparison

Status: complete. All 15 initial trials, 5 tuning candidates and 18 fresh validation trials
with the selected configurations completed without failed or missing requests.
All benchmark-owned servers are stopped; see `cleanup.json`.

## Selected-configuration results

These are fresh validation measurements after tuning. Review and retrieval use
two repeats per engine; cap256 and 16 sessions are single sensitivity probes.
Throughput includes prefill, queueing, generated reasoning and scripted tool
waits. It is not decode-only speed. Capped generation is not a completed task.

| Workload | Fleet | vLLM-Metal | llama.cpp |
|---|---:|---:|---:|
| 8 sessions, cap64: generated tok/s | 85.78 | 73.15 | 48.92 |
| 8 sessions, cap256: generated tok/s | 200.83 | 132.32 | 121.33 |
| 8 sessions, 32 graded answers: seconds (lower is better) | 19.86 | 17.61 | 35.05 |
| 16 sessions, cap64: generated tok/s | 89.01 | 81.22 | 48.91 |

All engines returned 32/32 correct retrieval answers on both repeats, with the
same 108 generated tokens and no caps. Every validation trial also passed all six
separate short probes. All 32/64 requests in cap64 traces hit the cap and generated
exactly 2048/4096 tokens per engine. In cap256, actual generation was Fleet 8083,
vLLM 8167, llama 8192 tokens; rates use those actual counts. Most/all replies were
still capped, so equal quality or completed review utility is not established.

Fleet leads these capped-generation traces: approximately 1.17× vLLM and 1.75× llama
at 8 sessions/cap64, and 1.10× vLLM/1.82× llama in the single 16-session probe.
vLLM finishes this short-answer retrieval workload about 13% faster than Fleet.
This is not a universal engine ranking or a 3–7× competitive claim. GGUF versus
MLX quantization remains a confound for llama.cpp; Mac Metal results do not
predict vLLM CUDA performance. This experiment does not isolate which individual
cache policy or kernel causes the cross-engine difference.

| Repeated trace | Fleet seconds, min–max | vLLM seconds, min–max | llama seconds, min–max |
|---|---:|---:|---:|
| review | 23.57–24.19 | 27.58–28.42 | 41.26–42.49 |
| qa | 19.59–20.14 | 17.44–17.78 | 34.63–35.48 |

The 16-session trial keeps 8 GB cache capacity and 16-bit attention in all arms,
rather than comparing different KV precisions or giving one engine more cache.
It does not double aggregate throughput on this capped, prefill-heavy workload.

`request-audit.json` verifies identical request-body hashes and identical prompt-
token counts across the audited selected-engine arms. Raw replies and usage are
preserved; numerical/generation differences are not silently discarded.


## Question and limits

Compare current optimized Fleet with cache-enabled, GPU-backed llama.cpp and
vLLM-Metal on the same Apple M5 Pro (20 GPU cores, 48 GB RAM). This is a local
Mac comparison of vLLM's Metal/MLX plugin, not a benchmark of its CUDA backend.
The prior 3.5–7× Fleet-versus-Fleet gains are not competitor speedups.

All workloads are the already-frozen source-review/retrieval traces from
`results/throughput-exploration`. The primary traces have eight sessions with four growing
turns; source histories, delays, sampling temperature zero, and output caps
match across engines. Generated replies never become subsequent history or
executed edits. Review outputs are capped and do not establish coding success.
Retrieval has 32 graded answers and thinking disabled. Six separate short
quality probes follow each timed trace and are excluded from trace time.

## Models and numerics

Fleet and vLLM-Metal load the exact same local MLX checkpoint revision:
`mlx-community/Qwen3.5-4B-4bit@0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`,
affine four-bit weights with group size 64. llama.cpp loads the same base model
in `unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523`,
`Qwen3.5-4B-Q4_K_M.gguf`; its downloaded SHA-256 was checked against the published
LFS hash. GGUF Q4_K_M is a different quantization and weight mix. This is a
practical four-bit engine comparison, not identical weight tensors for llama.cpp.
Answer quality is measured separately; no general quality equivalence is claimed.

Attention caches use 16 bits in all arms: FP16 for Fleet/llama.cpp, backend-native
BF16 for vLLM. Recurrent state is FP32 in all arms. vLLM's native precision permits
its supported optimized GPU path; no attention approximation or layer skipping
is enabled. Request bodies explicitly set both `reasoning_effort` and Qwen's
`enable_thinking` template argument. QA setup prompt-token counts match Fleet.

## Cache envelope and configurations

All primary runs target approximately 8 GB of combined inference cache capacity;
model weights and runtime workspaces are additional. This is not a hard whole-
process memory cap. Record actual usage, not only nominal command-line values.

| Engine | Cache configuration | GPU / scheduling settings |
|---|---|---|
| Fleet |8 GB combined attention+recurrent budget; default recurrent fraction.25; prompt checkpoints, phase priority, state moves and concurrent prefix coalescing | Native Metal; HD256 NAX prefill enabled; admission, maxactive8,rowbudget128 |
| llama.cpp |2304MiB live attention KV +402MiB resident recurrent state +up to804 MiB private recurrent checkpoints +4096MiB RAM prompt-cache limit =7606MiB (7.98 GB) |33/33 layers offloaded; FlashAttention on; unified KV;8slots; context73728; logicalbatch2048/physical512;2 checkpoints/slot, minimum spacing0; idle-slot saving/clearing disabled; prefix and RAM prompt caching remain enabled |
| vLLM-Metal |448grouped blocks, startup-verified7.44GiB (~7.99GB) hybrid backing; automatic prefix caching and align-mode recurrent checkpoints | Metal GPU/MLX; native NAX and decode pipeline retained; compiled quantized MLP enabled;8sequences; batchedtokens2048; modelcontext16384 |

The installed vLLM-Metal0.32.0 accepts but ignores `--kv-cache-memory-bytes` on
this hybrid path. The setup trial requested4GB but allocated11.75GiB. That trial
is excluded. A supported explicit block-count override gives the verified budget;
224blocks was checked at3.72GiB before choosing448for the8 GB comparison.

llama.cpp's separate RAM prompt cache was initially too small to hold some whole
prompt snapshots. The rejected-snapshot warnings are preserved in the setup log.
The primary 4 GiB RAM allowance avoids that simple misconfiguration. A physical
prefill batch of 1024 (logical 4096) was also tested on retrieval; it did not improve
over 512, so 512 was retained. No competitor's prefix caching is disabled for the
comparison. No speculation is enabled in any primary arm.

## Measurement protocol

One benchmark-owned GPU server runs at a time. Each trial starts a fresh server,
warms8 concurrent unrelated ~2k-token prompts with cap 16/thinking disabled, then
runs the measured trace and separate short probes. Startup and warmup are
excluded. Frozen-context prefixes are cold at trace start; the warmup uses a
different system prompt and source. Tool waits and request queueing are included.
Selected review/retrieval each have two fresh trials per engine in differing orders;
cap 256 is a single sensitivity trial per engine. All attempts are retained.

Per-call JSON contains actual request bodies, content, reasoning, finish reasons,
usage and latency. Incremental events survive interruption. An unfinished or
failed trial is never counted as successful throughput. Configs include hashes,
exact arguments, environments, source revision and completion. Per-engine cache
hit reporting can differ; timing and API work counts are primary measurements.

`resources.csv` sums process RSS and can double-count shared pages; it is not
complete GPU/Metal physical-memory accounting. Its maximum covers the server
lifecycle, including startup/warmup, rather than only the timed trace. Startup allocation logs are used
for cache-capacity checks. `environment.json` records platform, AC/thermal
observations; no thermal warning does not prove fixed GPU clocks. This is a
live desktop, not an isolated lab machine. Two repeats are not a confidence
interval or proof of a universal ranking.

## Sources and reproduction

- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/6cb3e672e8d10fd4feab1a048bf2967c40623133/tools/server/README.md)
- [vLLM-Metal configuration](https://docs.vllm.ai/projects/vllm-metal/en/latest/configuration/)
- [vLLM-Metal macOS benchmarking guidance](https://docs.vllm.ai/projects/vllm-metal/en/latest/benchmarking-macos/)

Installed behavior and startup logs take precedence over newer online docs.
`environment.json` and `package-versions.json` pin the local versions. All server
configs and exact per-trial commands are in `configs/` and `matrix/plan.json` (initial) and `validation-matrix/plan.json` (selected).
`bench/engine_comparison.py` owns one server; `bench/throughput_workflows.py`
provides the common streaming workload runner. Regenerate primary summaries with:

```sh
python3 bench/summarize_engine_comparison.py results/engine-comparison \
  --plan results/engine-comparison/validation-matrix/plan.json --prefix validated
```


## Tuning and selection

The independent installed-source audit in `competitor-audit/` found no obvious
vLLM handicap and identified llama.cpp's idle-slot policy as a useful candidate.
The current llama server saves idle slots to RAM and clears their unified-KV
state on each new task. `--no-cache-idle-slots` retains the resident state; it
**does not disable prefix caching**. Selection is recorded in
`selected-configs.json`; all candidate QA trials returned 32/32 correct answers
and 6/6 short probes, with no caps/errors.

| Candidate, eight-session retrieval | Seconds | Decision |
|---|---:|---|
| llama resident idle KV,2 checkpoints |34.43|Selected for fresh validation |
| llama default idle policy,4 checkpoints |47.67|Rejected |
| llama resident idle KV,4 checkpoints |34.46|No benefit over 2; rejected |
| vLLM prefill batch 4096 |17.32|Near-flat versus 17.13 s initial mean; keep 2048 |
| Fleet row budget 512 |19.92|Near-flat versus 19.56 s initial mean; keep 128 |

Four llama checkpoints used 3292 MiB RAM cache instead of 4096 MiB, preserving
the same 7606 MiB nominal cache envelope. The selected two-checkpoint configuration
retains 4096 MiB. Fresh validation uses a single fixed selected preset per engine
across all workload types; it does not choose a different winner per question.
The main result is never selected from the fastest tuning trial.

The 16-session probes keep approximately8 GB combined cache capacity and 16-bit
attention KV. llama uses 126976 positions (3968 MiB live KV),804 MiB resident state,
up to1608 MiB checkpoints, and1248 MiB RAM cache, totaling 7628 MiB. Fleet uses the
same8GBbudget with recurrent fraction 0.5. vLLM keeps 448 blocks and raises its
sequence limit to 16. Each is a single scaling probe, not a repeated estimate.
Thirty-two sessions were not tested in this common8GB/16-bit envelope.


## Preserved artifacts

`validated-comparison.csv` and `validated-aggregates.json` are the final selected-
configuration results. `comparison.csv` and `aggregates.json` retain the initial
configuration results. Setup/tuning trials are excluded from both final aggregates.
All 38 planned trials and the earlier setup attempts remain under `runs/`.
Configs and summaries are plain files. `allocation-and-route-checks.json` keeps
key GPU/allocation evidence directly readable. Full logs, lifecycle resource
samples and warmups are in the archive. Diagnostic/CSV line endings are normalized.

Per-request JSON, streaming events and full diagnostics are compressed in
`raw-artifacts.tar.gz`: 2,164 members, 38.8 MB uncompressed,
3.35 MB compressed. `raw-manifest.json` records
every member's SHA-256 and the archive hash; all members were read back and verified.
Unpacked local copies are ignored to avoid duplicating large prompts in git.
Frozen workload inputs remain in the preceding experiment's tracked archive.
Restore both sets before reproducing exact saved commands:

```sh
tar -xzf results/throughput-exploration/raw-artifacts.tar.gz \
  -C results/throughput-exploration
tar -xzf results/engine-comparison/raw-artifacts.tar.gz \
  -C results/engine-comparison
```

Run commands from the repository root. Use fresh output labels/directories to
avoid overwriting evidence; existing output directories deliberately cause failure.
The runner sources for each serial phase are retained next to their frozen plans.
The llama binary/source path is local to this Mac; rebuild the pinned commit and
adjust that executable path on another machine. These results belong to branch
`experiment/agent-workflows`; this work did not modify the original checkout or its Pi/GPQA files.
