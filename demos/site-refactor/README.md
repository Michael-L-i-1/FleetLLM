# Four-worker website refactoring demo

A plain local workshop website becomes a polished, responsive site. One model-driven
coordinator delegates **exactly four concurrent workers**, each with one file:

| Worker | File | Responsibility |
|---|---|---|
| Structure | index.html | Semantic structure and accessibility |
| Theme | theme.css | Typography, colors, shared tokens |
| Components | styles.css | Cards, navigation, inputs, buttons |
| Responsive | responsive.css | Page layout, spacing, mobile behavior |

The coordinator reviews the merged files, repairs integration issues, and runs checks.
The four workers use independent filesystem copies; only owned files are merged.
The coordinator is idle during delegation, so peak simultaneous model requests is four,
while the run has five agent identities. No external agent framework or hosted API is needed.
The model must actually issue the delegation; there is no scripted replacement if it fails.

## Setup and checks (no inference)

From the repository root:

```sh
npm ci --prefix demos/site-refactor
npm exec --prefix demos/site-refactor -- playwright install chromium
python3 demos/site-refactor/test_harness.py
python3 demos/site-refactor/run.py check
```

The fake-model integration test requires four requests to reach a barrier concurrently,
checks file integration and path rejection, and verifies summaries. The browser check
exercises search, empty search, newsletter submission, input labels, JS errors, and horizontal
overflow at 390px and 1440px; it captures screenshots. These checks do not score aesthetics
or certify WCAG compliance. Visually review the output. Browser HTTP requests are blocked.
`app.js` must remain byte-for-byte unchanged. Tools accept exact filenames, not paths;
there is no arbitrary command tool. Generated HTML/JS executes in headless Chromium, not
an OS security sandbox. Use this trusted local fixture rather than untrusted repositories.

## Proposed Mac configuration — not started

Verified machine: Apple M5 Pro, 20-core GPU, 48 GB unified memory.
Target: cached `mlx-community/Muse-Glimmer-30B-4bit`, revision
`3e7677d7a40d348a3daba263a2b1c0aa41910710`.
Its config specifies **affine 4-bit, group size 64**. The full checkpoint includes vision
weights; checkpoint disk size is not GPU allocation. Historical Fleet logs record 14.91 GB
of target 4-bit NAX tiles and 1.44 GB of drafter tiles. New-run memory/speed remain unmeasured.
Drafter: `meta-models/Muse-Glimmer-30B-assistant`, revision
`c676cce91658f9568aaf264a416211e2782735b7`.

After approval, start a fresh server, inspect its Metal/NAX startup logs, and warm it
with one untimed short request before launching agents:

```sh
./build/fleet serve \
  --model "$HOME/.cache/huggingface/hub/models--mlx-community--Muse-Glimmer-30B-4bit/snapshots/3e7677d7a40d348a3daba263a2b1c0aa41910710" \
  --draft "$HOME/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-assistant/snapshots/c676cce91658f9568aaf264a416211e2782735b7" \
  --device metal --name glimmer-mlx4 --budget-gb 26 --kv-gb 4 \
  --draft-kv-gb 0.5 --max-active 4 --admit \
  --row-budget 128 --ngram --spec-k 15 --spec-rows 64 \
  --reasoning low --max-tokens 4096 --port 8090
```

This enables GPU execution, exact prefix sharing (default), continuous batching,
prompt lookup and target-verified DFlash speculation. FP16 KV is deliberate for the first
quality baseline; no layer skipping, activation sparsity, or approximate span reuse.
The memory budget is an engine allocation budget, not a promise about whole-system RSS.
No model is loaded by setup, tests, viewer, or `check`.

Then launch the coding agents explicitly:

```sh
python3 demos/site-refactor/run.py run \
  --base http://127.0.0.1:8090/v1 --model glimmer-mlx4 \
  --label fleet-mlx4-dflash --output demos/site-refactor/runs/fleet-mlx4-dflash-01
```

In another terminal:

```sh
python3 demos/site-refactor/watch.py demos/site-refactor/runs/fleet-mlx4-dflash-01
```

Every run requires a new output folder. `runs/` is ignored by Git. The fixture stays intact.
Outputs: merged site, worker copies, full request/response transcripts, event log,
per-file patches, before/after screenshots, JSON summary, CSV summary.
The 4096-token limit is **per model call**, including reasoning; there is a 20-call limit
per agent, so long or malformed runs fail rather than receiving unlimited retries.
Partial artifacts remain after interruption, but automatic resume is not implemented.

## Comparison protocol

1. Smoke-test the MLX four-bit Fleet configuration above; inspect quality and tool reliability.
2. Repeat with DFlash disabled (omit `--draft`, `--draft-kv-gb`, `--spec-k`) to measure its benefit.
3. For the engine comparison, use one identical Q4_K_M GGUF in both Fleet and llama.cpp
   with Metal acceleration, the same brief, agent harness, concurrency, reasoning/token budgets,
   cache policy and checks. Enable supported speculative modes in an additional comparison.
   This Mac currently has no `llama-server` on PATH; provision and pin it before comparing.
4. Warm each fresh server before timing, keep other model servers stopped, and repeat runs.
   Restarting between repetitions prevents earlier runs from donating their task-specific KV.

MLX affine four-bit and GGUF Q4_K_M are different quantizations. Keep the fastest MLX showcase
separate from an engine-only speedup claim. The structured JSON action protocol is the same
for both engines because Fleet's Glimmer chat formatter currently lacks native tool schemas.
Report time to a checked result, completion/failure rate, repair calls and generated tokens;
include server logs for GPU allocation, cache/speculation metrics and decode throughput.
Harness timing includes agent requests, tool execution, integration and final checks, and
excludes fixture preflight/model loading/warmup. Use screenshots for manual design review:
a fast but broken or unattractive result is not a successful refactor.

## Current comparison policy (user update, November 3)

The main comparison is **optimized FleetLLM vs standard GPU serving**. The comparator
commands do not enable optional speculative decoding, prompt lookup, cache shifting/reuse,
or attention-kernel tuning. Keep normal batching (four concurrent requests), native kernels
and built-in defaults. Model formatting, low reasoning, context limits and a safe memory
budget are configuration requirements. In particular, vLLM may enable prefix caching by
default; we do not deliberately slow its normal behavior by overriding that default.
Earlier explicitly tuned vLLM n-gram runs are archived as extra experiments, not the main
baseline. No llama.cpp speculative run is scheduled under this updated policy.

`compare.py` starts exactly one owned server, waits for health, sends an unrelated warmup,
runs the harness (or fixed-worker replay), samples its process tree, and stops the server:

```sh
python3 demos/site-refactor/compare.py --label UNIQUE_LABEL \
  --server-command path/to/server-argv.json --base http://127.0.0.1:8090/v1
```

`--server-command` is a JSON argv array, not shell text. Configs and results are saved under
`results/mac/2026-11-03/site-refactor/`. `analyze.py RESULTS_DIR` computes task/worker rates,
latency, cache counts when supplied by the API, and mean in-flight request concurrency.
The replay uses identical real worker prompts and a 256-token cap; engines can stop early,
so generated-token counts are reported rather than assumed equal.

Span/DFlash clarification: the current engine gates span processing on `!e->draft`. Supplying `--reuse-spans` alongside `--draft` does not run span reuse. The demo did not measure a combined benefit from these two options.

### Qwen3.5 agent workflow experiments

Native Qwen3.5-4B Metal support and its limits are documented in `docs/qwen35.md`.
The isolated experiment uses the same four-file task and browser checks. See
`results/agent-workflows/README.md` for checkpoint revisions and recorded settings.

`run.py --context-prune` optionally masks confirmed superseded file observations
with a short receipt when at least 512 characters long, keeping message roles, immutable app.js and current feedback.
The transcript preserves original messages plus the masked indices. Active-history
hashes are recorded in context_edit events. This changes model context and is an
opt-in quality experiment, not an exact-output optimization.

The harness can recover one missing final JSON object brace after a complete
write payload, preserving generated file contents exactly. Each recovery emits a
protocol_repair event. Incomplete strings and other malformed syntax are rejected.
`--max-consecutive-errors 3` bounds repeated invalid actions; default 0 preserves
the previous unlimited-repair behavior within the overall step budget.

`workflow-replay.py` replays the first three recorded requests of each worker,
with four concurrent clients and a 128-token output cap. It does not execute model
outputs, change files, or grade task success. Histories are fixed even if responses
differ, so it isolates repeated inference/cache behavior from agent decisions.
The amount of generated output can still vary; compare token counts as well as
wall time. `compare.py --workflow-replay-source PATH` starts/stops the owned server
around this experiment and preserves its command, logs and resource samples.
