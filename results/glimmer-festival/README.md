# Glimmer festival website: eight pages, three real rounds

All modes use **Muse Glimmer 30B affine four-bit** on the same Mac GPU. This demo builds a festival site, applies a family-friendly stakeholder revision, then adds keyboard/accessibility polish. The model actually edits the files. Protected browser checks gate success; recordings show real intermediate states.

## Results

The single-agent attempt did not finish: the common 50,000-character client-history policy repeatedly trimmed prior work, after which the model repeated edits and reads. It reached 219/251 checks. This is a limitation of this bounded agent harness, not proof that vLLM or a full-context single agent cannot solve the task. **No speedup ratio is claimed against that incomplete attempt.** Both eight-agent runs had zero context-compaction events and passed all final checks.

One server runs at a time. One trial per mode is delivered. The original nine-trial counterbalanced plan is retained, but six extra repetitions were deferred after the user asked about elapsed time, before the single-agent outcome was known. The task, presets and first cohort were unchanged. Failed or incomplete attempts remain in the data and have no time-to-success.

| Mode | Successful / completed / planned | Median time to pass among successes | Successful range | Pooled output tok/s | Max sampled model GiB |
|---|---:|---:|---:|---:|---:|
| Fleet · eight agents | 1 / 1 / 1 | 418.0s | 418.0–418.0s | 63.4 | 18.19 |
| Default vLLM-Metal · eight agents | 1 / 1 / 1 | 762.8s | 762.8–762.8s | 37.2 | 18.17 |
| Default vLLM-Metal · one agent | 0 / 1 / 1 | — | — | 11.4 | 18.04 |

Output tok/s includes reported reasoning and code tokens divided by total task wall time, including tools, checks and repairs. It is aggregate workflow throughput, not per-agent decode speed or useful-code throughput. Startup and an unrelated warmup are excluded. Time-to-pass medians exclude failed trials, so interpret them alongside success counts. One trial per mode describes this demo, not broad model quality or statistically established general superiority.

| Trial | Final checks | First final-round pass | Total task seconds | Output tokens | Output tok/s | All rounds passed |
|---|---:|---:|---:|---:|---:|---|
| fleet-eight-r1 | 251/251 | 418.0s | 426.8 | 27079 | 63.4 | True |
| vllm-eight-r1 | 251/251 | 762.8s | 771.2 | 28688 | 37.2 | True |
| vllm-single-r1 | 219/251 | — | 1389.9 | 15799 | 11.4 | False |

## Watch and inspect

- [Actual three-lane development video](video/comparison.mp4) and [poster](video/poster.png). All lanes use the same elapsed-time axis and acceleration; finished lanes freeze. This is a sampled timelapse, not a fabricated simulation or continuous screen recording.
- [Generated website gallery](website/index.html), preserving the actual first-repetition outputs without manual design edits.
- [Per-trial metrics CSV](primary-summary.csv), [JSON including per-round metrics](primary-summary.json), and [aggregate results](primary-aggregate.json).

## Serving configurations

Hardware is an Apple M5 Pro with 20 GPU cores and 48 GB unified memory. The target checkpoint is `mlx-community/Muse-Glimmer-30B-4bit` revision `3e7677d7a40d348a3daba263a2b1c0aa41910710`, affine four-bit/group 64. Both engines load these exact weights. Native runtime arithmetic differs: Fleet FP16 versus vLLM/MLX BF16.

**Fleet agent preset:** native Metal/NAX, continuous batching, paged KV, exact prefix reuse, target-verified n-gram/prompt lookup (up to 15 proposals, 128 verification rows), and the validated shared-prefix SWA boundary fix. It has a 28 GB runtime budget with a 4 GB target KV cap. Learned DFlash drafting and approximate SPAN are not enabled in this selected preset. This is a documented agent preset, not a claim that all flags are current global engine defaults.

**Default vLLM-Metal 0.32.0:** performance flags and tuning environment overrides are absent. The same server command handles one or eight clients. Native default prefix caching and chunked prefill remain enabled; compiled MLP is opt-in and left unset. The default cache allocator profiles available machine memory, so effective pool sizes can vary by trial and are retained in server logs. This is a default-deployment comparison on one machine, not a matched-KV-budget kernel experiment. The only nonperformance launch settings select local binding, a model alias, text-only loading and the Glimmer reasoning parser.

The installed Metal package requires the isolated Glimmer correctness overlay in `demos/agent-task-bench/compat/`; both vLLM modes use the same patch, and the global installation is unchanged. Therefore these are default performance settings plus the required model-support correction, rather than an entirely unmodified package. Sixteen reference checks from the prior comparison cover the patch; its hashes are recorded.

## What changed in Fleet

Glimmer sliding-window attention could overshoot a shared-prefix endpoint while prefilling. Trimming then released an older page required by waiting agents, forcing them to repeat prefill. The opt-in `--shared-prefix-coalescing` clamp stops at the waiting boundary before forking. It changes scheduling/page ownership, not attention math, and is currently validated without a learned drafter.

The actual-engine CPU ASAN/UBSAN fixture reduced an unaligned eight-way prefill from 48,128 to 6,128 rows, with identical outputs and retained KV. Aligned/unaligned/staggered boundaries, copy-on-write, page references, cancellation, leader fallback and constrained memory were checked. Darwin leak sanitization is unsupported; explicit pool reclamation was verified with leak detection disabled.

The matched real-Glimmer JSON-edit pilot used the same binary, input requests, quantization, memory and lookup settings, changing only the coalescing flag: **136.52s → 44.95s (3.04×)**, 7,815 completion tokens per arm, all eight full messages identical, all eight requested edits valid. That is a Fleet-versus-Fleet mechanism pilot, not a 3× competitor or full-website claim. See `setup/json-edit-pilot-summary.json`.

A separate generation-phase checkpoint prototype increased reuse but failed strict first-response equivalence, so it was excluded. Earlier ambiguous copy-pilot prompts returned raw HTML rather than executable tool actions; those setup attempts are retained, not scored as agent successes. The actual harness repeats the JSON action contract at every round.

Later-turn reuse remains imperfect. A read-only audit identified stale, unusable cached leaders as a possible cause of some misses; individual trace events do not record enough leader-selection data to prove every cause. The frozen benchmark code was not changed after the trials started. The first Fleet trial reused context in all 20 third-round calls, but only six of 20 second-round calls.

## Task and fairness

Each fresh starter contains eight plain editable HTML fragments with protected assembly/navigation/JavaScript and a supplied local SVG. Round 1 builds eight pages and working local schedule filters, ticket/volunteer forms and FAQ; round 2 changes the action theme and introduces family advisories/page-specific updates; round 3 adds genuine keyboard focus and accessibility help. Browser acceptance is cumulative: 187, 219 and 251 checks. A temporary independent feasibility implementation passed all three stages, then was removed before model runs. The starter fails 136 of the 251 final checks.

Both eight-agent arms use identical disjoint assignments and tools; the single agent owns all eight files. Persistent histories survive between rounds. All modes receive the same natural shared brief once (not padded or duplicated), the same stakeholder requests at the same round boundaries, and the same guarded read/write/check/done protocol. Logical decomposition is supplied by the common harness; the demo is not evidence of a novel autonomous planner.

Common limits: temperature 0, low reasoning, 4,096 tokens per response, 108 calls/96k completed-token admission threshold/2,700 seconds per task. Each round has an 800-second ceiling; eight workers receive up to three calls each, then an integration agent receives up to 12. Single-agent rounds allow up to 36 calls. Three consecutive invalid/capped actions stop the current agent; partial output is preserved and later rounds can repair it. In-flight usage can overshoot the token admission threshold. A final cumulative pass determines task success; all-round success is separately reported.

Generated actions are executed; protected tests and source hashes guard the contract. Failed streams are not applied as partial edits. Full requests, outputs, errors, histories, round transitions, checks, generated workspaces and sampled resource data are durable. The first eight-worker request bodies are audited between engines; later histories naturally diverge.

Model and whole-run physical footprints are sampled at 1 Hz using Darwin process accounting, including charged GPU/IOKit memory. They may miss short peaks or overlap shared allocations and are not whole-machine RAM. RSS is retained but can undercount Metal memory. Browser capture is CPU-rendered and identical across modes.

Visual quality is reviewed separately from bounded functional checks. No generated design is manually beautified, and changed HTML is never labelled as a passing acceptance result. This is a synthetic local festival task, not an external coding benchmark. Its repeated-context/editing structure was deliberately chosen to exercise agent serving; results should not be transferred to arbitrary prose generation, CUDA vLLM, other models or Raspberry Pi.

## Reproduce

The original frozen hashes and nine-case plan remain in `matrix/plan.json`; the delivered three-case cohort and timestamped scope amendment are in `matrix/delivery-plan.json`. Actual order is Fleet/eight-vLLM/single-vLLM. The other six cases were not started, and are not failures or discarded results. Every delivered trial gets a fresh server and workspace; no task cache is donated across trials. Setup pilots are excluded. Model/code source and binary hashes are retained per run.

Run from the festival worktree root:

```sh
python3 results/glimmer-festival/make_report.py
```

The original checkout and its Raspberry Pi/GPQA changes are untouched by this work. All owned inference servers and recorders are stopped.

## Preserved raw artifacts

`archives/raw-text-traces.tar.gz` contains all non-PNG requests, responses, histories, events, generated sources, checker results, server logs and resource samples. Final sites, screenshots, MP4 and poster remain directly accessible. Complete raw runs including every captured/debug PNG are preserved locally in `local-archives/`, ignored to keep the Git repository small. `archives/manifest.json` records every file and archive SHA-256; every archived member was verified by reading it back. Unpacked `runs/` remain locally available and ignored.

To restore full frame data before recreating the video, extract each local archive from the worktree root, for example `tar -xzf results/glimmer-festival/local-archives/fleet-eight-r1.tar.gz -C results/glimmer-festival`. Rebuild the movie with `node demos/agent-task-bench/visual/compose.mjs results/glimmer-festival/video-config.json /tmp/new-festival-video`.
