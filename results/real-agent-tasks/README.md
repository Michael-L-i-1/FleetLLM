# Muse Glimmer 30B: real agent coding tasks

Fleet won the website task in this trial: **1.40× faster to a passing result than eight-worker vLLM-Metal**, and **3.01× faster than single-agent vLLM-Metal**. The latter comparison includes orchestration, not just engine changes. The Rust migration was mixed: Fleet passed, eight-worker vLLM exhausted its calls without passing, and **single-agent vLLM was fastest among successful runs**.

These are two small synthetic tasks designed for this demo, with actual generated edits, protected acceptance checks and saved intermediate states. The Rust task supplies a protected CLI shell and asks the model to port eight modules; this is not a full arbitrary repository migration or an external coding benchmark. One trial per task/mode supports a demonstration, not a general speed or quality claim.

| Task | Mode | Final checks | First passing result | Total task time | Output tok/s | Peak model GiB |
|---|---|---:|---:|---:|---:|---:|
| Website | Fleet · 8 workers | 63/63 | 187.7s | 190.4s | 61.1 | 19.97 |
| Website | vLLM-Metal · 8 workers | 63/63 | 263.6s | 265.9s | 40.5 | 18.05 |
| Website | vLLM-Metal · 1 agent | 63/63 | 564.4s | 566.7s | 15.8 | 17.84 |
| C → Rust | vLLM-Metal · 1 agent | 11/11 | 515.8s | 516.8s | 14.2 | 17.85 |
| C → Rust | vLLM-Metal · 8 workers | 7/11 | Did not pass | 1649.7s | 18.6 | 17.85 |
| C → Rust | Fleet · 8 workers | 11/11 | 908.4s | 909.5s | 24.9 | 19.93 |

Output tok/s = all reported completion tokens / total task wall time, including reasoning, reads, edits, repairs and checks. It is aggregate workflow throughput, not per-agent GPU decode speed or useful-code throughput. Loading and an unrelated warmup are excluded. The CSV also reports API-active throughput, latency percentiles, prompt/cache statistics, request counts and memory.

The failed Rust vLLM-eight trial consumed all 56 calls, produced 30,705 completion tokens, hit the response cap three times and ended at 7/11; its best intermediate result was 10/11. It has no time-to-success. Fleet used 37 calls/22,644 tokens and repaired its initial 7/11 result to 11/11. Single-agent vLLM used 23 calls/7,355 tokens. More parallel generation did not guarantee a faster correct program.

## Watch and inspect

- [Actual development timelapse](video/comparison.mp4): three sequential runs aligned to one real elapsed-time axis at 20× playback, with finished lanes frozen. Screenshots are sampled, not a continuous screen recording.
- [Final generated website gallery](website/index.html): all three sites and full-page screenshots, preserved without manual edits.
- [Full metrics CSV](primary-summary.csv) and [JSON](primary-summary.json).
- Raw requests, responses, generated workspaces, checker details, frame captures and resource samples are preserved in the raw-run archive described below.

All three final sites were visually reviewed: coherent cream/green nonprofit-cafe designs with different layouts, functioning navigation, menu search and signup. The tests verify bounded functional/responsive requirements, not professional design quality, WCAG certification or general coding competence. Fleet has some uneven section widths; the single-agent site has a plainer visit/map treatment. No design was manually improved for the video.

## Conditions and implementation

- Hardware: Apple M5 Pro, 20 GPU cores, 48 GB unified memory. All model runs were serial on the Mac GPU.
- Exact shared target: `mlx-community/Muse-Glimmer-30B-4bit`, revision `3e7677d7a40d348a3daba263a2b1c0aa41910710`, affine four-bit with group size 64. Target runtime arithmetic differs: Fleet FP16 versus native MLX/vLLM BF16. Four-bit weights are the same; responses are not expected to be bit-identical across engines.
- Both eight-worker arms use the same engine-neutral harness, JSON tools, disjoint ownership assignments, initial request bodies and integration policy. The single agent owns the whole task. Fresh copies of the same starters, no solution code supplied. Tests and assembly logic are protected.
- Temperature 0, low reasoning, 4,096 tokens per response, 56 calls and 2,700 seconds per task. The 96k token admission threshold uses completed usage and can be overshot by already active calls; all observed runs remain below it. Histories and generated token counts naturally diverge.
- Each target KV pool is approximately 4 GB. Fleet adds the learned Glimmer DFlash assistant (about 1.44 GB repacked weights plus 0.5 GB draft KV), k=3, plus n-gram drafting. This extra memory is included in the measured footprint. Approximate SPAN was not combined with the drafter because that combination is unsupported; recurrent Qwen-only switches do not apply to Glimmer.
- vLLM-Metal uses prefix caching, native Metal/NAX, compiled MLP and its native decode pipeline. Prompt lookup was slower in the coding pilot, so it was not selected. The installed DFlash loader supports a Qwen3 draft format, not this Glimmer assistant; this is backend/model coverage, not a Fleet-exclusive algorithm.
- Installed vLLM-Metal 0.32.0 required an isolated Glimmer compatibility patch. Both vLLM modes use it; the global package is unchanged. Sixteen reference tests cover attention normalization, gating, scaling, local/global windows and wrapper metadata. Details and the preserved strict-tolerance failure are in setup logs and the benchmark README.
- Sampled model-tree physical footprint is reported at 1 Hz during task time, including charged GPU/IOKit allocations. It is an approximate process-tree measure, not whole-machine RAM; shared allocations may overlap and brief peaks may be missed. RSS is also retained but substantially undercounts some Metal allocations.

The primary Fleet preset includes native Metal/NAX kernels, continuous batching, paged KV, exact prefix reuse and target-verified DFlash. These are general serving techniques with hardware/backend-specific implementations. The fixed-assignment agent harness itself is not a novel autonomous planner. These results do not establish a universal Fleet advantage over vLLM, or transfer to Raspberry Pi.

## Pilot selection and ablation

The eight-request coding pilot used identical Fleet response messages with and without its learned drafter: 189.6 versus 116.5 output tok/s (1.63×). vLLM native pipeline produced 59.6 tok/s; its tested prompt-lookup configuration produced 54.1 tok/s. The native calibration used 5,000 KV blocks, while the lookup pilot used 4,695 and disabled async scheduling; this tuning comparison does not isolate prompt lookup alone. All scored vLLM trials use 4,695 blocks. Pilots are setup evidence, not the real-task results above. Frozen configurations and selection rationale are retained in `configs/` and `selected-presets.json`.

A separately registered real Rust ablation removes only the learned drafter and its KV; n-gram drafting remains enabled. The user stopped further Rust work while it was in progress. Its partial traces are preserved with `cancelled.json`; it is excluded from completed comparisons and no ablation speedup is claimed. Different agent trajectories prevent treating it as a fixed-output kernel microbenchmark.

## Corrections and exclusions

The original website grader checked DOM/interaction behavior but missed an opaque pseudo-element hiding vLLM-eight hero text. A neutral pixel-contribution check exposed this. All three modes were rerun from a fresh corrected v2 starter; only those reruns appear above. The corrected task has 63 checks, including heading/CTA visibility at two widths. Five focused visibility regressions passed. Original v1 runs remain preserved and excluded, including their misleading automatic pass.

The original Rust attempt had a checker-output contract error: the manifest supplied a directory where the checker expected a JSON file. That attempt was interrupted and invalidated as infrastructure failure, not scored as a model failure. The manifest now supplies `{output}/details.json`; all three corrected Rust trials used this same correction with unchanged test semantics. The harness now stops on a crashed/non-JSON grader. Original manifests, logs and invalidation records remain preserved.

The six corrected primary configurations and task-file hashes were registered before their runs and rechecked afterward. A read-only audit found no blocking fairness defect. The four model-free harness tests cover concurrency, ownership/protection, partial-stream safety and grader failure; compatibility and visibility checks are preserved in setup logs.

## Reproduce

Run from this worktree root; exact commands, file hashes, source revisions and binary hashes are in `corrected-matrix/plan.json`, per-run configurations and setup records. See [harness documentation](../../demos/agent-task-bench/README.md). Do not reuse output directories for new trials. The original checkout and its Raspberry Pi/GPQA changes were not modified by this work.

```sh
python3 demos/agent-task-bench/report.py results/real-agent-tasks
node demos/agent-task-bench/visual/compose.mjs results/real-agent-tasks/video-config.json /tmp/new-comparison-video
```

## Preserved artifacts

`raw-runs.tar.gz` contains every raw run, including excluded/invalidated attempts and the cancelled ablation. `raw-runs-manifest.json` lists file sizes and SHA-256 values, verified by reading every archived member back. Unpacked local `runs/` are ignored to avoid duplicating binary captures and traces in Git. Restore with `tar -xzf results/real-agent-tasks/raw-runs.tar.gz -C results/real-agent-tasks` before regenerating reports or the video. Generated sites, final screenshots, MP4, poster, frozen plans and summaries remain directly accessible.

Further Rust work was stopped at the user’s request. All owned inference servers and samplers are stopped; port 8090 is free.
