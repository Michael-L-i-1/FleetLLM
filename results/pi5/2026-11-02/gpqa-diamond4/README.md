# Four-question GPQA Diamond pilot on Pi 5

Muse-Glimmer-30B standard Q2_K, four concurrent independent questions, greedy,
low reasoning, 1,024 generated-token cap (reasoning and final answer combined).
The pinned OpenAI simple-evals evaluator supplies the questions, choice
permutations, prompt and grading. Deterministic four-example subset, one pass.
This is a timing/plumbing pilot, not a representative estimate of GPQA accuracy.

## Outcome

- Wall time: **37 minutes 23 seconds** (2,243.3 s), excluding model loading.
- Official score: **2/4 (50%)**.
- Both completed final answers were correct. Two other requests hit the token
  cap entirely in reasoning, with no final answer; the official grader scored
  those zero. No post-hoc answer extraction from reasoning was used.
- Generated tokens: **3,124**, including reasoning and header/channel tokens.
- End-to-end aggregate throughput: **1.393 tok/s**, including prefill and tail.
- Peak logged generation window: **2.7 aggregate tok/s**. Server logging windows
  are about ten seconds (step-aligned), so this is not a sustained whole-run rate.
- No timed swap growth or KV preemptions; throttle flags stayed zero.

| Question | Outcome | Generated tokens | Completion time from batch start |
| --- | --- | ---: | ---: |
| 1 | Token cap, no final answer | 1024 | 33.88 min |
| 2 | Token cap, no final answer | 1024 | 37.39 min |
| 3 | Correct, natural stop | 465 | 25.93 min |
| 4 | Correct, natural stop | 611 | 24.78 min |

## What was enabled

Native packed Q2/Q3/Q4/Q6 CPU kernels, four CPU threads, continuous batching,
paged FP16 KV (1 GB), exact-prefix sharing, memory-aware admission, and verified
prompt-lookup speculation (`--ngram --spec-k 15 --spec-rows 16`). Default target
activation scales remain 32 values; all layers and FFN features are retained.
The server shared 297 prompt tokens. Its last logged speculation counter was
778 accepted /1,949 proposed drafts in472 verifies (about40%); that snapshot
precedes the final few tokens and is not claimed as an exact final counter.

DFlash was **not enabled**, and its value on this GPQA workload is unmeasured.
Span reuse was **not enabled** (`0 tokens reused in0 spans`). GPQA questions
largely have different bodies; repeated prefix instructions already share KV.
There are no repeated file/tool-output blocks or multi-turn sessions in this
pilot, so this does not exercise the main span-reuse use case. Int8 KV, SSD KV
swapping, target pruning, coarser activations and rejected LUT kernels were off.
This was not an all-options optimization sweep or an ablation study.

The four requests did not communicate, debate, vote, share answers or delegate
work. The engine coordinated scheduling, weight reuse and shared prefix pages.
Each question retained its own reasoning history and was graded separately.

## Timing interpretation and concurrency

The earlier short trial measured2.18/2.62 aggregate decode tok/s at4/8 requests
with32 generated tokens. This pilot includes long reasoning, prompt processing
and a tail with only one or two requests active. Those timing definitions and
workloads differ. Raising client concurrency does not multiply physical compute.

The current1GB KV allocation contains1,176 full-attention pages and1,173 SWA
pages,16 tokens per page: roughly18,800 token slots in each pool. Conservatively
assuming1,500-token histories (prompt plus maximum output), this supports about
12 active histories before page rounding/sharing and admission details. Eight
similarly sized requests should fit, but eight GPQA requests were not tested.
Longer questions/output limits reduce this bound. More KV could raise the
memory-only ceiling into the dozens, without guaranteeing higher throughput.

A linear extrapolation from this tiny batch is about30.8 hours for198 questions
at the same cap. This is not a reliable full-run ETA: selected question lengths,
completion lengths, truncation, concurrency and caching can change the rate.
Increasing the token cap invalidates that estimate. No full run was started.

## Artifacts and reproduction

- [pilot.json](pilot.json): official summary; [pilot.html](pilot.html): question/answer/grade report.
- [pilot.events.jsonl](pilot.events.jsonl): starts, visible output milestones,
  progress character counts, full completed reasoning/final text, usage and grades.
- [analysis.json](analysis.json): per-question cap/answer/timing analysis.
- [server-run.json](server-run.json) and [server.log](server.log): exact server
  command, RSS/swap guards and performance logs.
- [launch.json](launch.json), [pilot.manifest.json](pilot.manifest.json),
  [source-manifest.json](source-manifest.json): versions, settings and input IDs.
- [validation/transport-check.log](validation/transport-check.log): synthetic local
  four-stream test of final-only grading, usage, truncation and question identity.

The eval client and SSH tunnel ran on the Mac; only inference ran on the Pi.
Server: CPU4threads, budget14.8GB, FP16 KV1GB, row budget64, no pinning,
`--ngram --spec-k15 --spec-rows16 --reasoning low --max-active4 --admit`.
The complete launchers are archived as [run-client.py](run-client.py) and
[server-guard.py](server-guard.py); run-client.py expects a separately started
server guard. The measured wrapper source is saved as [gpqa-wrapper.py](gpqa-wrapper.py).
The official harness stayed at652c89d0ca9df547706735883097e9537d40dc47, unchanged.

The model server and SSH tunnel were stopped after completion. `scripts/pimon.py`
was not changed; the event file is a future viewer input, not an implemented tab.
