# Glimmer Q2_K on Raspberry Pi 5: one-agent CPU tuning

Standard Bartowski Q2_K replaces the earlier Unsloth UD-Q2_K_XL deployment.
Pinned source, size and verified SHA-256 are in `model.json`. The Pi has 16 GB
RAM and four Cortex-A76 CPU cores. All inference is local CPU inference.

## Engine changes

- Exact Q2_K repack into four-row tiles: two-bit codes remain packed, with
  original per-16-value scales/minima. Same storage size as the source.
- Exact Q3_K repack and integer dot products too: standard Q2_K still contains
  3.579 GB of Q3_K tensors. Its tiled metadata uses 116 rather than 110 bytes
  per 256 weights, while retaining three-bit codes.
- All streamed matrices in this checkpoint now use native integer kernels;
  no float row-dequant fallback remains. Streamed CPU tiles occupy 10.79 GB,
  versus 10.58 GB in the source (embedding is not scanned each decode step).
- On Linux, discard each redundant GGUF source range from the mapping and file
  cache as soon as it is repacked. This avoids the transient duplication that
  triggered the memory monitor on preliminary attempts. Their logs are kept.
  The draft BF16 checkpoint is likewise released immediately after quantization,
  before target repacking begins.

Repack checks preserve the original dequantized values exactly. Computation
uses int8-quantized activations, as in the existing optimized CPU backend;
this is not a claim of bitwise equality to float-activation reference inference.

## Method

`bench/pi_glimmer_q2.py` runs one process at a time, one agent, greedy sampling,
80 prompt tokens, 32 generated tokens, FP16 KV, a 250 MB target cache, row budget
64, and no source weight pinning. DFlash has a separate 64 MB cache. Logs split
prompt processing from generation; generation measures the 31 tokens after the
first. These are short tuning trials, not a representative agent evaluation.

The earlier UD-Q2_K_XL run measured 0.109 tok/s plain and 0.182 tok/s with DFlash
on a shorter Paris question. Both quantization and kernels changed here, so a
ratio against those results is not an isolated kernel speedup measurement.

## Validation

- Exact repack against the independent GGUF dequantizer for Q2_K and Q3_K;
  storage-size checks and dense matrix products against dequantized weights.
- Native versus portable kernels, all seven formats, batch sizes 1–17, row
  partitions, padded final groups, activation rounding/zero/tie checks.
- Address/undefined-behavior sanitizer kernel checks on the Mac.
- Independent dense PyTorch DFlash checks and 24 engine acceptance/rejection
  checks, covering FP32/FP16/int8 KV and batched versus sequential behavior.

The coherent text in these short runs does not establish retained coding or
agent-task accuracy after switching quantization.

## One-agent results

| Setting | Prompt processing | Generation | Peak RSS |
| --- | ---: | ---: | ---: |
| Plain, 4 threads (two trials) | 30.85–31.00 s | 0.93 tok/s | 11.94 GB |
| Plain, 3 threads | 41.13 s | 0.80 tok/s | 11.94 GB |
| Plain, 2 threads | 60.21 s | 0.57 tok/s | 11.94 GB |
| DFlash, 1 draft token | 31.21 s | 0.98 tok/s | 13.89 GB |
| DFlash, 2 draft tokens | 31.07 s | 0.99 tok/s | 13.89 GB |
| DFlash, 3 draft tokens | 31.07 s | 0.82 tok/s | 13.89 GB |

Every successful Fleet run produced exactly the same 32-token text, and every
successful run had zero observed system-wide swap growth. No throttling was
reported. DFlash acceptance was 66.7% / 50.0% / 30.4% for 1 / 2 / 3 proposals.
These are short trials on one prompt; the ~6% speculative decode gain is small
and needs more samples to distinguish from variation.

**Recommended short-request default: plain four-thread CPU with FP16 KV.**
It avoids ~2 GB additional peak RSS and extra draft startup time for a marginal
decode gain. Later plain/DFlash cold loads took ~24 / 37 seconds; the first
plain load benefited from cached source pages and took 9.7 seconds. Keeping a
server running would amortize startup, but these tests do not install a service.

The per-part CPU profiler in the early DFlash logs accidentally counted draft
matmuls in the target-only breakdown, producing negative "other" time. This is
fixed in the final code and `final/draft-k2.log`; overall decode timings and
engine draft/target timing counters are unaffected.

On the Pi:

```sh
~/run-glimmer "Explain how a rainbow forms in a few sentences."
# Optional generation limit:
~/run-glimmer "What is the capital of France?" --n 16
```

The helper is `scripts/pi-glimmer.sh`, deployed as `~/run-glimmer`. It uses the
tested 250 MB target KV pool for short one-agent requests; this is not an 8k
context or multi-agent capacity configuration. `summary.json` preserves the
measurements and `bench/pi_glimmer_q2.py` reproduces the tuning trials.

## llama.cpp comparison

`llama_compare.py` starts a localhost-only llama.cpp server, sends one cold
request, and stops it. Same Q2_K file, four threads, FP16 KV, 80 formatted prompt
tokens, 32 output tokens, greedy sampling, zero cached tokens. It auto-adds BOS;
the request omits the explicit BOS to match Fleet's 80-token input. The generated
text matches Fleet's exactly on this sample. Native activation quantization
schemes differ between implementations; output equality is not universal.

| Metric | Fleet plain, final repeat | llama.cpp |
| --- | ---: | ---: |
| Prompt processing / first token | 30.85 s | 58.02 s |
| Generation | 0.93 tok/s | 1.05 tok/s |
| Request time excluding model load | 64.16 s | 87.43 s |

Fleet's request is ~1.36x faster in this one trial because of prompt processing;
llama.cpp is ~13% faster during decode. This is not a broad benchmark claim.
llama.cpp's CPU-only build identifies as `b1-785322d`; all commands and response
usage/timing data are in `llama.cpp/result.json` and `llama.cpp/server.log`.

## Final deployment check

`~/run-glimmer` answered the 60-token Paris prompt with
`Paris is the capital of France.<|eot|>`. Prompt processing took 23.42 s,
7 decode intervals took 7.53 s (0.93 tok/s), and the process exited 0.
Final launcher peak RSS was 11.94 GB. This is an additional prompt, not an
accuracy benchmark. `q23-helper.*` preserves the command and measurements.

After verifying the replacement and launcher, removed only the superseded Pi
file `~/models/Muse-Glimmer-30B-UD-Q2_K_XL.gguf`, freeing 12,444,212,256 bytes.
The Llama baseline and Glimmer drafter remain. All deployed source/script hashes
match `source-sha256.json`; `deployment.json` records the verification. Final Pi
checks report 46.6 C and `get_throttled=0x0`. Test servers/processes have exited.

Final native Pi kernel checks and 24 sanitizer-instrumented CPU draft engine
checks pass (`q23-final-check.log`, `q23-final-draftcheck.log`). The sanitizer
fixture driver includes the CPU implementation; linked library code is built
normally. `kernel-asan.log` separately instruments the new quantized kernels.

## Review of earlier Claude optimization claims

The older CPU-optimization chat (2026-10-29 02:42 UTC) described “10–30×”
**over the reference CPU path**, and explicitly expected the optimized engine
to reach “the same league as llama.cpp’s ARM path per token.” Its separate
5–10× multi-agent estimate was versus naive sequential serving/recomputation.
Those are not measurements of a 10× gain over modern llama.cpp. Repacking,
NEON int8 math, activation quantization and four-core use are overlapping parts
of the foundation, not independent multipliers. Most of that foundation is now
implemented. Earlier Mac GPU prefill/attention results are separate hardware.

The relevant unimplemented ideas are T-MAC lookup-table matmul, TEAL-style
activation sparsity, and early-exit/self-speculative inference with suitable
model support. [T-MAC](https://github.com/microsoft/T-MAC) reports Pi 5 results
for other models, but its [published baseline](https://github.com/microsoft/T-MAC/blob/main/docs/profiling_data.md)
is llama.cpp b2794 from May 2024, so those ratios cannot be assumed against our
current engine or modern llama.cpp. It also needs proper adaptation to the
current Q2_K/Q3_K group scales and tile layout. This is the strongest remaining
kernel experiment, not an already-working Glimmer quantization choice.

[TEAL](https://arxiv.org/abs/2408.14690) reports up to 1.53–1.8× for evaluated
model families using approximate activation sparsity and specialized kernels;
quantized Glimmer accuracy/speed are untested. [LayerSkip](https://github.com/facebookresearch/LayerSkip)
requires appropriately trained early-exit checkpoints for its demonstrated
speedups; generic layer skipping is not a proven Glimmer optimization.
Prefix/session reuse, n-gram drafts, batching and int8 KV already exist, but
their benefits depend on copying, repeated context, concurrency or long-context
attention. [Modern llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
also has prompt caching and continuous batching; comparison to a naive baseline
does not establish superiority over its server.

### Extra DFlash check: full trained block

Tested `FLEET_FULL_BLOCK=1`, keeping the drafter's trained 16-position block
while verifying three proposals. Same 80-token input and 32-token output;
generated text identical. Acceptance remained 14/46 (30.4%), while draft time
rose from 4.37 to 10.82 s and decode fell from 0.82 to **0.70 tok/s**. Logs and
explicit environment are in `full-block/`. This did not reveal a hidden fast
setting. No source/model/default configuration changed during this audit.

`optimization-audit.json` records the chat timestamps, implementation status,
primary sources and next experiment. Startup and prefill advantages do not
change the current conclusion: **single-agent steady generation remains below
llama.cpp in these short trials**, including the best DFlash setting tested.
The precise remaining plain-kernel gap has not been isolated by operation;
it should not be attributed solely to unpacking or a measured bandwidth ceiling.
