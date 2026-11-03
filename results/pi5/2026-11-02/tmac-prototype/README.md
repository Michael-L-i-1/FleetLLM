# T-MAC-inspired Q2_K prototype: rejected

**Decision: do not merge this kernel.** Exact sixteen-row lookup generated
0.802 tokens/sec versus 0.960 for the existing kernel, **16.5% lower throughput** (19.7% longer decode time),
with identical output tokens. The rounded variant's tiny isolated micro gain did
not justify extra approximation, preparation overhead or a quality sweep.

This is an independent subset-sum lookup-table prototype, not an integration of
Microsoft's generated T-MAC runtime. It retains the actual Glimmer Q2_K weight
codes and per-16-value scale/min metadata. Other streamed formats remain on the
existing kernels. No system configuration or primary deployment is changed.

## Scope and source

Reviewed upstream source: [microsoft/T-MAC](https://github.com/microsoft/T-MAC),
commit `7042f8f73330bd083bc1e4bc5ccb3f88a4904aee`. Its published end-to-end table
uses different models and an older llama.cpp baseline. Its GPTQ/group-scale
integration is not a drop-in implementation for Q2_K's metadata.

For each group of four quantized activations, precompute the 16 possible subset
sums. Repack each weight's two bits into two four-bit indices. The inner loop
looks up each bitplane's sum and combines the planes with weights one and two.
Q2_K's separate half-block scales/mins are applied in the same float order as
the original target. Weight code storage is unchanged except padding in tiny
non-multiple-of-16 test shapes.

Four variants are compared:

- `16`: exact int16 LUT entries, four output rows per block, NEON TBL4.
- `8`: rounded int8 LUT entries, four output rows, NEON TBL4.
- `116`: exact int16 LUT entries, sixteen output rows, NEON TBL1.
- `108`: rounded int8 LUT entries, sixteen output rows, NEON TBL1.

Exact LUTs retain the selected target arithmetic. Rounded LUTs add approximation
on top of the existing int8 activation quantization; their max-error ratio on a
synthetic activation vector is not a perplexity or benchmark-accuracy measure.
The rounded variants are not accepted as quality-preserving settings.

## Validation and initial negative result

Exact int16 lookup passed address/undefined-behavior sanitizer checks over padded
seven-row matrices, partitions, batches 1–17, zero and scaled activations for
both output layouts, with bitwise-identical results versus the original kernel.
See `exact-sanitizer.log`.

The initial microbench automatically selected `token_embd.weight`, the largest
Q2_K matrix. **This is a synthetic dense operation on real embedding weights,
not a streamed decoder hot path.** Its initial four-row timings are preserved
in `micro-embedding.csv` for transparency; they cannot stand in for model speed.
The revised selector excludes token embeddings and the output head by default.

Initial embedding dense multiplication, 202,048 x 6,656, four threads, batch one:

| Kernel | Median time | Relative speed |
| --- | ---: | ---: |
| Existing NEON dot | 41.44 ms | 1.00x |
| Four-row exact int16 LUT | 64.48 ms | 0.64x |
| Four-row rounded int8 LUT | 45.85 ms | 0.90x |

Table preparation was outside the micro timing and took 0.245 ms for one input
row (both exact and rounded representations constructed together). The exact
variant matched every output bitwise. The rounded variant's max absolute error
was 0.281, or 1.51% of the maximum absolute reference output on that vector.

## Streamed decoder matrices

The revised test selects actual streamed matrices and compares five modes in
three rotating-order rounds, ten repetitions per mode for FFN and twenty for
attention gate. Activation table preparation is excluded from these micro rates.
Every output row is checked; the exact modes matched bitwise for batches 1/2/4.

| Batch-one kernel | FFN gate 19,968 x 6,656 | Attention gate 4,096 x 6,656 |
| --- | ---: | ---: |
| Native dot | **4.127 ms** | **0.799 ms** |
| Four-row exact int16 LUT | 6.749 ms (0.61x speed) | 1.427 ms (0.56x) |
| Sixteen-row exact int16 LUT | 5.543 ms (0.74x) | 1.148 ms (0.70x) |
| Four-row rounded int8 LUT | 4.578 ms (0.90x) | 0.932 ms (0.86x) |
| Sixteen-row rounded int8 LUT | 4.023 ms (1.026x) | 0.798 ms (1.000x) |

The best rounded variant's isolated FFN gain is only 2.6%, with no attention-gate
gain and extra LUT preparation work. All lookup variants lose at batch two/four;
this prototype repeats the single-input loop rather than reusing weight blocks
across activation rows. The rounded variant changes outputs (FFN max-error ratio
about 1.6% on the synthetic input). No perplexity or accuracy claim is made;
these kernel results do not justify a quality tradeoff or an expensive quality
sweep for the rounded variant.

The sixteen-row exact layout is used for the full-model comparison. The code
format is replaced in-place, avoiding a second persistent Q2 weight copy. It
cannot use the old native prefill kernel on those new codes; the benchmark
records prefill separately, and the decision is based on single-agent decode.
Native sanitizer validation instruments the new LUT module and fixture while
linking previously tested native kernels. The local sanitizer run instruments
both the LUT module and the inherited cpuq.c. See the corresponding logs.

## Full-model result

Same Glimmer file, four threads, greedy, fresh sequence, 61 prompt tokens,
24 generated tokens, row budget 64, FP16 KV 250 MB, no pinning or pruning. Model
loading and prefill are excluded from decode rates. Exact LUT preserves all
24 output token IDs and all input token IDs from the baseline.

| Metric | Native baseline | Sixteen-row exact LUT |
| --- | ---: | ---: |
| Decode tokens/sec | **0.959526** | 0.801545 |
| Decode seconds after first token | **23.970** | 28.695 |
| Prefill seconds | **19.519** | 48.112 |
| Model load seconds | **23.9** | 54.7 |

Both runs completed without swap growth or memory guard termination. Raw output,
commands, memory observations and logs are under [raw](raw/). Source/binary
hashes match the Pi in [deployment.json](deployment.json). Machine-readable
aggregation is [summary.json](summary.json). These are short negative probes,
not statistical estimates across workloads.

A larger LUT row tile improved the initial design, but the best exact kernel
still lost on both actual streamed decoder matrices and full-model decode.
This supports rejecting **this prototype**, not concluding that every tuned
upstream T-MAC integration or different low-bit checkpoint would lose. Upstream
code generation, packing and different quantization layouts remain separate
work. No production default, model file, system setting or demo launcher was
changed by the experiment.

## Reproduce

The complete experimental code is archived in [prototype.patch](prototype.patch)
against baseline `dfd72cb`. It is intentionally not installed into the production
kernel. To reconstruct an isolated checkout, first build the baseline, then apply
the patch and build the prototype:

```sh
git worktree add --detach /tmp/fleet-tmac-repro dfd72cb
cd /tmp/fleet-tmac-repro
make NO_METAL=1 BUILD=build-baseline -j2
git apply /absolute/path/to/tmac-prototype/prototype.patch
make NO_METAL=1 -j2
```

On the isolated Pi checkout, use the shared benchmark lock so no second model
or kernel test competes for CPU/memory bandwidth:

```sh
make NO_METAL=1 -j2
cc -O3 -mcpu=native -Isrc bench/experimental/q2_lut.c build/libfleetllm.a -lpthread -lm -o build/q2-lut
flock ~/.cache/fleetllm-bench.lock ./build/q2-lut ~/models/Muse-Glimmer-30B-Q2_K.gguf 10
```

The optional fourth argument filters tensor names; `--list` as second argument
lists Q2 tensor names/shapes without executing matmul. A default invocation
selects the largest Q2 matrix excluding embedding/output. Mode `0` is the
unchanged native kernel. Rounds rotate mode order, with all output rows checked
against the native kernel; exact modes fail the run on any bitwise mismatch.
Batch-2/4 LUT code currently repeats the single-input loop and is not a fully
optimized T-MAC batched implementation.

`python3 bench/experimental/q2_lut_model.py` runs the guarded full-model pair
using `build-baseline` and `build`, with an exclusive benchmark lock and
assertions on exact output token identity. Its default uses mode 116.

`FLEET_Q2_LUT=16` selects the four-row exact kernel in the isolated full-model
build; `116` selects sixteen-row exact. Unset leaves the baseline path. The
primary `~/fleetllm` checkout and demo launchers never use these flags.
