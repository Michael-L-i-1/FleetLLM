# Exact packed-weight prefetch experiment

This experiment changes memory-fetch hints only. It does not change quantization,
weights, activation scales, accumulation order, model layers or verification.
Production `src/cpuq.c` remains unchanged until a useful full-model gain is proven.

`build.py` generates eight compile-time variants of the existing dense single-row
kernel, plus its unmodified baseline. Variants prefetch packed codes and subscales
4, 8, 16 or 32 32-value blocks ahead, using streaming (`locality=0`) or cache-keep
(`locality=3`) hints. Fetches stay inside the current row group's column range.

`measure.c` selects the largest two-dimensional Q2_K/Q3_K/Q4_K/Q6_K matrix present
in the real GGUF, repacks it with the normal code and runs all variants against
the same quantized activations. All output floats must compare bitwise equal.
Activation batches one and four are tested, with four threads and row grain 64.
Variant order rotates each round. Activation preparation and repacking are outside
the timer. Four-row timings are a regression control: the prototype only enables
prefetch for a single input row.

On the Pi, in an isolated checkout:

```sh
flock ~/.cache/fleetllm-bench.lock python3 bench/experiments/pi-stream/build.py
flock ~/.cache/fleetllm-bench.lock build/stream/measure ~/models/Muse-Glimmer-30B-Q2_K.gguf 5 > stream.csv
python3 bench/experiments/pi-stream/analyze.py stream.csv --output stream-summary.json
```

No ggml, TVM, downloaded model, or external kernel library is required by this
experiment. This is conventional software prefetch, not T-MAC. Microbenchmarks
cannot establish a whole-model speedup; run matched full-model tests before
selecting an implementation.

Add a fourth argument `decoder` to exclude `token_embd.weight` from matrix
selection. Embeddings are row lookups during inference, so decoder-only results
are the relevant follow-up for this experiment.

A second screen reuses the unmodified kernel library to vary threadpool row-task
grain and test four contiguous partitions (`static4`). It compiles in seconds:

```sh
cc -O3 -mcpu=native -D_GNU_SOURCE -DSTREAM_GRAIN_ONLY -Isrc \
  bench/experiments/pi-stream/measure.c build/libfleetllm.a -lpthread -lm \
  -o build/stream/grain
flock ~/.cache/fleetllm-bench.lock build/stream/grain ~/models/Muse-Glimmer-30B-Q2_K.gguf 5 decoder > grain.csv
```

The baseline is the existing grain of 64 output rows; all tested grains and
partition starts are multiples of four, as required by packed row tiles.
