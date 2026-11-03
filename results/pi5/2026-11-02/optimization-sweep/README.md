# Pi 5: one-agent optimization sweep

The useful default-precision result is **2.25 tokens/sec with Qwen3-8B Q4_K_M**.
Keeping Glimmer 30B Q2_K gives about **0.96 tokens/sec on fresh prose**; verified
prompt lookup reaches **2.31 tokens/sec on the file-edit task**. Qwen reaches
**7.27 tokens/sec on that edit**. The editing gains depend on copying text already
in the prompt, and do not generalize to arbitrary new responses.

No layer skipping, FFN sparsity or coarser activation precision is enabled in the
demo presets. Changing to Qwen is a model choice, not an engine-only speedup.

## Measurement conditions

- Raspberry Pi 5, four Cortex-A76 cores, 16 GB RAM, NVMe, CPU backend, four threads.
  Observed load frequency 2.4 GHz; governor ondemand, no overclock changes.
- Greedy sampling, fresh sequence per task, weights loaded once per configuration.
  FP16 target KV 250 MB, row budget 64, memory budget 14.8 GB, source pinning off.
- Rate is `(generated_tokens - 1) / time_after_first_token`; excludes loading
  and prefill. Prompt processing and memory measurements remain in raw files.
- This is a short exploratory sweep, not a statistical performance study. Most
  configurations have one sample. Qwen's repeated rainbow and complete
  51-token physics response both measured about 2.25 tokens/sec.
- Sweep metadata records zero system swap growth. No throttling was observed.
  Large-context and multi-agent memory/quality were not validated here.

Model identities: Glimmer is Bartowski `Muse-Glimmer-30B-GGUF`, standard
`Muse-Glimmer-30B-Q2_K.gguf`, 11,035,579,936 bytes, HF revision
`01e1b7c30e19e1823ee9a73dec51d12f59ec4857`, SHA256
`af4b15f0f3268d23e0f48f796a4ab563454b9cf8bab60d2bbe251b314b342fb5`.
The official Qwen file is pinned and hash-verified in [qwen-model.json](qwen-model.json).
Both model files and the Glimmer drafter remain on the Pi. The superseded
Unsloth UD-Q2_K_XL file was removed after verifying the replacement.

## Results

| Model / setting | Workload | Decode tok/s | Quality relationship |
| --- | --- | ---: | --- |
| Glimmer, default group 32 | Rainbow prose | 0.961 | Unpruned baseline |
| Glimmer, prompt lookup 7 | File edit | 2.202 | Identical tokens to baseline |
| Glimmer, prompt lookup 15 | File edit | **2.308** | Identical tokens to baseline |
| Glimmer, group 256 | Rainbow prose | 0.973 | Different activation rounding |
| Glimmer, group 256 + lookup 7 | File edit | 2.386 | Identical to group-256 target |
| Glimmer, DFlash 2 | Rainbow prose | 1.062 | Default-precision target verification |
| Glimmer, group 256 + DFlash 2 | Rainbow prose | 1.253 | Identical to group-256 target |
| Qwen3-8B, default group 32 | Rainbow prose | **2.251** | Different, smaller model |
| Qwen3-8B, prompt lookup 7 | Rainbow prose | 2.246 | Identical tokens to Qwen baseline |
| Qwen3-8B, prompt lookup 7 | File edit | **7.270** | Identical tokens to Qwen baseline |

Lookup-7 accepted 54/55 Glimmer edit proposals; lookup-15 accepted 57/58.
Qwen accepted 49/58 edit proposals. On the novel rainbow response, lookup found
no useful matches. Group-256 DFlash accepted 15/29 proposals, spending 3.92 s
drafting and 20.82 s verifying/target work during the decode interval.

All seven Qwen lookup outputs exactly matched ordinary Qwen greedy token IDs.
Glimmer lookup-7/15 matched its ordinary target; group-256 lookup/DFlash matched
their group-256 target. These are checks on tested prompts, not a claim of
full-precision equivalence or an evaluation of stochastic sampling.

Qwen peak RSS was about 5.3–5.4 GB; Glimmer ordinary decoding about 11.94 GB.
This provides more headroom with Qwen, but remaining RAM is not all usable KV:
the OS, temporary buffers and any drafter also need space. The current preset
allocates only 250 MB target KV (roughly 1,680 Qwen token slots before allocator
details). Do not infer large agent-fleet capacity from these short requests.

## Quality and rejected settings

Six objective smoke tasks cover arithmetic, a fact, strict JSON, a small Python
function, an exact instruction and a file edit. Qwen passed all six, with and
without lookup. Glimmer passed five; it added narration/fences to JSON-only
requests both at group 32 and 256. This is not a comprehensive model ranking.
Generated code was compared structurally, never executed.

The 32-token rainbow probe is intentionally truncated and cannot establish
complete-answer quality. A separate Qwen response in [qwen-final](qwen-final/)
completed a correct two-sentence refraction/reflection/dispersion explanation.
See [quality-checks.json](quality-checks.json) and the saved response text.

Group-256 calibration perplexity was 11.2235 versus 11.1266 at group 32:
**+0.87% over only 158 scored tokens**. This tiny self-written corpus is a
diagnostic, not a universal quality-loss bound. Coarser activations remain opt-in.

| Glimmer approximation screen | tok/s | Decision |
| --- | ---: | --- |
| FFN down-input block sparsity 25% | 0.961 | Changed outputs, no useful speedup |
| FFN down-input block sparsity 50% | 0.966 | Changed outputs, no useful speedup |
| Skip layers 8,20,32,44 | 1.034 | Small gain; insufficient quality evidence |
| Skip eight layers | 1.123 | False physics explanation; rejected |
| Skip 25 of 52 layers | 1.699 | Unusable text; diagnostic only |

The sparsity prototype only skips zeroed 32-feature blocks at the FFN down
projection; it is not full TEAL. No pruning setting is selected for the demo.

## Comparison with llama.cpp

Both servers received the **same input token IDs and same GGUF**, four threads,
one request, no GPU, greedy, 32 generated tokens. llama.cpp used context 512
and no prompt cache. Its row arithmetic differs from Fleet's, and output text
was not identical. These are short probes, not broad superiority claims.

| Probe | Fleet tok/s | llama.cpp tok/s |
| --- | ---: | ---: |
| Qwen ordinary decode | 2.250 | **2.570** |
| Glimmer group-256 + DFlash 2 | **1.253** | 1.040 |

Ordinary default-precision Glimmer remains about 0.96, below llama.cpp's 1.04.
The optional Glimmer result is about 20% above this llama.cpp probe, but includes
an explicit precision change. The larger editing gains above are comparisons
against Fleet's own target without speculation; no matching llama.cpp editing
speculation run was performed. Startup is excluded from all decode rates.

## Implementation and validation

The shared CPU path now precomputes activation half-block sums used by Q2_K
bias corrections, avoiding repeated work per weight row. Interleaved kernel
tests measured roughly 6% for one row and 17–23% for verification batches;
whole-model ordinary Glimmer improved only modestly, from the previous ~0.93
to ~0.96. Multi-row ggml microbenchmarks repeat its row-dot function; they do
not represent all optimized llama.cpp batched GEMM kernels.

An optional 256-value activation scale uses integer superblock accumulation.
Default scales stay at 32. Speculation and approximation controls are shared
runtime options; Pi model paths and hardware choices live in launchers.
`--spec-k` now also controls prompt lookup. Chat serving honors explicit
`reasoning_effort: "none"` and the server's `--reasoning none` default, enabling
the same direct-answer template used by CLI evaluation.

Passed native/scalar tests for all seven CPU tile formats, batch sizes 1–17,
row partitions, group 32/256, dense/sparse inputs, exact Q2/Q3 repacking, and
quantization rounding cases. CPU DFlash matched an independent dense reference;
24 acceptance/rejection/KV cases passed. Local and Pi sanitizer checks passed.
The deployed CLI and chat API passed a JSON-only request; the repeated API
request additionally checks prefix reuse. Logs are under [validation](validation/)
and [api-check](api-check/). No test server or automatic service is left running.

## Run and reproduce

On the Pi:

```sh
~/run-demo qwen "Explain why rainbows have different colors, in two sentences."
~/run-demo glimmer "Your editing instruction followed by the current file"
~/serve-demo qwen
```

The first two commands exit after one response. The server keeps weights loaded
on `127.0.0.1:8090`, with one active request; stop it with Ctrl-C. From the Mac,
forward that port with `ssh -L 8090:127.0.0.1:8090 mli5@100.85.175.6` to use
`http://127.0.0.1:8090/v1/chat/completions`. No authentication/external exposure
is configured. Optional Fleet flags follow the prompt (CLI) or model (server).

```sh
cd ~/fleetllm
python3 bench/pi_sweep.py --profiles exact ngram7 --model ~/models/Qwen3-8B-Q4_K_M.gguf --output ~/my-qwen-run
python3 bench/check_pi_quality.py ~/my-qwen-run
python3 bench/pi_sweep.py --profiles exact group256 --ppl --output ~/my-glimmer-ppl
```

Each run directory saves exact commands, timings, RSS, swap growth, token IDs
and answers. [summary.json](summary.json) aggregates rates and token identity;
[source-manifest.json](source-manifest.json) identifies the deployed source.
[kernels](kernels/) contains microbenchmarks and the two reference snapshots:
`interleaved` compares before/after half-sums; `super8-vs4` uses the historical
label `fleet-before-sums` for the group-256 four-output-row reference, not the
original half-sum baseline. Treat those labels with this context.

The next fleet experiment should hold model, target precision and tasks fixed,
then compare sequential requests, shared-prefix batching and verified lookup at
increasing concurrency. Single-agent model replacement gains must remain
separate from agent-coordination gains.
