# FleetLLM CPU DFlash: first Glimmer test on Pi 5

Implemented CPU DFlash and tested on the 16 GB Raspberry Pi 5, four CPU threads.
This is a short smoke test, not a representative performance benchmark.

Target: `unsloth/Muse-Glimmer-30B-GGUF`, revision
`faa5b025c584459c13febfa5c59883516710ae39`,
`Muse-Glimmer-30B-UD-Q2_K_XL.gguf` (12,444,212,256 bytes).
Draft: `meta-models/Muse-Glimmer-30B-assistant`, revision
`c676cce91658f9568aaf264a416211e2782735b7`; its 5,111,976,608-byte BF16
checkpoint is quantized at load to affine 4-bit/group-64 (1.44 GB), then
repacked into 1.60 GB of CPU tiles. No llama.cpp inference was used.

Prompt: `What is the capital of France? Answer in one sentence.`
Both runs generated exactly the same answer text:

```text
The capital of France is Paris.<|eot|>
```

| Metric | Plain | DFlash, up to 3 drafts |
| --- | ---: | ---: |
| Prompt processing, 60 tokens | 131.87 s | 137.33 s |
| Generation interval, 7 tokens after first token | 64.33 s | 38.54 s |
| Generation rate | 0.109 tok/s | 0.182 tok/s |
| Whole process, including load/startup | 197.79 s | 185.28 s |
| Observed peak resident set | 12.94 GB | 14.40 GB |

Generation was 1.67x faster, but total process time improved only about 6%.
The prompt-processing bottleneck and draft load cost dominate this tiny answer.
The shared engine reports 5/9 proposed draft tokens accepted (55.6%), 2.67
tokens per verification, and 1.06 s spent in the drafter. Acceptance counters
can include verified proposals beyond the final stop token.

Target KV was FP16, capped at 0.25 GB; draft KV 0.064 GB, `--spec-k 3`,
`--spec-rows 4`, `--row-budget 8`, greedy sampling, maximum 16 generated tokens,
and no explicit weight pinning. `run.json` contains exact commands and memory
samples. RSS includes file-backed mappings; `MemAvailable` includes reclaimable
file-cache pages and should not be interpreted as unused RAM beside the model.
Successful DFlash run's peak increase in **system-wide** swap use was 376,832
bytes, less than 1 MB; it did not develop substantial swapping. This counter is
not per-process. Final `vcgencmd get_throttled` was `0x0`.

## Implementation and validation

- CPU target-layer hidden capture, context projection, per-layer draft KV,
  bidirectional anchor/mask attention, sliding windows, and shared target head.
- Quantized CPU draft matmuls reuse the existing NEON dot-product tiles.
- Q2/Q3/Q5 target matrices retain their packed representation instead of being
  implicitly expanded to CQ8. They currently use row dequantization and NEON
  float dot products; native packed kernels remain the main optimization need.
- On Linux, discard the no-longer-used temporary quantization arena immediately
  after repack and release the drafter checkpoint's redundant resident pages.
  A preliminary run was stopped by the memory monitor after swap growth;
  `run-before-memory-fix.json` and `dflash-memory-stopped.log` preserve that result.
- `scripts/cpudraftcheck.py`: six dense PyTorch hidden-state/token comparisons
  covering FP32/FP16/int8 KV, mixed versus context-only draft updates, multiple
  sequences, page boundaries, and sliding windows. Also checks target-layer
  capture layout and prevents compact-weight expansion.
- 24 engine checks on Mac and Pi: reference/native CPU, three KV formats, draft
  lengths 1/3, forced full acceptance and rejection; all preserve greedy tokens.
  New CPU code is compiled with address/undefined-behavior sanitizers for these
  fixture checks. The real checkpoint comparison verifies identical text.
- Existing CPU quantization checks pass; Pi Llama regression passes 8/8 tasks.

The successful plain run preceded only the Linux draft-memory cleanup; the
plain path does not use that temporary draft arena. Both measurements use the
same target kernels. Model weights remain on the Pi; no system upgrades or
network configuration changes were performed. Test processes exited normally.

To repeat, copy `smoke.py` to the Pi and run `python3 smoke.py`. It monitors
system memory/swap and terminates a run on substantial pressure or timeout.
To repeat code checks locally: `python3 scripts/cpudraftcheck.py` (NumPy/PyTorch
are development-only dependencies) and `sh scripts/cpuqcheck.sh`.

## Follow-up CPU profile and cache capacity

`cpu-profile.log` records a plain-target run with the same 60-token prompt,
four threads, `FLEET_PROFILE=1`, `--n 2`, and `--row-budget 128` instead of 8.
Prefill took 139.93 s and the one measured decode interval 9.18 s. Matrix
multiplication accounted for 74,207.65 of 74,557.95 ms per forward (99.5%,
averaged across prompt and decode forwards). Increasing the row budget alone
did not resolve the bottleneck. This is a diagnostic run, not a repeated benchmark.

Physical RAM is 17,006,116,864 bytes. Subtracting the observed DFlash peak RSS
and roughly 343 MB idle OS use leaves about 2.10 GiB, before additional scratch
and operating-system reserve. Plan conservatively for roughly 1–2 GB additional
cache, then validate resident memory under the intended concurrency/context load.
The smoke-test limits (250 MB target, 64 MB draft) are configuration limits,
not a measurement of all remaining memory.

Ideal FP16 KV storage per independent sequence, excluding page rounding:

| Context tokens | Target | Draft | Combined |
| --- | ---: | ---: | ---: |
| 2,048 | 104 MiB | 40 MiB | 144 MiB |
| 4,096 | 130 MiB | 80 MiB | 210 MiB |
| 8,192 | 182 MiB | 160 MiB | 342 MiB |
| 16,384 | 286 MiB | 320 MiB | 606 MiB |

Target storage is 13 KiB/token plus 39 KiB/token capped at the 2,048-token
sliding window. The current draft cache retains its full history at 20 KiB/token.
Int8 KV uses approximately half this storage, including scale overhead; Glimmer
quality still needs evaluation. Weights are shared across agents. These are
storage estimates, not validated agent-capacity claims: the current target
allocator reserves up to 75% for sliding-window pages, which can waste capacity
for long contexts and needs tuning before an 8k multi-agent test.
