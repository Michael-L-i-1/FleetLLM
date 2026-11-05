# GPQA Diamond: Muse Glimmer 30B Q2_K, phased token budget (Raspberry Pi 5 + MacBook)

**Headline (one uniform run, all on the Raspberry Pi 5):** **115 / 198 = 58.1%** at a
2,048-token output cap (48 wrong, 35 hit the cap and count as wrong).

**Phased, extended-budget result:** **130 / 198 = 65.7%** (66 wrong, of which 2 still hit the
32,768-token cap). This is *not* one uniform run; see "Method" and "Reproducibility".

## Phases

| Phase | Run | Host | Token cap | At once | Questions in run | Answers used | Correct used |
|---|---|---|---|---|---|---|---|
| 1 | `results/pi5/2026-11-03/gpqa-diamond-full` | Raspberry Pi 5 16 GB (CPU, 4x Cortex-A76) | 2,048 | 16 | 198 | 163 | 115 |
| 2 | `results/pi5/2026-11-03/gpqa-diamond-8k` | Raspberry Pi 5 16 GB (CPU) | 8,192 | 8 | 37 (35 capped + 2 controls) | 7 | 5 |
| 3 | `results/mac/2026-11-04/gpqa-diamond-32k` | MacBook Pro M5 Pro 48 GB (Metal GPU) | 32,768 | 16 | 28 | 28 | 10 |

- **Answered on the Raspberry Pi (170 questions):** all 163 that finished under 2,048 tokens in
  phase 1, plus 7 of the capped questions finished at 8,192 tokens in phase 2.
- **Answered on the MacBook (28 questions):** 37, 64, 70, 80, 89, 91, 92, 100, 102, 103, 110,
  116, 121, 122, 127, 128, 129, 132, 140, 148, 150, 174, 180, 183, 184, 188, 195, 197. These had
  hit 2,048 tokens on the Pi and phase 2 was stopped before reaching them (27) or capped them at
  8,192 (1). They ran on the Mac only to finish in reasonable time; they say nothing about Pi speed.

Per-question rows, with the phase, host and cap that produced each answer: `questions.csv`.
Machine-readable summary: `summary.json`.

## Method (from `summary.json`)

> Phased run (token caps 2048 -> 8192 -> 32768; hosts: MacBook Pro M5 Pro 48 GB (Metal GPU), Raspberry Pi 5 16 GB (CPU, 4 x Cortex-A76)). Each question keeps its first final answer; questions that hit a cap or did not finish were run again in the next phase with a larger cap. Decoding did not reproduce token for token across phases, so this is an extended-budget estimate, not one uniform run; the uniform result is phase 1 (115/198 at 2048 tokens).

## Reproducibility

The plan was that a question finishing under a cap would produce identical greedy output under
a larger cap, making the phased result equal to one run at the largest cap. It did not hold on
the Pi: phase-2 reruns of Q12 and Q54 matched phase 1 for only ~400-470 characters (~100 tokens)
before diverging (both then finished correctly in under 1,000 tokens, where phase 1 had run past
2,048). The likely cause is that n-gram draft verification was active in phase 2 (8 running,
16-row budget) but effectively off in phase 1 (16 running), and the CPU kernels round slightly
differently when several rows are verified at once. Greedy decoding amplifies that into different
text. Questions restarted after the two power cycles in phase 1 are affected the same way. Phase 3
also used a different backend (Metal). So the phased score mixes three numerical paths; the clean
comparison number is the phase-1 result. This is worth fixing in the engine: batched and
speculative outputs on the CPU path should match sequential greedy output.

## Timing

- Phase 1: 2026-11-03 01:23 to 2026-11-04 06:03 (~28.7 h), Pi at ~2.7-2.8 tok/s aggregate with
  16 in flight. The Pi lost power twice (~20:32 and 00:58); the run resumed from its checkpoint
  each time, and the 16 questions in progress restarted from scratch ("attempt 2/3").
- Phase 2: 2026-11-04 06:08 to 10:30 (stopped early), ~1.5-2.8 tok/s with 8 in flight.
- Phase 3: 2026-11-04 10:39:48 to 11:16:55 (~37 min), ~70 tok/s aggregate on the M5 Pro GPU.

## Settings (all phases)

- Model: `bartowski/Muse-Glimmer-30B-GGUF` `Muse-Glimmer-30B-Q2_K.gguf`, revision `01e1b7c3`,
  sha256 `af4b15f0f3268d23e0f48f796a4ab563454b9cf8bab60d2bbe251b314b342fb5` (verified on both hosts).
- FleetLLM `fleet serve`: 8-bit KV (`--kv-int8`; 1.5 / 2 / 12 GB by phase), `--ngram --spec-k 15
  --spec-rows 16`, `--reasoning low`, `--admit`, `--max-active` = concurrency; Pi phases `--backend
  cpu --threads 4 --budget-gb 14.8 --row-budget 64 --no-pin`, Mac phase Metal `--budget-gb 32`.
- Harness: OpenAI simple-evals GPQA Diamond (pinned commit `652c89d0`), unchanged prompt, choice
  permutations and grading, via `evals/gpqa.py --stream --reasoning-effort low --temperature 0`
  with the resumable CSV checkpoint (`evals/checkpoint.py`). Questions frozen once in phase 1.
- Launchers: `evals/pi_gpqa.sh` (Pi), `evals/run_gpqa_local.sh` (Mac); phase selection and merge:
  `evals/gpqa_extend.py`.
