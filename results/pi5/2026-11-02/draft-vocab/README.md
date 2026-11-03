# Draft-only vocabulary shortlist experiment

Goal: reduce learned DFlash proposal cost while keeping full target verification
and the selected target's greedy answers unchanged. This is not target-vocabulary
pruning, activation sparsity, or a smaller target model.

The preceding target forward already computes complete output logits. Rank
four-token weight tiles by their largest logit distance from the maximum across
all target output rows; keep a configurable number of tiles. DFlash projects
only these vocabulary rows and proposes the best candidate. Missing a useful
proposal can reduce acceptance, but full-vocabulary target verification still
corrects the answer. The target uses all layers, default group-32 activation
precision, FP16 KV and its complete vocabulary.

The prototype's experimental environment variable is `FLEET_DRAFT_VOCAB=N`,
where N is a row budget rounded up to four. Unset means the ordinary full draft
head. This is an experiment, not a new recommended default.

## Outcome: rejected for the current Pi demo

The shortlist reduced drafting time but lowered acceptance enough to increase
full-target work. Both tested shortlist budgets were **6–7% slower overall**.
No production option or source change is retained; the prototype is archived
as a reproducible patch. No broader quality/performance claim is made from this
single 32-token prose probe.

| Draft vocabulary rows | Decode tok/s | Draft seconds | Target seconds | Accepted/proposed |
| --- | ---: | ---: | ---: | ---: |
| Full vocabulary, first control | 1.109451 | 3.937830 | 24.003818 | 15/30 |
| 32,768 shortlist | 1.030248 | 3.075041 | 27.014679 | 13/34 |
| 8,192 shortlist | 1.043300 | 2.831406 | 26.881886 | 13/33 |
| Full vocabulary, repeated control | 1.110521 | 3.904960 | 24.009754 | 15/30 |

All four runs produced exactly the same output token IDs as the saved ordinary
Glimmer baseline. All recorded system swap-growth values were zero. Matching
controls differed by about 0.1%, so the measured shortlist slowdown is larger
than the observed control variation. No claim is made about other tasks or
vocabulary-selection algorithms.

## Correctness checks

- Sparse head projection matches the selected rows of the full head exactly,
  both native CPU and scalar reference; includes noncontiguous/padded vocab rows.
- Partial top-tile selection matches complete sorting at 74 budget cutoffs.
- Native and reference end-to-end speculation passed 24 acceptance/rejection
  cases at KV precision 1/2/4 bytes and depths 1/3, with identical target tokens.
- Address/undefined-behavior sanitizer checks passed. See `validation/`.

## Measurement

The isolated driver `bench/pi_draft_vocab.py` (included in the archived prototype
patch) locks `~/.cache/fleetllm-bench.lock`, saves commands and output token IDs, measures
RSS/swap growth, and checks greedy token equality against a supplied ordinary
target baseline (or the first unrestricted draft run if no reference is supplied). No model files, production launchers or installed services are changed.

## Reproduce the archived prototype

No draft-shortlist production code is selected for the main branch. The patch
includes the CPU implementation, isolated driver and a standalone unit check.
Apply it to a disposable worktree based on `dfd72cb`, then run on the provisioned
Pi:

```sh
git apply --unidiff-zero /path/to/prototype.patch
python3 bench/pi_draft_vocab.py --build --binary build/fleet --output results/my-draft-vocab --sizes 0 32768 8192 0 --reference /path/to/ordinary-target.jsonl
cc -O1 -g -fsanitize=address,undefined -Isrc bench/draft_shortlist_check.c build/libfleetllm.a -lpthread -lm -o build/draft-shortlist-check
ASAN_OPTIONS=detect_leaks=0 build/draft-shortlist-check
```

The driver takes an exclusive benchmark lock across the build and all cases.
All cases retain default target activation precision, and load weights separately.
The decode rate excludes model loading and prompt processing.

This experiment used four CPU threads, default group-32 target activation
scaling, FP16 target KV 250 MB, draft KV 64 MB, proposal depth two and three
verification rows. The exact commands, binary/source/task hashes, timings and
raw token IDs are under [screen](screen/); [summary.json](summary.json) aggregates
results. Loading took about 35–37 seconds per case and is excluded from decode
rates. The input is a truncated speed probe, not a complete-answer capability
evaluation. The correctness invariant preserves the existing target's answers;
it does not establish their scientific accuracy.
