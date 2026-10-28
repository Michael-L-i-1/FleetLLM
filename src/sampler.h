// Temperature / top-k / top-p sampling with a per-sequence seeded RNG.
#pragma once
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    float temperature; // <= 0: greedy
    int top_k;
    float top_p;
    uint64_t seed;
    int max_new;
    bool ignore_eos; // keep generating past stop tokens (fixed-length benchmarks)
    int32_t stop[8]; // n_stop > 0: these end generation instead of the tokenizer's stop set
    int n_stop;
} sample_params;

sample_params sample_defaults(void);
int32_t sample(const float *logits, int vocab, const sample_params *p, uint64_t *rng);
int32_t argmax(const float *x, int n);
// Speculative verification of a deterministic draft token: keeps it with the
// target's probability, else samples the target distribution without it (the
// standard rejection rule for a point-mass draft), so the output distribution
// is exactly that of plain sampling; greedy: keep iff it is the argmax.
int32_t sample_verify(const float *logits, int vocab, const sample_params *p, uint64_t *rng, int32_t draft, bool *accepted);
