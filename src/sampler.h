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
} sample_params;

sample_params sample_defaults(void);
int32_t sample(const float *logits, int vocab, const sample_params *p, uint64_t *rng);
int32_t argmax(const float *x, int n);
