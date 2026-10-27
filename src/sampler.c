#include "sampler.h"
#include <math.h>
#include <stdlib.h>

sample_params sample_defaults(void) {
    return (sample_params){ .temperature = 0.7f, .top_k = 64, .top_p = 0.95f, .seed = 0, .max_new = 256, .ignore_eos = false };
}

int32_t argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

static float next_f32(uint64_t *s) { // xorshift64*
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    uint64_t r = *s * 0x2545F4914F6CDD1DULL;
    return (float)(r >> 40) / (float)(1u << 24);
}

int32_t sample(const float *logits, int vocab, const sample_params *p, uint64_t *rng) {
    if (p->temperature <= 0) return argmax(logits, vocab);
    int k = p->top_k > 0 && p->top_k < vocab ? p->top_k : (vocab < 256 ? vocab : 256);
    // top-k by insertion into a small sorted buffer (k is small)
    int32_t *idx = malloc(sizeof(int32_t) * k);
    float *val = malloc(sizeof(float) * k);
    int m = 0;
    for (int i = 0; i < vocab; i++) {
        float v = logits[i];
        if (m == k && v <= val[m - 1]) continue;
        int j = m < k ? m++ : k - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
        val[j] = v; idx[j] = i;
    }
    float mx = val[0], sum = 0;
    for (int i = 0; i < m; i++) { val[i] = expf((val[i] - mx) / p->temperature); sum += val[i]; }
    float cum = 0;
    int cut = m;
    for (int i = 0; i < m; i++) {
        cum += val[i] / sum;
        if (cum >= p->top_p) { cut = i + 1; break; }
    }
    float total = 0;
    for (int i = 0; i < cut; i++) total += val[i];
    float r = next_f32(rng) * total;
    int32_t out = idx[cut - 1];
    for (int i = 0; i < cut; i++) { r -= val[i]; if (r <= 0) { out = idx[i]; break; } }
    free(idx);
    free(val);
    return out;
}
