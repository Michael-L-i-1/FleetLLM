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

// Candidate set after temperature, top-k and top-p: ids and unnormalized
// probabilities (sorted, highest first); returns the count. idx/val hold k entries.
static int candidates(const float *logits, int vocab, const sample_params *p, int32_t *idx, float *val, int k) {
    int m = 0;
    for (int i = 0; i < vocab; i++) { // top-k by insertion into a small sorted buffer (k is small)
        float v = logits[i];
        if (m == k && v <= val[m - 1]) continue;
        int j = m < k ? m++ : k - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
        val[j] = v; idx[j] = i;
    }
    float mx = val[0], sum = 0;
    for (int i = 0; i < m; i++) { val[i] = expf((val[i] - mx) / p->temperature); sum += val[i]; }
    float cum = 0;
    for (int i = 0; i < m; i++) {
        cum += val[i] / sum;
        if (cum >= p->top_p) return i + 1;
    }
    return m;
}

static int cand_k(const sample_params *p, int vocab) { return p->top_k > 0 && p->top_k < vocab ? p->top_k : (vocab < 256 ? vocab : 256); }

int32_t sample(const float *logits, int vocab, const sample_params *p, uint64_t *rng) {
    if (p->temperature <= 0) return argmax(logits, vocab);
    int k = cand_k(p, vocab);
    int32_t *idx = malloc(sizeof(int32_t) * k);
    float *val = malloc(sizeof(float) * k);
    int cut = candidates(logits, vocab, p, idx, val, k);
    float total = 0;
    for (int i = 0; i < cut; i++) total += val[i];
    float r = next_f32(rng) * total;
    int32_t out = idx[cut - 1];
    for (int i = 0; i < cut; i++) { r -= val[i]; if (r <= 0) { out = idx[i]; break; } }
    free(idx);
    free(val);
    return out;
}

int32_t sample_verify(const float *logits, int vocab, const sample_params *p, uint64_t *rng, int32_t draft, bool *accepted) {
    if (p->temperature <= 0) {
        int32_t t = argmax(logits, vocab);
        *accepted = t == draft;
        return t;
    }
    int k = cand_k(p, vocab);
    int32_t *idx = malloc(sizeof(int32_t) * k);
    float *val = malloc(sizeof(float) * k);
    int cut = candidates(logits, vocab, p, idx, val, k);
    float total = 0, pd = 0;
    for (int i = 0; i < cut; i++) { total += val[i]; if (idx[i] == draft) pd = val[i]; }
    int32_t out = draft;
    *accepted = next_f32(rng) * total < pd; // keep the draft with its target probability
    if (!*accepted) { // otherwise sample the target distribution with the draft removed
        float r = next_f32(rng) * (total - pd);
        out = idx[cut - 1] != draft ? idx[cut - 1] : idx[0];
        for (int i = 0; i < cut; i++) {
            if (idx[i] == draft) continue;
            r -= val[i];
            if (r <= 0) { out = idx[i]; break; }
        }
    }
    free(idx);
    free(val);
    return out;
}
