#include "spans.h"
#include <stdlib.h>

static uint32_t block_key(const int32_t *t) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < SPAN_BLOCK; i++) { h ^= (uint32_t)t[i]; h *= 16777619u; h ^= h >> 13; }
    return h ? h : 1;
}

void span_init(span_index *x, int log2_slots) {
    x->mask = (1u << log2_slots) - 1;
    x->t = calloc((size_t)x->mask + 1, sizeof(span_slot));
}

void span_free(span_index *x) {
    free(x->t);
    x->t = NULL;
}

int span_add(span_index *x, int seq, int gen, const int32_t *tok, int from, int n) {
    int b = (from + SPAN_BLOCK - 1) / SPAN_BLOCK * SPAN_BLOCK;
    for (; b + SPAN_BLOCK <= n; b += SPAN_BLOCK) {
        uint32_t k = block_key(tok + b);
        x->t[k & x->mask] = (span_slot){ k, seq, b, gen };
    }
    return b;
}

bool span_find(const span_index *x, int self, const int32_t *pend, int n, int min_len, ng_view (*view)(void *u, int seq), void *u, span_run *out) {
    span_run best = { 0, 0, -1, 0 };
    for (int o = 0; o + SPAN_BLOCK <= n;) {
        uint32_t k = block_key(pend + o);
        span_slot s = x->t[k & x->mask];
        int adv = 1;
        if (s.key == k && s.seq != self) {
            ng_view v = view(u, s.seq);
            int i = 0;
            if (v.tok && v.gen == s.gen && s.pos + SPAN_BLOCK <= v.n)
                while (i < SPAN_BLOCK && v.tok[s.pos + i] == pend[o + i]) i++;
            if (i == SPAN_BLOCK) { // verified block: extend both ways
                int a = 0, z = SPAN_BLOCK;
                while (o - a > 0 && s.pos - a > 0 && pend[o - a - 1] == v.tok[s.pos - a - 1]) a++;
                while (o + z < n && s.pos + z < v.n && pend[o + z] == v.tok[s.pos + z]) z++;
                if (a + z > best.len) best = (span_run){ o - a, a + z, s.seq, s.pos - a };
                adv = z;
            }
        }
        o += adv;
    }
    if (best.len < min_len) return false;
    *out = best;
    return true;
}
