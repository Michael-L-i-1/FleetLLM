#include "ngram.h"
#include <stdlib.h>
#include <string.h>

#define NG_MAX_MATCH 256

static uint32_t window_key(const int32_t *w) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < NG_N; i++) { h ^= (uint32_t)w[i]; h *= 16777619u; h ^= h >> 15; }
    return h ? h : 1;
}

void ng_seq_free(ng_seq *x) {
    free(x->pos);
    free(x->key);
    memset(x, 0, sizeof *x);
}

static void seq_put(ng_seq *x, uint32_t key, int32_t pos) {
    if ((x->used + 1) * 2 > x->mask + 1) { // grow and rehash
        uint32_t old = x->mask + 1, cap = x->mask ? old * 2 : 1024;
        int32_t *op = x->pos;
        uint32_t *ok = x->key;
        x->pos = malloc(sizeof(int32_t) * cap);
        x->key = malloc(sizeof(uint32_t) * cap);
        memset(x->pos, 0xff, sizeof(int32_t) * cap);
        x->mask = cap - 1;
        x->used = 0;
        for (uint32_t i = 0; op && i < old; i++) if (op[i] >= 0) seq_put(x, ok[i], op[i]);
        free(op);
        free(ok);
    }
    uint32_t i = key & x->mask;
    while (x->pos[i] >= 0 && x->key[i] != key) i = (i + 1) & x->mask;
    if (x->pos[i] < 0) x->used++;
    x->key[i] = key;
    x->pos[i] = pos; // latest occurrence wins
}

static int32_t seq_get(const ng_seq *x, uint32_t key) {
    if (!x->mask) return -1;
    uint32_t i = key & x->mask;
    while (x->pos[i] >= 0) {
        if (x->key[i] == key) return x->pos[i];
        i = (i + 1) & x->mask;
    }
    return -1;
}

void ng_global_init(ng_global *g, int log2_slots) {
    g->mask = (1u << log2_slots) - 1;
    g->t = calloc((size_t)g->mask + 1, sizeof(ng_slot));
}

void ng_global_free(ng_global *g) {
    free(g->t);
    g->t = NULL;
}

void ng_update(ng_seq *x, ng_global *g, int seq, int gen, const int32_t *ctx, int n) {
    // windows ending at positions [indexed, n-2]; the window ending at n-1 is the query
    int from = x->indexed > NG_N - 1 ? x->indexed : NG_N - 1;
    for (int e = from; e <= n - 2; e++) {
        uint32_t key = window_key(ctx + e - NG_N + 1);
        seq_put(x, key, e + 1);
        if (g && g->t) g->t[key & g->mask] = (ng_slot){ key, seq, e + 1, gen };
    }
    if (n - 1 > x->indexed) x->indexed = n - 1;
}

// tokens before `pos` in src that match the tail of ctx
static int match_len(const int32_t *src, int pos, const int32_t *ctx, int n) {
    int m = 0;
    while (m < NG_MAX_MATCH && pos - 1 - m >= 0 && n - 1 - m >= 0 && src[pos - 1 - m] == ctx[n - 1 - m]) m++;
    return m;
}

int ng_draft(const ng_seq *x, const ng_global *g, int seq, const int32_t *ctx, int n, ng_view (*view)(void *u, int seq), void *u,
             int32_t *out, int max, int *match) {
    *match = 0;
    if (n < NG_N || max <= 0) return 0;
    uint32_t key = window_key(ctx + n - NG_N);
    int best = 0, k = 0;
    // own context: an earlier occurrence; past the end, the text repeats with period n - p
    int32_t p = seq_get(x, key);
    if (p > 0 && p < n) {
        int m = match_len(ctx, p, ctx, n);
        if (m >= NG_N) {
            best = m;
            int period = n - p;
            k = max < best ? max : best;
            for (int i = 0; i < k; i++) out[i] = ctx[p + i % period];
        }
    }
    // other sequences: the latest occurrence anywhere
    if (g && g->t) {
        ng_slot s = g->t[key & g->mask];
        if (s.key == key && s.seq != seq) {
            ng_view v = view(u, s.seq);
            if (v.tok && v.gen == s.gen && s.pos <= v.n) {
                int m = match_len(v.tok, s.pos, ctx, n);
                int avail = v.n - s.pos;
                if (m >= NG_N && m > best && avail > 0) {
                    best = m;
                    k = max < best ? max : best;
                    if (k > avail) k = avail;
                    memcpy(out, v.tok + s.pos, sizeof(int32_t) * k);
                }
            }
        }
    }
    *match = best;
    return k;
}
