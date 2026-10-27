// CPU reference backend: f32 activations and KV, weights dequantized one row at
// a time and dotted against every token in the batch (weights are read once per
// step regardless of batch size). Correctness reference for the GPU backend and
// the starting point for the Pi path; NEON kernels come later.
#include "backend.h"
#include "quant.h"
#include "threadpool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *x, *h, *q, *k, *v, *g, *att, *o, *fg, *fu, *cs, *sn, *hsel;
    size_t cap; // rows the buffers hold
} cpu_state;

static void *cpu_alloc(backend *b, size_t bytes, void **handle) {
    *handle = NULL;
    void *p = calloc(1, bytes);
    if (!p) { fprintf(stderr, "cpu: KV allocation of %zu bytes failed\n", bytes); exit(1); }
    return p;
}

// ---- batched matmul: y[t*ldy + r] = W[r] . x[t] ----
typedef struct { const qmat *w; const float *x; float *y; int n, ldy; } mm_job;

static void mm_rows(void *ctx, int r0, int r1) {
    mm_job *j = ctx;
    const qmat *w = j->w;
    float *row = malloc(sizeof(float) * w->cols);
    for (int r = r0; r < r1; r++) {
        dequant_row(w->type, w->data + (size_t)r * w->row_bytes, row, w->cols);
        for (int t = 0; t < j->n; t++) {
            const float *xt = j->x + (size_t)t * w->cols;
            float s = 0;
            for (int i = 0; i < w->cols; i++) s += row[i] * xt[i];
            j->y[(size_t)t * j->ldy + r] = s;
        }
    }
    free(row);
}

static void matmul(const qmat *w, const float *x, float *y, int n, int ldy) {
    mm_job j = { w, x, y, n, ldy };
    parallel_for(w->rows, 16, mm_rows, &j);
}

static void rms(float *x, int d, const float *w, float eps) {
    double ss = 0;
    for (int i = 0; i < d; i++) ss += (double)x[i] * x[i];
    float s = 1.f / sqrtf((float)(ss / d) + eps);
    if (w) for (int i = 0; i < d; i++) x[i] *= s * w[i];
    else for (int i = 0; i < d; i++) x[i] *= s;
}

static void rope(float *x, const float *cs, const float *sn, int half, bool neox) {
    for (int i = 0; i < half; i++) {
        int ia = neox ? i : 2 * i, ib = neox ? i + half : 2 * i + 1;
        float a = x[ia], b = x[ib];
        x[ia] = a * cs[i] - b * sn[i];
        x[ib] = a * sn[i] + b * cs[i];
    }
}

// ---- attention over paged KV, one task per (row, q head) ----
typedef struct {
    const model *m; const layer *l; const kv_cache *kv; const seq_pages *const *pages; const row_t *rows;
    const float *q; float *out;
} att_job;

static void att_tasks(void *ctx, int t0, int t1) {
    att_job *j = ctx;
    const model_cfg *c = &j->m->c;
    const kv_pool *pool = j->l->swa ? &j->kv->swa : &j->kv->full;
    const float *data = (const float *)pool->data;
    int hd = c->hd, group = c->n_head / c->n_kv, qd = c->n_head * hd;
    float *sc = NULL;
    int sc_cap = 0;
    for (int t = t0; t < t1; t++) {
        int ri = t / c->n_head, h = t % c->n_head, kvh = h / group;
        const row_t *r = &j->rows[ri];
        const seq_pages *pg = j->pages[r->seq];
        int end = r->ctx_end;
        int start = j->l->swa ? (end + 1 > j->kv->window ? end + 1 - j->kv->window : 0) : 0;
        int ctx = end + 1 - start;
        if (ctx > sc_cap) { sc_cap = ctx; sc = realloc(sc, sizeof(float) * sc_cap); }
        const float *qh = j->q + (size_t)ri * qd + (size_t)h * hd;
        float mx = -INFINITY;
        for (int p = start; p <= end; p++) {
            uint32_t page = j->l->swa ? pg->swa[p / KV_PAGE] : pg->full[p / KV_PAGE];
            const float *kr = data + kv_tile(pool, page, j->l->pool_layer, 0, kvh) + (size_t)(p % KV_PAGE) * hd;
            float s = 0;
            for (int i = 0; i < hd; i++) s += qh[i] * kr[i];
            s *= c->attn_scale;
            sc[p - start] = s;
            if (s > mx) mx = s;
        }
        float sum = 0;
        for (int i = 0; i < ctx; i++) { sc[i] = expf(sc[i] - mx); sum += sc[i]; }
        float *o = j->out + (size_t)ri * qd + (size_t)h * hd;
        memset(o, 0, sizeof(float) * hd);
        for (int p = start; p <= end; p++) {
            uint32_t page = j->l->swa ? pg->swa[p / KV_PAGE] : pg->full[p / KV_PAGE];
            const float *vr = data + kv_tile(pool, page, j->l->pool_layer, 1, kvh) + (size_t)(p % KV_PAGE) * hd;
            float w = sc[p - start] / sum;
            for (int i = 0; i < hd; i++) o[i] += w * vr[i];
        }
    }
    free(sc);
}

static void ensure(cpu_state *s, const model_cfg *c, int n) {
    if ((size_t)n <= s->cap) return;
    size_t qd = (size_t)c->n_head * c->hd, kvd = (size_t)c->n_kv * c->hd;
    float **bufs[] = { &s->x, &s->h, &s->q, &s->k, &s->v, &s->g, &s->att, &s->o, &s->fg, &s->fu, &s->cs, &s->sn, &s->hsel };
    size_t per[] = { c->d, c->d, qd, kvd, kvd, qd, qd, c->d, c->ff, c->ff, c->hd / 2, c->hd / 2, c->d };
    for (size_t i = 0; i < sizeof per / sizeof *per; i++) { free(*bufs[i]); *bufs[i] = malloc(sizeof(float) * per[i] * n); }
    s->cap = n;
}

static void residual(float *x, float *o, int n, int d, const float *post, float eps) {
    for (int r = 0; r < n; r++) {
        float *orow = o + (size_t)r * d;
        if (post) rms(orow, d, post, eps);
        float *xr = x + (size_t)r * d;
        for (int i = 0; i < d; i++) xr[i] += orow[i];
    }
}

static int cpu_forward(backend *b, const model *m, kv_cache *kv, const seq_pages *const *pages, const row_t *rows, int n, float *logits) {
    cpu_state *s = b->impl;
    const model_cfg *c = &m->c;
    ensure(s, c, n);
    int d = c->d, hd = c->hd, half = hd / 2, qd = c->n_head * hd, kvd = c->n_kv * hd;
    int32_t *toks = malloc(sizeof(int32_t) * n);
    for (int i = 0; i < n; i++) toks[i] = rows[i].token;
    model_embed(m, toks, n, s->x);
    free(toks);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < half; j++) {
            float a = rows[i].pos * m->inv_freq[j];
            s->cs[i * half + j] = cosf(a);
            s->sn[i * half + j] = sinf(a);
        }
    for (int li = 0; li < c->n_layer; li++) {
        const layer *l = &m->layers[li];
        memcpy(s->h, s->x, sizeof(float) * n * d);
        for (int r = 0; r < n; r++) rms(s->h + (size_t)r * d, d, l->attn_norm, c->eps);
        matmul(&l->wq, s->h, s->q, n, qd);
        matmul(&l->wk, s->h, s->k, n, kvd);
        matmul(&l->wv, s->h, s->v, n, kvd);
        if (l->wg.data) matmul(&l->wg, s->h, s->g, n, qd);
        for (int r = 0; r < n; r++) {
            if (l->bq) for (int i = 0; i < qd; i++) s->q[(size_t)r * qd + i] += l->bq[i];
            if (l->bk) for (int i = 0; i < kvd; i++) s->k[(size_t)r * kvd + i] += l->bk[i];
            if (l->bv) for (int i = 0; i < kvd; i++) s->v[(size_t)r * kvd + i] += l->bv[i];
            for (int h = 0; h < c->n_head + c->n_kv; h++) {
                bool isq = h < c->n_head;
                float *x = isq ? s->q + (size_t)r * qd + (size_t)h * hd : s->k + (size_t)r * kvd + (size_t)(h - c->n_head) * hd;
                const float *nw = isq ? l->q_norm : l->k_norm;
                if (nw) rms(x, hd, nw, c->eps);
                if (l->rope) rope(x, s->cs + r * half, s->sn + r * half, half, c->rope_neox);
            }
        }
        // write K/V for all rows first so prefill rows see each other
        kv_pool *pool = l->swa ? &kv->swa : &kv->full;
        float *data = (float *)pool->data;
        for (int r = 0; r < n; r++) {
            const seq_pages *pg = pages[rows[r].seq];
            uint32_t page = l->swa ? pg->swa[rows[r].pos / KV_PAGE] : pg->full[rows[r].pos / KV_PAGE];
            for (int h = 0; h < c->n_kv; h++) {
                size_t off = (size_t)(rows[r].pos % KV_PAGE) * hd;
                memcpy(data + kv_tile(pool, page, l->pool_layer, 0, h) + off, s->k + (size_t)r * kvd + (size_t)h * hd, sizeof(float) * hd);
                memcpy(data + kv_tile(pool, page, l->pool_layer, 1, h) + off, s->v + (size_t)r * kvd + (size_t)h * hd, sizeof(float) * hd);
            }
        }
        att_job aj = { m, l, kv, pages, rows, s->q, s->att };
        parallel_for(n * c->n_head, 4, att_tasks, &aj);
        if (l->wg.data)
            for (size_t i = 0; i < (size_t)n * qd; i++) s->att[i] *= 1.f / (1.f + expf(-s->g[i]));
        matmul(&l->wo, s->att, s->o, n, d);
        residual(s->x, s->o, n, d, l->post_attn_norm, c->post_eps);

        memcpy(s->h, s->x, sizeof(float) * n * d);
        for (int r = 0; r < n; r++) rms(s->h + (size_t)r * d, d, l->ffn_norm, c->eps);
        matmul(&l->w_gate, s->h, s->fg, n, c->ff);
        matmul(&l->w_up, s->h, s->fu, n, c->ff);
        for (size_t i = 0; i < (size_t)n * c->ff; i++) { float g = s->fg[i]; s->fg[i] = g / (1.f + expf(-g)) * s->fu[i]; }
        matmul(&l->w_down, s->fg, s->o, n, d);
        residual(s->x, s->o, n, d, l->post_ffn_norm, c->post_eps);
    }
    int ns = 0;
    for (int r = 0; r < n; r++) {
        if (!rows[r].logits) continue;
        memcpy(s->hsel + (size_t)ns * d, s->x + (size_t)r * d, sizeof(float) * d);
        rms(s->hsel + (size_t)ns * d, d, m->out_norm, c->eps);
        ns++;
    }
    if (ns) {
        matmul(&m->output, s->hsel, logits, ns, c->vocab);
        size_t nv = (size_t)ns * c->vocab;
        if (c->softcap > 0) for (size_t i = 0; i < nv; i++) logits[i] = c->softcap * tanhf(logits[i] * c->logit_scale / c->softcap);
        else if (c->logit_scale != 1.f) for (size_t i = 0; i < nv; i++) logits[i] *= c->logit_scale;
    }
    return 0;
}

static void cpu_destroy(backend *b) { free(b->impl); free(b); }

backend *cpu_backend_new(void) {
    backend *b = calloc(1, sizeof *b);
    b->name = "cpu";
    b->kv_elem = 4;
    b->alloc_kv = cpu_alloc;
    b->forward = cpu_forward;
    b->destroy = cpu_destroy;
    b->impl = calloc(1, sizeof(cpu_state));
    tp_init(0);
    return b;
}

backend *backend_create(const char *name, const model *m) {
#ifdef CI_METAL
    if (!strcmp(name, "metal") || !strcmp(name, "auto")) return metal_backend_new(m);
#else
    if (!strcmp(name, "metal")) { fprintf(stderr, "built without Metal\n"); exit(1); }
#endif
    return cpu_backend_new();
}
