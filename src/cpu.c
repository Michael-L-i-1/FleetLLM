// CPU backend: f32 activations and KV. Weight matrices are repacked at load into
// 4/8-bit blocks (cpuq.h) and multiplied against int8-quantized activations with
// dot-product instructions; each weight block is read once per step and reused
// across the batch. "cpu-ref" skips the repack and dequantizes rows to f32 (the
// correctness reference for the other paths).
#include "backend.h"
#include "cpuq.h"
#include "quant.h"
#include "threadpool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static inline float dot(const float *a, const float *b, int n) {
#if defined(__ARM_NEON)
    if (!(n & 15)) {
        float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
        for (int i = 0; i < n; i += 16) {
            s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
            s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
            s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
            s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        }
        return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    }
#endif
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// y += w * x
static inline void axpy(float *y, float w, const float *x, int n) {
#if defined(__ARM_NEON)
    if (!(n & 3)) {
        for (int i = 0; i < n; i += 4) vst1q_f32(y + i, vfmaq_n_f32(vld1q_f32(y + i), vld1q_f32(x + i), w));
        return;
    }
#endif
    for (int i = 0; i < n; i++) y[i] += w * x[i];
}

typedef struct {
    float *x, *h, *q, *k, *v, *g, *att, *o, *fg, *fu, *cs, *sn, *hsel;
    size_t cap; // rows the buffers hold
    const float *in; int in_n, in_cols; // matmul input set by mm_input
    cq_act act; bool act_ok;            // its int8 copy (made on first use)
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
        qmat_row(w, r, row);
        for (int t = 0; t < j->n; t++) {
            const float *xt = j->x + (size_t)t * w->cols;
            float s = 0;
            for (int i = 0; i < w->cols; i++) s += row[i] * xt[i];
            j->y[(size_t)t * j->ldy + r] = s;
        }
    }
    free(row);
}

typedef struct { const cq_mat *w; const cq_act *a; float *y; int ldy; } cq_job;

static void cq_rows(void *ctx, int r0, int r1) {
    cq_job *j = ctx;
    cq_matmul_rows(j->w, j->a, j->y, j->ldy, r0, r1);
}

// Set the input (n x cols) of the following matmuls.
static void mm_input(cpu_state *s, const float *x, int n, int cols) {
    s->in = x;
    s->in_n = n;
    s->in_cols = cols;
    s->act_ok = false;
}

static void matmul(cpu_state *s, const qmat *w, float *y, int ldy) {
    if (w->gpu) {
        if (!s->act_ok) { cq_quantize(&s->act, s->in, s->in_n, s->in_cols); s->act_ok = true; }
        cq_job j = { w->gpu, &s->act, y, ldy };
        parallel_for(w->rows, 16, cq_rows, &j);
        return;
    }
    mm_job j = { w, s->in, y, s->in_n, ldy };
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

// ---- attention over paged KV, one task per (row, kv head) ----
// The q heads sharing a kv head are scored together, so each K / V vector is
// read (and converted from f16 / int8) once per row.
typedef struct {
    const model *m; const layer *l; const kv_cache *kv; const seq_pages *const *pages; const row_t *rows;
    const float *q; float *out;
} att_job;

// K or V vector of one token as floats (f32 storage: the stored vector itself)
static inline const float *kv_vec(const kv_pool *p, uint32_t page, int layer, int kv, int head, int slot, float *buf) {
    size_t at = kv_tile(p, page, layer, kv, head) + (size_t)slot * p->head_dim;
    int hd = p->head_dim;
    if (p->elem == 4) return (const float *)p->data + at;
    if (p->elem == 2) {
#if defined(__aarch64__)
        const __fp16 *h = (const __fp16 *)p->data + at;
        for (int i = 0; i < hd; i++) buf[i] = h[i];
#else
        for (int i = 0; i < hd; i++) buf[i] = fp16_to_f32(((const uint16_t *)p->data)[at + i]);
#endif
        return buf;
    }
    const int8_t *q = (const int8_t *)p->data + at;
    float sc = fp16_to_f32(kv_scales(p, page, layer, kv, head)[slot]);
    for (int i = 0; i < hd; i++) buf[i] = q[i] * sc;
    return buf;
}

static void att_tasks(void *ctx, int t0, int t1) {
    att_job *j = ctx;
    const model_cfg *c = &j->m->c;
    const kv_pool *pool = j->l->swa ? &j->kv->swa : &j->kv->full;
    int hd = c->hd, group = c->n_head / c->n_kv, qd = c->n_head * hd, pl = j->l->pool_layer;
    float *sc = NULL, buf[512];
    int sc_cap = 0;
    for (int t = t0; t < t1; t++) {
        int ri = t / c->n_kv, kvh = t % c->n_kv;
        const row_t *r = &j->rows[ri];
        const seq_pages *pg = j->pages[r->seq];
        int end = r->ctx_end;
        int start = j->l->swa ? (end + 1 > j->kv->window ? end + 1 - j->kv->window : 0) : 0;
        int ctx = end + 1 - start;
        if (ctx * group > sc_cap) { sc_cap = ctx * group; sc = realloc(sc, sizeof(float) * sc_cap); }
        const float *q0 = j->q + (size_t)ri * qd + (size_t)kvh * group * hd;
        float mx[64], sum[64];
        for (int g = 0; g < group; g++) mx[g] = -INFINITY;
        for (int p = start; p <= end; p++) {
            uint32_t page = j->l->swa ? pg->swa[p / KV_PAGE] : pg->full[p / KV_PAGE];
            const float *kr = kv_vec(pool, page, pl, 0, kvh, p % KV_PAGE, buf);
            for (int g = 0; g < group; g++) {
                float s = dot(q0 + (size_t)g * hd, kr, hd) * c->attn_scale;
                sc[(size_t)g * ctx + p - start] = s;
                if (s > mx[g]) mx[g] = s;
            }
        }
        for (int g = 0; g < group; g++) {
            float *sg = sc + (size_t)g * ctx, z = 0;
            for (int i = 0; i < ctx; i++) { sg[i] = expf(sg[i] - mx[g]); z += sg[i]; }
            sum[g] = 1.f / z;
        }
        float *o0 = j->out + (size_t)ri * qd + (size_t)kvh * group * hd;
        memset(o0, 0, sizeof(float) * group * hd);
        for (int p = start; p <= end; p++) {
            uint32_t page = j->l->swa ? pg->swa[p / KV_PAGE] : pg->full[p / KV_PAGE];
            const float *vr = kv_vec(pool, page, pl, 1, kvh, p % KV_PAGE, buf);
            for (int g = 0; g < group; g++) axpy(o0 + (size_t)g * hd, sc[(size_t)g * ctx + p - start] * sum[g], vr, hd);
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
    if (c->hd > 512 || c->n_head / c->n_kv > 64) { fprintf(stderr, "cpu: head_dim %d / group %d not supported\n", c->hd, c->n_head / c->n_kv); return 1; }
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
        mm_input(s, s->h, n, d);
        matmul(s, &l->wq, s->q, qd);
        matmul(s, &l->wk, s->k, kvd);
        matmul(s, &l->wv, s->v, kvd);
        if (l->wg.data) matmul(s, &l->wg, s->g, qd);
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
        for (int r = 0; r < n; r++) {
            const seq_pages *pg = pages[rows[r].seq];
            uint32_t page = l->swa ? pg->swa[rows[r].pos / KV_PAGE] : pg->full[rows[r].pos / KV_PAGE];
            for (int h = 0; h < c->n_kv; h++) {
                kv_vec_store(pool, page, l->pool_layer, 0, h, rows[r].pos % KV_PAGE, s->k + (size_t)r * kvd + (size_t)h * hd);
                kv_vec_store(pool, page, l->pool_layer, 1, h, rows[r].pos % KV_PAGE, s->v + (size_t)r * kvd + (size_t)h * hd);
            }
        }
        att_job aj = { m, l, kv, pages, rows, s->q, s->att };
        parallel_for(n * c->n_kv, 1, att_tasks, &aj);
        if (l->wg.data)
            for (size_t i = 0; i < (size_t)n * qd; i++) s->att[i] *= 1.f / (1.f + expf(-s->g[i]));
        mm_input(s, s->att, n, qd);
        matmul(s, &l->wo, s->o, d);
        residual(s->x, s->o, n, d, l->post_attn_norm, c->post_eps);

        memcpy(s->h, s->x, sizeof(float) * n * d);
        for (int r = 0; r < n; r++) rms(s->h + (size_t)r * d, d, l->ffn_norm, c->eps);
        mm_input(s, s->h, n, d);
        matmul(s, &l->w_gate, s->fg, c->ff);
        matmul(s, &l->w_up, s->fu, c->ff);
        for (size_t i = 0; i < (size_t)n * c->ff; i++) { float g = s->fg[i]; s->fg[i] = g / (1.f + expf(-g)) * s->fu[i]; }
        mm_input(s, s->fg, n, c->ff);
        matmul(s, &l->w_down, s->o, d);
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
        mm_input(s, s->hsel, ns, d);
        matmul(s, &m->output, logits, c->vocab);
        size_t nv = (size_t)ns * c->vocab;
        if (c->softcap > 0) for (size_t i = 0; i < nv; i++) logits[i] = c->softcap * tanhf(logits[i] * c->logit_scale / c->softcap);
        else if (c->logit_scale != 1.f) for (size_t i = 0; i < nv; i++) logits[i] *= c->logit_scale;
    }
    return 0;
}

static void cpu_destroy(backend *b) {
    cpu_state *s = b->impl;
    free(s->act.q); free(s->act.s); free(s->act.sq);
    free(s);
    free(b);
}

// ---- load-time repack of every streamed matrix ----
typedef struct { qmat **mats; } repack_job;

static void repack_mats(void *ctx, int i0, int i1) {
    repack_job *j = ctx;
    for (int i = i0; i < i1; i++) j->mats[i]->gpu = cq_from_qmat(j->mats[i]);
}

static void cpu_repack(model *m) {
    int n = 0;
    qmat **mats = malloc(sizeof(qmat *) * (8 * m->c.n_layer + 1));
    for (int i = 0; i < m->c.n_layer; i++) {
        layer *l = &m->layers[i];
        qmat *ls[] = { &l->wq, &l->wk, &l->wv, &l->wg, &l->wo, &l->w_gate, &l->w_up, &l->w_down };
        for (int k = 0; k < 8; k++) if (ls[k]->data && !ls[k]->gpu) mats[n++] = ls[k];
    }
    if (m->output.data && !m->output.gpu) mats[n++] = &m->output;
    repack_job j = { mats };
    parallel_for(n, 1, repack_mats, &j);
    // per format: count, and the worst error against the reference dequantizer
    // over each matrix's first rows (0 when the repack is exact)
    size_t src = 0, dst = 0;
    int count[CQ8 + 1] = { 0 }, kept = 0;
    float err[CQ8 + 1] = { 0 };
    for (int i = 0; i < n; i++) {
        const cq_mat *c = mats[i]->gpu;
        if (!c) { kept++; continue; }
        src += qmat_bytes(mats[i]);
        dst += cq_bytes(c);
        count[c->kind]++;
        float e = cq_check(mats[i], c, 4);
        if (e > err[c->kind]) err[c->kind] = e;
    }
    fprintf(stderr, "[cpu] %s, %d threads; weights %.2f GB -> %.2f GB:", cq_isa(), tp_threads(), src / 1e9, dst / 1e9);
    for (int k = 0; k <= CQ8; k++) if (count[k]) fprintf(stderr, " %d %s (err %.1e)", count[k], cq_kind_name(k), err[k]);
    if (kept) fprintf(stderr, " %d f32-dequant", kept);
    fprintf(stderr, "\n");
    free(mats);
    model_drop_host_copies(m);
}

backend *cpu_backend_new(model *m) {
    backend *b = calloc(1, sizeof *b);
    b->name = "cpu";
    b->kv_elem = m ? 2 : 4; // f16 KV (8-bit with --kv-int8); the reference path keeps f32
    b->alloc_kv = cpu_alloc;
    b->forward = cpu_forward;
    b->destroy = cpu_destroy;
    b->impl = calloc(1, sizeof(cpu_state));
    tp_init(0);
    if (m) cpu_repack(m);
    else fprintf(stderr, "[cpu] reference path (f32 dequant), %d threads\n", tp_threads());
    return b;
}

backend *backend_create(const char *name, model *m) {
#ifdef CI_METAL
    if (!strcmp(name, "metal") || !strcmp(name, "auto")) return metal_backend_new(m);
#else
    if (!strcmp(name, "metal")) { fprintf(stderr, "built without Metal\n"); exit(1); }
#endif
    if (!strcmp(name, "cpu-ref")) return cpu_backend_new(NULL);
    if (strcmp(name, "cpu") && strcmp(name, "auto")) { fprintf(stderr, "unknown device %s (auto, cpu, cpu-ref, metal)\n", name); exit(1); }
    return cpu_backend_new(m);
}
