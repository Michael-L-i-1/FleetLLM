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

typedef struct { int ri, kvh, p0, p1, part; } att_task; // attention work item; part: partial slot (-1: single chunk)

typedef struct {
    float *x, *h, *q, *k, *v, *g, *att, *o, *fg, *fu, *cs, *sn, *hsel;
    size_t cap; // rows the buffers hold
    const float *in; int in_n, in_cols; // matmul input set by mm_input
    cq_act act; bool act_ok;            // its int8 copy (made on first use)
    att_task *att_tasks; int *att_first; size_t att_cap; // attention task list (see attention())
    float *part; size_t part_cap;                    // attention chunk partials
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

// FLEET_PROFILE=1: time per forward part, printed at exit
enum { P_MATMUL, P_OUTPUT, P_ATTN, P_FORWARD, P_N };
static double prof[P_N];
static long prof_steps;
static bool prof_on;
double now_secs(void);

static void prof_print(void) {
    if (!prof_steps) return;
    double f = prof[P_FORWARD] * 1e3 / prof_steps;
    fprintf(stderr, "[cpu profile] %ld forwards, %.2f ms each: matmul %.2f, output layer %.2f, attention %.2f, other %.2f\n", prof_steps, f,
            prof[P_MATMUL] * 1e3 / prof_steps, prof[P_OUTPUT] * 1e3 / prof_steps, prof[P_ATTN] * 1e3 / prof_steps,
            f - (prof[P_MATMUL] + prof[P_OUTPUT] + prof[P_ATTN]) * 1e3 / prof_steps);
}

static void matmul(cpu_state *s, const qmat *w, float *y, int ldy) {
    double t0 = prof_on ? now_secs() : 0;
    if (w->gpu) {
        if (!s->act_ok) { cq_quantize(&s->act, s->in, s->in_n, s->in_cols); s->act_ok = true; }
        cq_job j = { w->gpu, &s->act, y, ldy };
        parallel_for(w->rows, 64, cq_rows, &j); // contiguous 64-row runs stream better than finer chunks
    } else {
        mm_job j = { w, s->in, y, s->in_n, ldy };
        parallel_for(w->rows, 16, mm_rows, &j);
    }
    if (prof_on) prof[P_MATMUL] += now_secs() - t0;
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

// e^x for x <= 0 (softmax, after subtracting the max): Cephes-style range
// reduction and degree-6 polynomial, ~1 ulp; vectorized where NEON is available.
#if defined(__ARM_NEON)
static inline float32x4_t vexp(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.f));
    float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504f));
    float32x4_t r = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0.693359375f)), n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    p = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.f)), p, vmulq_f32(r, r));
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(vcvtq_s32_f32(n), 23)));
}
#endif

// x[i] = e^(x[i] - mx); returns the sum
static float exp_sum(float *x, int n, float mx) {
    float z = 0;
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t vz = vdupq_n_f32(0), vm = vdupq_n_f32(mx);
    for (; i + 4 <= n; i += 4) {
        float32x4_t e = vexp(vsubq_f32(vld1q_f32(x + i), vm));
        vst1q_f32(x + i, e);
        vz = vaddq_f32(vz, e);
    }
    z = vaddvq_f32(vz);
#endif
    for (; i < n; i++) { x[i] = expf(x[i] - mx); z += x[i]; }
    return z;
}

// ---- attention over paged KV ----
// One task per (row, kv head, chunk of ATT_CHUNK positions): the q heads sharing
// a kv head are scored together, so each K / V vector is read (and converted
// from f16 / int8) once per row, and a single decode row still spreads over all
// threads. Chunks write unnormalized partials (max, sum, output) that a second
// pass merges (flash-decoding); a row with one chunk writes its output directly.
#define ATT_CHUNK 256

typedef struct {
    const model *m; const layer *l; const kv_cache *kv; const seq_pages *const *pages; const row_t *rows;
    const float *q; float *out;
    const att_task *tasks;
    float *part; // per partial slot: group maxima, group sums, group x hd outputs
} att_job;

// K or V of slots [s0, s1) of one page tile as rows of head_dim floats (f32
// storage: the stored tile itself; else converted into buf)
static const float *kv_rows(const kv_pool *p, uint32_t page, int layer, int kv, int head, int s0, int s1, float *buf) {
    int hd = p->head_dim, n = (s1 - s0) * hd;
    size_t at = kv_tile(p, page, layer, kv, head) + (size_t)s0 * hd;
    if (p->elem == 4) return (const float *)p->data + at;
    if (p->elem == 2) {
#if defined(__aarch64__)
        const __fp16 *h = (const __fp16 *)p->data + at;
        for (int i = 0; i < n; i++) buf[i] = h[i];
#else
        for (int i = 0; i < n; i++) buf[i] = fp16_to_f32(((const uint16_t *)p->data)[at + i]);
#endif
        return buf;
    }
    const int8_t *q = (const int8_t *)p->data + at;
    const uint16_t *sc = kv_scales(p, page, layer, kv, head);
    for (int t = 0; t < s1 - s0; t++) {
        float f = fp16_to_f32(sc[s0 + t]);
        for (int i = 0; i < hd; i++) buf[t * hd + i] = q[t * hd + i] * f;
    }
    return buf;
}

// out[i] = scale * q . k[i] for np rows k of hd floats
static void tile_scores(const float *q, const float *k, int np, int hd, float scale, float *out) {
    int i = 0;
#if defined(__ARM_NEON)
    if (!(hd & 3))
        for (; i + 4 <= np; i += 4) { // 4 rows per pass share each q load
            const float *k0 = k + (size_t)i * hd;
            float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
            for (int e = 0; e < hd; e += 4) {
                float32x4_t qv = vld1q_f32(q + e);
                a0 = vfmaq_f32(a0, qv, vld1q_f32(k0 + e));
                a1 = vfmaq_f32(a1, qv, vld1q_f32(k0 + hd + e));
                a2 = vfmaq_f32(a2, qv, vld1q_f32(k0 + 2 * hd + e));
                a3 = vfmaq_f32(a3, qv, vld1q_f32(k0 + 3 * hd + e));
            }
            vst1q_f32(out + i, vmulq_n_f32(vpaddq_f32(vpaddq_f32(a0, a1), vpaddq_f32(a2, a3)), scale));
        }
#endif
    for (; i < np; i++) out[i] = dot(q, k + (size_t)i * hd, hd) * scale;
}

// o += sum_i w[i] * v[i] for np rows v of hd floats
static void tile_accum(float *o, const float *w, const float *v, int np, int hd) {
#if defined(__ARM_NEON)
    if (!(hd & 31)) {
        for (int e = 0; e < hd; e += 32) { // 32 outputs stay in registers across the rows
            float32x4_t a[8];
            for (int k = 0; k < 8; k++) a[k] = vld1q_f32(o + e + 4 * k);
            for (int i = 0; i < np; i++) {
                const float *vr = v + (size_t)i * hd + e;
                for (int k = 0; k < 8; k++) a[k] = vfmaq_n_f32(a[k], vld1q_f32(vr + 4 * k), w[i]);
            }
            for (int k = 0; k < 8; k++) vst1q_f32(o + e + 4 * k, a[k]);
        }
        return;
    }
#endif
    for (int i = 0; i < np; i++) axpy(o, w[i], v + (size_t)i * hd, hd);
}

static float max_of(const float *x, int n) {
    float m = -INFINITY;
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t vm = vdupq_n_f32(-INFINITY);
    for (; i + 4 <= n; i += 4) vm = vmaxq_f32(vm, vld1q_f32(x + i));
    m = vmaxvq_f32(vm);
#endif
    for (; i < n; i++) m = fmaxf(m, x[i]);
    return m;
}

// Prefetch the (layer, kv, head) tile of the page holding position p (if < end):
// tiles of consecutive pages are a whole page stride apart, too far for the
// hardware prefetcher to follow.
static inline void kv_prefetch(const kv_pool *pool, const uint32_t *pages, int p, int end, int layer, int kv, int head) {
    if (p >= end) return;
    const uint8_t *t = pool->data + kv_tile(pool, pages[p / KV_PAGE], layer, kv, head) * pool->elem;
    size_t bytes = (size_t)KV_PAGE * pool->head_dim * pool->elem;
    for (size_t o = 0; o < bytes; o += 64) __builtin_prefetch(t + o);
    if (pool->elem == 1) __builtin_prefetch(kv_scales(pool, pages[p / KV_PAGE], layer, kv, head));
}

static void att_tasks(void *ctx, int t0, int t1) {
    att_job *j = ctx;
    const model_cfg *c = &j->m->c;
    const kv_pool *pool = j->l->swa ? &j->kv->swa : &j->kv->full;
    int hd = c->hd, group = c->n_head / c->n_kv, qd = c->n_head * hd, pl = j->l->pool_layer;
    float sc[64 * ATT_CHUNK], buf[KV_PAGE * 512];
    for (int t = t0; t < t1; t++) {
        const att_task *tk = &j->tasks[t];
        const seq_pages *pg = j->pages[j->rows[tk->ri].seq];
        const uint32_t *pages = j->l->swa ? pg->swa : pg->full;
        int ctx = tk->p1 - tk->p0;
        const float *q0 = j->q + (size_t)tk->ri * qd + (size_t)tk->kvh * group * hd;
        // scores, one KV page tile at a time
        for (int p = tk->p0; p < tk->p1;) {
            int s0 = p % KV_PAGE, s1 = s0 + (tk->p1 - p) < KV_PAGE ? s0 + (tk->p1 - p) : KV_PAGE;
            kv_prefetch(pool, pages, p - s0 + 2 * KV_PAGE, tk->p1, pl, 0, tk->kvh);
            if (p == tk->p0) kv_prefetch(pool, pages, p - s0 + KV_PAGE, tk->p1, pl, 0, tk->kvh);
            const float *kt = kv_rows(pool, pages[p / KV_PAGE], pl, 0, tk->kvh, s0, s1, buf);
            for (int g = 0; g < group; g++) tile_scores(q0 + (size_t)g * hd, kt, s1 - s0, hd, c->attn_scale, sc + g * ctx + p - tk->p0);
            p += s1 - s0;
        }
        float mx[64], sum[64];
        float *part = tk->part >= 0 ? j->part + (size_t)tk->part * group * (hd + 2) : NULL;
        for (int g = 0; g < group; g++) {
            float *sg = sc + g * ctx;
            mx[g] = max_of(sg, ctx);
            sum[g] = exp_sum(sg, ctx, mx[g]);
            if (!part) for (int i = 0; i < ctx; i++) sg[i] /= sum[g]; // final weights
        }
        float *o0 = part ? part + 2 * group : j->out + (size_t)tk->ri * qd + (size_t)tk->kvh * group * hd;
        memset(o0, 0, sizeof(float) * group * hd);
        for (int p = tk->p0; p < tk->p1;) {
            int s0 = p % KV_PAGE, s1 = s0 + (tk->p1 - p) < KV_PAGE ? s0 + (tk->p1 - p) : KV_PAGE;
            kv_prefetch(pool, pages, p - s0 + 2 * KV_PAGE, tk->p1, pl, 1, tk->kvh);
            if (p == tk->p0) kv_prefetch(pool, pages, p - s0 + KV_PAGE, tk->p1, pl, 1, tk->kvh);
            const float *vt = kv_rows(pool, pages[p / KV_PAGE], pl, 1, tk->kvh, s0, s1, buf);
            for (int g = 0; g < group; g++) tile_accum(o0 + (size_t)g * hd, sc + g * ctx + p - tk->p0, vt, s1 - s0, hd);
            p += s1 - s0;
        }
        if (part) { memcpy(part, mx, sizeof(float) * group); memcpy(part + group, sum, sizeof(float) * group); }
    }
}

// x = silu(x) * u
static void silu_mul(float *x, const float *u, size_t n) {
    size_t i = 0;
#if defined(__ARM_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t g = vld1q_f32(x + i), e = vexp(vminq_f32(vnegq_f32(g), vdupq_n_f32(87.f)));
        vst1q_f32(x + i, vmulq_f32(vdivq_f32(g, vaddq_f32(vdupq_n_f32(1.f), e)), vld1q_f32(u + i)));
    }
#endif
    for (; i < n; i++) x[i] = x[i] / (1.f + expf(-x[i])) * u[i];
}

// merge the chunk partials of the (row, kv head) whose first task is tasks[t]
typedef struct { att_job *j; const int *first; } merge_job;

static void att_merge(void *ctx, int i0, int i1) {
    merge_job *mj = ctx;
    att_job *j = mj->j;
    const model_cfg *c = &j->m->c;
    int hd = c->hd, group = c->n_head / c->n_kv, qd = c->n_head * hd;
    for (int i = i0; i < i1; i++) {
        const att_task *t0 = &j->tasks[mj->first[i]];
        int nch = (j->rows[t0->ri].ctx_end + 1 - t0->p0 + ATT_CHUNK - 1) / ATT_CHUNK;
        float *o0 = j->out + (size_t)t0->ri * qd + (size_t)t0->kvh * group * hd;
        for (int g = 0; g < group; g++) {
            float M = -INFINITY, L = 0;
            for (int k = 0; k < nch; k++) M = fmaxf(M, j->part[(size_t)(t0->part + k) * group * (hd + 2) + g]);
            float *o = o0 + (size_t)g * hd;
            memset(o, 0, sizeof(float) * hd);
            for (int k = 0; k < nch; k++) {
                const float *pp = j->part + (size_t)(t0->part + k) * group * (hd + 2);
                float w = expf(pp[g] - M);
                L += pp[group + g] * w;
                axpy(o, w, pp + 2 * group + (size_t)g * hd, hd);
            }
            for (int d = 0; d < hd; d++) o[d] /= L;
        }
    }
}

// attention for all rows of one layer
static void attention(cpu_state *s, const model *m, const layer *l, const kv_cache *kv, const seq_pages *const *pages, const row_t *rows, int n) {
    const model_cfg *c = &m->c;
    int nt = 0, np = 0, nm = 0, group = c->n_head / c->n_kv;
    for (int pass = 0; pass < 2; pass++) { // count, then fill
        nt = np = nm = 0;
        for (int r = 0; r < n; r++) {
            int end = rows[r].ctx_end + 1, start = l->swa && end > kv->window ? end - kv->window : 0;
            int nch = (end - start + ATT_CHUNK - 1) / ATT_CHUNK;
            for (int h = 0; h < c->n_kv; h++) {
                if (pass && nch > 1) s->att_first[nm] = nt;
                nm += nch > 1;
                for (int k = 0; k < nch; k++, nt++) {
                    if (pass) {
                        int p0 = start + k * ATT_CHUNK;
                        s->att_tasks[nt] = (att_task){ r, h, p0, p0 + ATT_CHUNK < end ? p0 + ATT_CHUNK : end, nch > 1 ? np : -1 };
                    }
                    np += nch > 1;
                }
            }
        }
        if (!pass) {
            if ((size_t)nt > s->att_cap) {
                s->att_cap = nt;
                s->att_tasks = realloc(s->att_tasks, sizeof(att_task) * nt);
                s->att_first = realloc(s->att_first, sizeof(int) * (nt + 1));
            }
            if ((size_t)np * group * (c->hd + 2) > s->part_cap) { s->part_cap = (size_t)np * group * (c->hd + 2); free(s->part); s->part = malloc(sizeof(float) * s->part_cap); }
        }
    }
    s->att_first[nm] = nt;
    att_job aj = { m, l, kv, pages, rows, s->q, s->att, s->att_tasks, s->part };
    parallel_for(nt, 1, att_tasks, &aj);
    merge_job mj = { &aj, s->att_first };
    if (nm) parallel_for(nm, 1, att_merge, &mj);
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
    double tf = prof_on ? now_secs() : 0;
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
        double ta = prof_on ? now_secs() : 0;
        attention(s, m, l, kv, pages, rows, n);
        if (prof_on) prof[P_ATTN] += now_secs() - ta;
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
        silu_mul(s->fg, s->fu, (size_t)n * c->ff);
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
        double to = prof_on ? now_secs() : 0, tm = prof[P_MATMUL];
        matmul(s, &m->output, logits, c->vocab);
        if (prof_on) { prof[P_OUTPUT] += now_secs() - to; prof[P_MATMUL] = tm; }
        size_t nv = (size_t)ns * c->vocab;
        if (c->softcap > 0) for (size_t i = 0; i < nv; i++) logits[i] = c->softcap * tanhf(logits[i] * c->logit_scale / c->softcap);
        else if (c->logit_scale != 1.f) for (size_t i = 0; i < nv; i++) logits[i] *= c->logit_scale;
    }
    if (prof_on) { prof[P_FORWARD] += now_secs() - tf; prof_steps++; }
    return 0;
}

static void cpu_destroy(backend *b) {
    cpu_state *s = b->impl;
    free(s->act.q); free(s->act.s); free(s->act.sq);
    free(s->att_tasks); free(s->att_first); free(s->part);
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
    if (getenv("FLEET_PROFILE")) { prof_on = true; atexit(prof_print); }
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
