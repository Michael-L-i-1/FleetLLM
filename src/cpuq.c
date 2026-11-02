#include "cpuq.h"
#include "gguf.h"
#include "quant.h"
#include "threadpool.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

static inline float h2f(uint16_t h) {
#if defined(__aarch64__)
    __fp16 v;
    memcpy(&v, &h, 2);
    return v;
#else
    return fp16_to_f32(h);
#endif
}

static float rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return fp16_to_f32(v); }

static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

// ---------------------------------------------------------------- repacking

static cq_mat *cq_alloc(int kind, int rows, int cols) {
    cq_mat *m = calloc(1, sizeof *m);
    size_t nb = (size_t)((rows + 3) & ~3) * (cols / CQ_BLOCK); // padding rows stay zero
    m->kind = kind;
    m->rows = rows;
    m->cols = cols;
    m->codes = calloc(nb, kind == CQ4 ? 16 : 32);
    m->scale = calloc(nb, 2);
    if (kind == CQ4) m->bias = calloc(nb, 2);
    return m;
}

// index of block g of row r: groups of 4 rows, each block's 4 rows adjacent
static inline size_t bi(int G, int r, int g) { return ((size_t)(r >> 2) * G + g) * 4 + (r & 3); }

// one CQ4 block from 32 codes (0..15) and its scale / bias
static void put4(cq_mat *m, size_t blk, const uint8_t *code, float scale, float bias) {
    uint8_t *c = m->codes + blk * 16;
    for (int j = 0; j < 16; j++) c[j] = (uint8_t)(code[j] | code[j + 16] << 4);
    m->scale[blk] = f32_to_fp16(scale);
    m->bias[blk] = f32_to_fp16(bias);
}

// one CQ8 block quantized from 32 floats (symmetric, scale = max|x| / 127)
static void put8f(cq_mat *m, size_t blk, const float *x) {
    float amax = 0;
    for (int j = 0; j < CQ_BLOCK; j++) amax = fmaxf(amax, fabsf(x[j]));
    float s = amax / 127.f, inv = s > 0 ? 1.f / s : 0;
    int8_t *c = (int8_t *)m->codes + blk * 32;
    for (int j = 0; j < CQ_BLOCK; j++) c[j] = (int8_t)lrintf(x[j] * inv);
    m->scale[blk] = f32_to_fp16(s);
}

cq_mat *cq_from_qmat(const qmat *w) {
    if (!w->data || w->cols % CQ_BLOCK) return NULL;
    int G = w->cols / CQ_BLOCK;
    uint8_t code[CQ_BLOCK];
    if (w->type == QT_AFFINE && w->bits == 4 && w->group % CQ_BLOCK == 0) { // MLX: value = scale * q + bias per group
        cq_mat *m = cq_alloc(CQ4, w->rows, w->cols);
        int ng = w->cols / w->group;
        for (int r = 0; r < w->rows; r++) {
            const uint8_t *src = w->data + (size_t)r * w->row_bytes;
            for (int b = 0; b < G; b++) {
                int g = b * CQ_BLOCK / w->group;
                uint16_t hs, hb; // safetensors data may be misaligned
                memcpy(&hs, w->scales + (size_t)r * ng + g, 2);
                memcpy(&hb, w->biases + (size_t)r * ng + g, 2);
                float s = w->scales_f16 ? fp16_to_f32(hs) : bf16_to_f32(hs), bi_ = w->scales_f16 ? fp16_to_f32(hb) : bf16_to_f32(hb);
                for (int j = 0; j < CQ_BLOCK; j++) {
                    int i = b * CQ_BLOCK + j; // little-endian nibbles
                    code[j] = (src[i >> 1] >> ((i & 1) * 4)) & 0xF;
                }
                put4(m, bi(G, r, b), code, s, bi_);
            }
        }
        return m;
    }
    if (w->type == GGML_Q4_0 || w->type == GGML_Q4_1) { // 32 per block: d (, m), nibbles j | j+16
        cq_mat *m = cq_alloc(CQ4, w->rows, w->cols);
        int bb = w->type == GGML_Q4_0 ? 18 : 20, off = w->type == GGML_Q4_0 ? 2 : 4;
        for (int r = 0; r < w->rows; r++)
            for (int b = 0; b < G; b++) {
                const uint8_t *p = w->data + (size_t)r * w->row_bytes + (size_t)b * bb;
                float d = rd16(p), bias = w->type == GGML_Q4_0 ? -8.f * d : rd16(p + 2);
                for (int j = 0; j < 16; j++) { code[j] = p[off + j] & 0xF; code[j + 16] = p[off + j] >> 4; }
                put4(m, bi(G, r, b), code, d, bias);
            }
        return m;
    }
    if (w->type == GGML_Q4_K) { // 256 per super-block: 8 sub-blocks of 32 with 6-bit scales and mins
        cq_mat *m = cq_alloc(CQ4, w->rows, w->cols);
        for (int r = 0; r < w->rows; r++)
            for (int sb = 0; sb < w->cols / 256; sb++) {
                const uint8_t *p = w->data + (size_t)r * w->row_bytes + (size_t)sb * 144, *sc = p + 4, *q = p + 16;
                float d = rd16(p), dmin = rd16(p + 2);
                for (int k = 0; k < 8; k++) {
                    uint8_t s, mn;
                    if (k < 4) { s = sc[k] & 63; mn = sc[k + 4] & 63; }
                    else { s = (sc[k + 4] & 0xF) | ((sc[k - 4] >> 6) << 4); mn = (sc[k + 4] >> 4) | ((sc[k] >> 6) << 4); }
                    const uint8_t *qq = q + (k / 2) * 32; // sub-block pairs share 32 bytes: low then high nibbles
                    for (int j = 0; j < CQ_BLOCK; j++) code[j] = k & 1 ? qq[j] >> 4 : qq[j] & 0xF;
                    put4(m, bi(G, r, sb * 8 + k), code, d * s, -dmin * mn);
                }
            }
        return m;
    }
    cq_mat *m = cq_alloc(CQ8, w->rows, w->cols);
    if (w->type == GGML_Q8_0) { // 32 per block: d, 32 int8
        for (int r = 0; r < w->rows; r++)
            for (int b = 0; b < G; b++) {
                const uint8_t *p = w->data + (size_t)r * w->row_bytes + (size_t)b * 34;
                memcpy(m->codes + bi(G, r, b) * 32, p + 2, 32);
                memcpy(&m->scale[bi(G, r, b)], p, 2);
            }
        return m;
    }
    float *row = malloc(sizeof(float) * w->cols); // anything else: dequantize, requantize to 8-bit
    for (int r = 0; r < w->rows; r++) {
        qmat_row(w, r, row);
        for (int b = 0; b < G; b++) put8f(m, bi(G, r, b), row + b * CQ_BLOCK);
    }
    free(row);
    return m;
}

void cq_free(cq_mat *m) {
    if (!m) return;
    free(m->codes);
    free(m->scale);
    free(m->bias);
    free(m);
}

size_t cq_bytes(const cq_mat *m) {
    size_t nb = (size_t)((m->rows + 3) & ~3) * (m->cols / CQ_BLOCK);
    return nb * ((m->kind == CQ4 ? 16 : 32) + (m->kind == CQ4 ? 4 : 2));
}

// ---------------------------------------------------------------- activations

static void quant_blocks(void *ctx, int b0, int b1) {
    cq_act *a = ctx;
    const float *x = a->src;
    for (size_t b = b0; b < (size_t)b1; b++) {
        const float *xb = x + b * CQ_BLOCK;
        float amax = 0;
        for (int j = 0; j < CQ_BLOCK; j++) amax = fmaxf(amax, fabsf(xb[j]));
        float s = amax / 127.f, inv = s > 0 ? 1.f / s : 0;
        int sum = 0;
        for (int j = 0; j < CQ_BLOCK; j++) {
            int q = (int)lrintf(xb[j] * inv);
            a->q[b * CQ_BLOCK + j] = (int8_t)q;
            sum += q;
        }
        a->s[b] = s;
        a->sq[b] = s * (float)sum;
    }
}

void cq_quantize(cq_act *a, const float *x, int n, int cols) {
    size_t need = (size_t)n * cols;
    if (need > a->cap) {
        a->cap = need;
        free(a->q); free(a->s); free(a->sq);
        a->q = malloc(need);
        a->s = malloc(sizeof(float) * need / CQ_BLOCK);
        a->sq = malloc(sizeof(float) * need / CQ_BLOCK);
    }
    a->n = n;
    a->cols = cols;
    a->src = x;
    int nb = (int)(need / CQ_BLOCK);
    if (nb < 4096) quant_blocks(a, 0, nb); // one decode row: not worth waking the pool
    else parallel_for(nb, 1024, quant_blocks, a);
}

// ---------------------------------------------------------------- kernels
// Per group of 4 weight rows and tile of up to 4 activation rows: each weight
// block is unpacked once and dotted against every activation row of the tile.

#if defined(__ARM_FEATURE_DOTPROD)
const char *cq_isa(void) { return "neon+dotprod"; }

static inline float32x4_t f16x4(const uint16_t *p) { return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p))); }

void cq_matmul_rows(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1) {
    const int G = w->cols / CQ_BLOCK, K = w->cols;
    const bool q4 = w->kind == CQ4;
    const uint8x16_t lowmask = vdupq_n_u8(0x0F);
    for (int rg = r0 >> 2; rg * 4 < r1; rg++) {
        const size_t base = (size_t)rg * G * 4; // first block of this row group
        for (int t0 = 0; t0 < a->n; t0 += 4) {
            const int nt = a->n - t0 < 4 ? a->n - t0 : 4;
            float32x4_t acc[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) }; // lanes: the 4 weight rows
            for (int g = 0; g < G; g++) {
                const size_t b = base + (size_t)g * 4;
                int8x16_t lo[4], hi[4];
                if (q4) {
                    for (int k = 0; k < 4; k++) {
                        uint8x16_t v = vld1q_u8(w->codes + (b + k) * 16);
                        lo[k] = vreinterpretq_s8_u8(vandq_u8(v, lowmask));
                        hi[k] = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
                    }
                } else
                    for (int k = 0; k < 4; k++) {
                        lo[k] = vld1q_s8((const int8_t *)w->codes + (b + k) * 32);
                        hi[k] = vld1q_s8((const int8_t *)w->codes + (b + k) * 32 + 16);
                    }
                const float32x4_t ws = f16x4(w->scale + b), wb = q4 ? f16x4(w->bias + b) : vdupq_n_f32(0);
                for (int i = 0; i < nt; i++) {
                    const size_t xb = (size_t)(t0 + i) * G + g;
                    const int8_t *xp = a->q + (size_t)(t0 + i) * K + (size_t)g * CQ_BLOCK;
                    const int8x16_t xl = vld1q_s8(xp), xh = vld1q_s8(xp + 16);
                    int32x4_t d[4];
                    for (int k = 0; k < 4; k++) d[k] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo[k], xl), hi[k], xh);
                    const int32x4_t dot = vpaddq_s32(vpaddq_s32(d[0], d[1]), vpaddq_s32(d[2], d[3])); // [row0..row3]
                    acc[i] = vfmaq_f32(acc[i], vcvtq_f32_s32(dot), vmulq_n_f32(ws, a->s[xb]));
                    if (q4) acc[i] = vfmaq_n_f32(acc[i], wb, a->sq[xb]);
                }
            }
            for (int i = 0; i < nt; i++) {
                float o[4];
                vst1q_f32(o, acc[i]);
                for (int k = 0; k < 4 && rg * 4 + k < r1; k++) y[(size_t)(t0 + i) * ldy + rg * 4 + k] = o[k];
            }
        }
    }
}
#else
const char *cq_isa(void) { return "scalar"; }

void cq_matmul_rows(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1) {
    const int G = w->cols / CQ_BLOCK, K = w->cols;
    const bool q4 = w->kind == CQ4;
    int8_t blk[CQ_BLOCK];
    for (int r = r0; r < r1; r++)
        for (int t = 0; t < a->n; t++) {
            float acc = 0;
            for (int g = 0; g < G; g++) {
                size_t wbk = bi(G, r, g), xb = (size_t)t * G + g;
                if (q4) {
                    const uint8_t *c = w->codes + wbk * 16;
                    for (int j = 0; j < 16; j++) { blk[j] = c[j] & 0xF; blk[j + 16] = c[j] >> 4; }
                } else memcpy(blk, w->codes + wbk * 32, 32);
                const int8_t *xp = a->q + (size_t)t * K + (size_t)g * CQ_BLOCK;
                int d = 0;
                for (int j = 0; j < CQ_BLOCK; j++) d += blk[j] * xp[j];
                acc += h2f(w->scale[wbk]) * a->s[xb] * (float)d + (q4 ? h2f(w->bias[wbk]) * a->sq[xb] : 0);
            }
            y[(size_t)t * ldy + r] = acc;
        }
}
#endif
