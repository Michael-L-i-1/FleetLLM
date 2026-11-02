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

static float rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return fp16_to_f32(v); }

static float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

static int code_bytes(int kind) { return kind == CQ8 ? 32 : kind == CQ6K ? 24 : 16; }

const char *cq_kind_name(int kind) {
    static const char *n[] = { "4-bit affine", "4-bit (Q4_0)", "4-bit (Q4_K)", "6-bit (Q6_K)", "8-bit" };
    return n[kind];
}

// ---------------------------------------------------------------- repacking

static cq_mat *cq_alloc(int kind, int rows, int cols) {
    cq_mat *m = calloc(1, sizeof *m);
    size_t nb = (size_t)((rows + 3) & ~3) * (cols / CQ_BLOCK); // padding rows stay zero
    m->kind = kind;
    m->rows = rows;
    m->cols = cols;
    m->codes = calloc(nb, code_bytes(kind));
    if (kind == CQ4K || kind == CQ6K) {
        m->sub = calloc(nb, 2);
        m->super = calloc(nb / 8 * 2, 2);
    } else m->scale = calloc(nb, 2);
    if (kind == CQ4) m->bias = calloc(nb, 2);
    return m;
}

// index of block g of row r: groups of 4 rows, each block's 4 rows adjacent
static inline size_t bi(int G, int r, int g) { return ((size_t)(r >> 2) * G + g) * 4 + (r & 3); }
// sub-block scale bytes of block g of row r (second byte at +4)
static inline uint8_t *sub_at(const cq_mat *m, int r, int g) { return m->sub + ((size_t)(r >> 2) * (m->cols / CQ_BLOCK) + g) * 8 + (r & 3); }
// f16 d of the 256 values of row r holding block g (dmin at +4)
static inline uint16_t *super_at(const cq_mat *m, int r, int g) { return m->super + ((size_t)(r >> 2) * (m->cols / 256) + g / 8) * 8 + (r & 3); }

// 4-bit codes (0..15) of one block
static void put_codes4(cq_mat *m, size_t blk, const uint8_t *code) {
    uint8_t *c = m->codes + blk * 16;
    for (int j = 0; j < 16; j++) c[j] = (uint8_t)(code[j] | code[j + 16] << 4);
}

// 6-bit codes (0..63) of one block
static void put_codes6(cq_mat *m, size_t blk, const uint8_t *code) {
    uint8_t *c = m->codes + blk * 24;
    for (int j = 0; j < 16; j++) c[j] = (uint8_t)((code[j] & 15) | (code[j + 16] & 15) << 4);
    for (int j = 0; j < 8; j++) c[16 + j] = (uint8_t)(code[j] >> 4 | (code[j + 8] >> 4) << 2 | (code[j + 16] >> 4) << 4 | (code[j + 24] >> 4) << 6);
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
    uint8_t code[256];
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
                for (int j = 0; j < CQ_BLOCK; j++) {
                    int i = b * CQ_BLOCK + j; // little-endian nibbles
                    code[j] = (src[i >> 1] >> ((i & 1) * 4)) & 0xF;
                }
                put_codes4(m, bi(G, r, b), code);
                m->scale[bi(G, r, b)] = w->scales_f16 ? hs : f32_to_fp16(bf16_to_f32(hs));
                m->bias[bi(G, r, b)] = w->scales_f16 ? hb : f32_to_fp16(bf16_to_f32(hb));
            }
        }
        return m;
    }
    if (w->type == GGML_Q4_0 || w->type == GGML_Q4_1) { // 32 per block: d (, m), nibbles j | j+16
        bool q40 = w->type == GGML_Q4_0;
        cq_mat *m = cq_alloc(q40 ? CQ4S : CQ4, w->rows, w->cols);
        int bb = q40 ? 18 : 20, off = q40 ? 2 : 4;
        for (int r = 0; r < w->rows; r++)
            for (int b = 0; b < G; b++) {
                const uint8_t *p = w->data + (size_t)r * w->row_bytes + (size_t)b * bb;
                size_t k = bi(G, r, b);
                memcpy(m->codes + k * 16, p + off, 16); // same nibble order
                memcpy(&m->scale[k], p, 2);
                if (!q40) memcpy(&m->bias[k], p + 2, 2);
            }
        return m;
    }
    if (w->type == GGML_Q4_K) { // 256 per super-block: 8 sub-blocks of 32 with 6-bit scales and mins
        cq_mat *m = cq_alloc(CQ4K, w->rows, w->cols);
        for (int r = 0; r < w->rows; r++)
            for (int sb = 0; sb < w->cols / 256; sb++) {
                const uint8_t *p = w->data + (size_t)r * w->row_bytes + (size_t)sb * 144, *sc = p + 4, *q = p + 16;
                memcpy(super_at(m, r, sb * 8), p, 2);
                memcpy(super_at(m, r, sb * 8) + 4, p + 2, 2);
                for (int k = 0; k < 8; k++) {
                    uint8_t s, mn;
                    if (k < 4) { s = sc[k] & 63; mn = sc[k + 4] & 63; }
                    else { s = (sc[k + 4] & 0xF) | ((sc[k - 4] >> 6) << 4); mn = (sc[k + 4] >> 4) | ((sc[k] >> 6) << 4); }
                    const uint8_t *qq = q + (k / 2) * 32; // sub-block pairs share 32 bytes: low then high nibbles
                    for (int j = 0; j < CQ_BLOCK; j++) code[j] = k & 1 ? qq[j] >> 4 : qq[j] & 0xF;
                    put_codes4(m, bi(G, r, sb * 8 + k), code);
                    sub_at(m, r, sb * 8 + k)[0] = s;
                    sub_at(m, r, sb * 8 + k)[4] = mn;
                }
            }
        return m;
    }
    if (w->type == GGML_Q6_K) { // 256 per super-block: 4 low + 2 high bits per value, int8 scale per 16, f16 d
        cq_mat *m = cq_alloc(CQ6K, w->rows, w->cols);
        for (int r = 0; r < w->rows; r++)
            for (int sb = 0; sb < w->cols / 256; sb++) {
                const uint8_t *p = w->data + (size_t)r * w->row_bytes + (size_t)sb * 210, *ql = p, *qh = p + 128;
                const int8_t *sc = (const int8_t *)(p + 192);
                for (int n = 0; n < 2; n++, ql += 64, qh += 32) // as ggml's dequantize_row_q6_K
                    for (int l = 0; l < 32; l++) {
                        code[n * 128 + l] = (ql[l] & 0xF) | ((qh[l] >> 0) & 3) << 4;
                        code[n * 128 + l + 32] = (ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4;
                        code[n * 128 + l + 64] = (ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4;
                        code[n * 128 + l + 96] = (ql[l + 32] >> 4) | ((qh[l] >> 6) & 3) << 4;
                    }
                memcpy(super_at(m, r, sb * 8), p + 208, 2);
                for (int k = 0; k < 8; k++) {
                    put_codes6(m, bi(G, r, sb * 8 + k), code + k * 32);
                    sub_at(m, r, sb * 8 + k)[0] = (uint8_t)sc[2 * k];
                    sub_at(m, r, sb * 8 + k)[4] = (uint8_t)sc[2 * k + 1];
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
    free(m->sub);
    free(m->super);
    free(m);
}

size_t cq_bytes(const cq_mat *m) {
    size_t nb = (size_t)((m->rows + 3) & ~3) * (m->cols / CQ_BLOCK);
    size_t per = code_bytes(m->kind) + 2 + (m->kind == CQ4 ? 2 : 0);
    return nb * per + (m->super ? nb / 8 * 4 : 0);
}

// Block g of row r as int8 codes q with value = (j < 16 ? lo : hi) * q[j] + bias.
static void block_ref(const cq_mat *m, int r, int g, int8_t *q, float *lo, float *hi, float *bias) {
    int G = m->cols / CQ_BLOCK;
    size_t k = bi(G, r, g);
    const uint8_t *c = m->codes + k * code_bytes(m->kind);
    *bias = 0;
    if (m->kind == CQ8) {
        memcpy(q, c, 32);
        *lo = *hi = fp16_to_f32(m->scale[k]);
        return;
    }
    for (int j = 0; j < 16; j++) { q[j] = c[j] & 15; q[j + 16] = c[j] >> 4; }
    if (m->kind == CQ6K)
        for (int j = 0; j < 32; j++) q[j] = (int8_t)((q[j] | ((c[16 + j % 8] >> (2 * (j / 8))) & 3) << 4) - 32);
    if (m->kind == CQ4S)
        for (int j = 0; j < 32; j++) q[j] -= 8;
    if (m->kind == CQ4K || m->kind == CQ6K) {
        const uint8_t *s = sub_at(m, r, g);
        const uint16_t *d = super_at(m, r, g);
        if (m->kind == CQ4K) {
            *lo = *hi = fp16_to_f32(d[0]) * s[0];
            *bias = -fp16_to_f32(d[4]) * s[4];
        } else {
            *lo = fp16_to_f32(d[0]) * (int8_t)s[0];
            *hi = fp16_to_f32(d[0]) * (int8_t)s[4];
        }
        return;
    }
    *lo = *hi = fp16_to_f32(m->scale[k]);
    if (m->kind == CQ4) *bias = fp16_to_f32(m->bias[k]);
}

float cq_check(const qmat *w, const cq_mat *m, int rows) {
    float *ref = malloc(sizeof(float) * w->cols), err = 0, mx = 0, lo, hi, bias;
    int8_t q[CQ_BLOCK];
    for (int r = 0; r < rows && r < w->rows; r++) {
        qmat_row(w, r, ref);
        for (int g = 0; g < w->cols / CQ_BLOCK; g++) {
            block_ref(m, r, g, q, &lo, &hi, &bias);
            for (int j = 0; j < CQ_BLOCK; j++) {
                float v = (j < 16 ? lo : hi) * q[j] + bias, x = ref[g * CQ_BLOCK + j];
                err = fmaxf(err, fabsf(v - x));
                mx = fmaxf(mx, fabsf(x));
            }
        }
    }
    free(ref);
    return mx > 0 ? err / mx : err;
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

// 4 rows' dot products of one block, reduced to one vector [row0..row3]
static inline int32x4_t dot4(const int8x16_t *w, int8x16_t x) {
    const int32x4_t z = vdupq_n_s32(0);
    return vpaddq_s32(vpaddq_s32(vdotq_s32(z, w[0], x), vdotq_s32(z, w[1], x)), vpaddq_s32(vdotq_s32(z, w[2], x), vdotq_s32(z, w[3], x)));
}

static inline __attribute__((always_inline)) void mm_kind(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1, const int kind) {
    const int G = w->cols / CQ_BLOCK, K = w->cols, cb = code_bytes(kind);
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3), m30 = vdupq_n_u8(0x30);
    const int8x16_t off = vdupq_n_s8(kind == CQ4S ? 8 : 32);
    for (int rg = r0 >> 2; rg * 4 < r1; rg++) {
        const size_t base = (size_t)rg * G * 4; // first block of this row group
        for (int t0 = 0; t0 < a->n; t0 += 4) {
            const int nt = a->n - t0 < 4 ? a->n - t0 : 4;
            float32x4_t acc[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) }; // lanes: the 4 weight rows
            float32x4_t d = vdupq_n_f32(0), dmin = d;
            for (int g = 0; g < G; g++) {
                const size_t b = base + (size_t)g * 4;
                int8x16_t lo[4], hi[4];
                for (int k = 0; k < 4; k++) {
                    const uint8_t *c = w->codes + (b + k) * cb;
                    if (kind == CQ8) {
                        lo[k] = vld1q_s8((const int8_t *)c);
                        hi[k] = vld1q_s8((const int8_t *)c + 16);
                        continue;
                    }
                    uint8x16_t v = vld1q_u8(c), l = vandq_u8(v, m4), h = vshrq_n_u8(v, 4);
                    if (kind == CQ6K) {
                        uint8x8_t h2 = vld1_u8(c + 16);
                        uint8x16_t hv = vcombine_u8(h2, vshr_n_u8(h2, 2));
                        l = vorrq_u8(l, vshlq_n_u8(vandq_u8(hv, m3), 4));
                        h = vorrq_u8(h, vandq_u8(hv, m30));
                    }
                    lo[k] = vreinterpretq_s8_u8(l);
                    hi[k] = vreinterpretq_s8_u8(h);
                    if (kind == CQ4S || kind == CQ6K) { lo[k] = vsubq_s8(lo[k], off); hi[k] = vsubq_s8(hi[k], off); }
                }
                float32x4_t ws, wsh = vdupq_n_f32(0), wb = vdupq_n_f32(0);
                if (kind == CQ4K || kind == CQ6K) {
                    if (!(g & 7)) {
                        const uint16_t *sp = w->super + ((size_t)rg * (G / 8) + g / 8) * 8;
                        d = f16x4(sp);
                        dmin = f16x4(sp + 4);
                    }
                    const uint8_t *sp = w->sub + b * 2;
                    if (kind == CQ4K) {
                        uint16x8_t s = vmovl_u8(vld1_u8(sp));
                        ws = vmulq_f32(d, vcvtq_f32_u32(vmovl_u16(vget_low_u16(s))));
                        wb = vnegq_f32(vmulq_f32(dmin, vcvtq_f32_u32(vmovl_u16(vget_high_u16(s)))));
                    } else {
                        int16x8_t s = vmovl_s8(vld1_s8((const int8_t *)sp));
                        ws = vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))));
                        wsh = vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))));
                    }
                } else {
                    ws = f16x4(w->scale + b);
                    if (kind == CQ4) wb = f16x4(w->bias + b);
                }
                for (int i = 0; i < nt; i++) {
                    const size_t xb = (size_t)(t0 + i) * G + g;
                    const int8_t *xp = a->q + (size_t)(t0 + i) * K + (size_t)g * CQ_BLOCK;
                    const int8x16_t xl = vld1q_s8(xp), xh = vld1q_s8(xp + 16);
                    if (kind == CQ6K) { // the two halves have their own scales
                        float32x4_t f = vfmaq_f32(vmulq_f32(vcvtq_f32_s32(dot4(lo, xl)), ws), vcvtq_f32_s32(dot4(hi, xh)), wsh);
                        acc[i] = vfmaq_n_f32(acc[i], f, a->s[xb]);
                        continue;
                    }
                    const int32x4_t z = vdupq_n_s32(0);
                    int32x4_t dk[4];
                    for (int k = 0; k < 4; k++) dk[k] = vdotq_s32(vdotq_s32(z, lo[k], xl), hi[k], xh);
                    const int32x4_t dot = vpaddq_s32(vpaddq_s32(dk[0], dk[1]), vpaddq_s32(dk[2], dk[3])); // [row0..row3]
                    acc[i] = vfmaq_f32(acc[i], vcvtq_f32_s32(dot), vmulq_n_f32(ws, a->s[xb]));
                    if (kind == CQ4 || kind == CQ4K) acc[i] = vfmaq_n_f32(acc[i], wb, a->sq[xb]);
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

void cq_matmul_rows(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1) {
    switch (w->kind) { // one specialized copy per format
    case CQ4: mm_kind(w, a, y, ldy, r0, r1, CQ4); break;
    case CQ4S: mm_kind(w, a, y, ldy, r0, r1, CQ4S); break;
    case CQ4K: mm_kind(w, a, y, ldy, r0, r1, CQ4K); break;
    case CQ6K: mm_kind(w, a, y, ldy, r0, r1, CQ6K); break;
    default: mm_kind(w, a, y, ldy, r0, r1, CQ8); break;
    }
}
#else
const char *cq_isa(void) { return "scalar"; }

void cq_matmul_rows(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1) {
    const int G = w->cols / CQ_BLOCK, K = w->cols;
    int8_t q[CQ_BLOCK];
    float lo, hi, bias;
    for (int r = r0; r < r1; r++)
        for (int t = 0; t < a->n; t++) {
            float acc = 0;
            for (int g = 0; g < G; g++) {
                block_ref(w, r, g, q, &lo, &hi, &bias);
                const int8_t *xp = a->q + (size_t)t * K + (size_t)g * CQ_BLOCK;
                int dl = 0, dh = 0;
                for (int j = 0; j < 16; j++) { dl += q[j] * xp[j]; dh += q[j + 16] * xp[j + 16]; }
                size_t xb = (size_t)t * G + g;
                acc += a->s[xb] * (lo * dl + hi * dh) + bias * a->sq[xb];
            }
            y[(size_t)t * ldy + r] = acc;
        }
}
#endif
