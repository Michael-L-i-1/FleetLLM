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

// Code bytes of a block's 4 rows are interleaved 4 bytes at a time: 16-byte
// vector v holds bytes 4v..4v+3 of rows 0, 1, 2, 3, so one dot-product-by-lane
// instruction accumulates all 4 rows into separate lanes (the layout of ggml's
// q4_K_8x4 repack, MIT, github.com/ggml-org/llama.cpp).
// Byte j of the packed codes of row r, block g (CQ6K: of its low-nibble part).
static inline uint8_t *code_at(const cq_mat *m, int r, int g, int j) {
    return m->codes + ((size_t)(r >> 2) * (m->cols / CQ_BLOCK) + g) * 4 * code_bytes(m->kind) + (j >> 2) * 16 + (r & 3) * 4 + (j & 3);
}

// 4-bit codes (0..15) of one block: byte j = code[j] | code[j + 16] << 4
static void put_codes4(cq_mat *m, int r, int g, const uint8_t *code) {
    for (int j = 0; j < 16; j++) *code_at(m, r, g, j) = (uint8_t)(code[j] | code[j + 16] << 4);
}

// 6-bit codes (0..63) of one block: low 4 bits as put_codes4, then the high 2
// bits as 2 vectors (64 bytes into the block group): vector p, row r, byte i
// holds, for v = 2p + k (k = 0, 1), the bits of code[4v + i] at 2k and of
// code[16 + 4v + i] at 4 + 2k
static void put_codes6(cq_mat *m, int r, int g, const uint8_t *code) {
    for (int j = 0; j < 16; j++) *code_at(m, r, g, j) = (uint8_t)((code[j] & 15) | (code[j + 16] & 15) << 4);
    uint8_t *h = code_at(m, r, g, 0) - (r & 3) * 4 + 64 + (r & 3) * 4;
    for (int p = 0; p < 2; p++)
        for (int i = 0; i < 4; i++) {
            uint8_t b = 0;
            for (int k = 0; k < 2; k++) b |= (uint8_t)((code[4 * (2 * p + k) + i] >> 4) << (2 * k) | (code[16 + 4 * (2 * p + k) + i] >> 4) << (4 + 2 * k));
            h[p * 16 + i] = b;
        }
}

// one CQ8 block of int8 codes
static void put_codes8(cq_mat *m, int r, int g, const int8_t *code) {
    for (int j = 0; j < 32; j++) *code_at(m, r, g, j) = (uint8_t)code[j];
}

// one CQ8 block quantized from 32 floats (symmetric, scale = max|x| / 127)
static void put8f(cq_mat *m, int r, int g, const float *x) {
    float amax = 0;
    for (int j = 0; j < CQ_BLOCK; j++) amax = fmaxf(amax, fabsf(x[j]));
    float s = amax / 127.f, inv = s > 0 ? 1.f / s : 0;
    int8_t c[CQ_BLOCK];
    for (int j = 0; j < CQ_BLOCK; j++) c[j] = (int8_t)lrintf(x[j] * inv);
    put_codes8(m, r, g, c);
    m->scale[bi(m->cols / CQ_BLOCK, r, g)] = f32_to_fp16(s);
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
                put_codes4(m, r, b, code);
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
                for (int j = 0; j < 16; j++) *code_at(m, r, b, j) = p[off + j]; // same nibble order
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
                    put_codes4(m, r, sb * 8 + k, code);
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
                    put_codes6(m, r, sb * 8 + k, code + k * 32);
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
                put_codes8(m, r, b, (const int8_t *)p + 2);
                memcpy(&m->scale[bi(G, r, b)], p, 2);
            }
        return m;
    }
    float *row = malloc(sizeof(float) * w->cols); // anything else: dequantize, requantize to 8-bit
    for (int r = 0; r < w->rows; r++) {
        qmat_row(w, r, row);
        for (int b = 0; b < G; b++) put8f(m, r, b, row + b * CQ_BLOCK);
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
    *bias = 0;
    if (m->kind == CQ8) {
        for (int j = 0; j < 32; j++) q[j] = (int8_t)*code_at(m, r, g, j);
        *lo = *hi = fp16_to_f32(m->scale[k]);
        return;
    }
    for (int j = 0; j < 16; j++) { uint8_t c = *code_at(m, r, g, j); q[j] = c & 15; q[j + 16] = c >> 4; }
    if (m->kind == CQ6K) {
        const uint8_t *h = code_at(m, r, g, 0) + 64;
        for (int j = 0; j < 16; j++) {
            int v = j / 4, b = h[(v / 2) * 16 + (j & 3)] >> (2 * (v & 1));
            q[j] = (int8_t)((q[j] | (b & 3) << 4) - 32);
            q[j + 16] = (int8_t)((q[j + 16] | ((b >> 4) & 3) << 4) - 32);
        }
    }
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
#define CQ_TILE 8 // activation rows per pass over a weight row group

// Per group of 4 weight rows and tile of up to CQ_TILE activation rows: each weight
// block is unpacked once and dotted against every activation row of the tile.

#if defined(__ARM_FEATURE_DOTPROD)
const char *cq_isa(void) { return "neon+dotprod"; }

static inline float32x4_t f16x4(const uint16_t *p) { return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p))); }

// Lane-wise dot of 4 interleaved code vectors (lane r: row r) with 16
// activations: vector v pairs with activation bytes 4v..4v+3 (lane v of x).
static inline int32x4_t dot4l(int32x4_t acc, const int8x16_t *w, int8x16_t x) {
    acc = vdotq_laneq_s32(acc, w[0], x, 0);
    acc = vdotq_laneq_s32(acc, w[1], x, 1);
    acc = vdotq_laneq_s32(acc, w[2], x, 2);
    return vdotq_laneq_s32(acc, w[3], x, 3);
}

// Unpack block g of row group rg: codes into lo / hi (lane r of vector v: row r,
// values 4v..4v+3 / 16+4v.. of the block; CQ8: vectors 0-3 / 4-7), per-row
// scales of the two halves (ws, wsh; equal except CQ6K) and bias wb.
// d / dmin: the CQ4K / CQ6K super-block scales, reloaded every 8 blocks.
static inline __attribute__((always_inline)) void unpack(const cq_mat *w, int rg, int g, const int kind, int8x16_t *lo, int8x16_t *hi,
                                                         float32x4_t *ws, float32x4_t *wsh, float32x4_t *wb, float32x4_t *d, float32x4_t *dmin) {
    const int G = w->cols / CQ_BLOCK;
    const size_t b = ((size_t)rg * G + g) * 4;
    const uint8_t *c = w->codes + b * code_bytes(kind);
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3), m30 = vdupq_n_u8(0x30);
    const int8x16_t off = vdupq_n_s8(kind == CQ4S ? 8 : 32);
    for (int v = 0; v < 4; v++) {
        if (kind == CQ8) {
            lo[v] = vld1q_s8((const int8_t *)c + 16 * v);
            hi[v] = vld1q_s8((const int8_t *)c + 64 + 16 * v);
            continue;
        }
        uint8x16_t x = vld1q_u8(c + 16 * v), l = vandq_u8(x, m4), h = vshrq_n_u8(x, 4);
        if (kind == CQ6K) {
            uint8x16_t hb = vld1q_u8(c + 64 + 16 * (v >> 1));
            if (v & 1) hb = vshrq_n_u8(hb, 2);
            l = vorrq_u8(l, vshlq_n_u8(vandq_u8(hb, m3), 4));
            h = vorrq_u8(h, vandq_u8(hb, m30));
        }
        lo[v] = vreinterpretq_s8_u8(l);
        hi[v] = vreinterpretq_s8_u8(h);
        if (kind == CQ4S || kind == CQ6K) { lo[v] = vsubq_s8(lo[v], off); hi[v] = vsubq_s8(hi[v], off); }
    }
    if (kind == CQ4K || kind == CQ6K) {
        if (!(g & 7)) {
            const uint16_t *sp = w->super + ((size_t)rg * (G / 8) + g / 8) * 8;
            *d = f16x4(sp);
            *dmin = f16x4(sp + 4);
        }
        const uint8_t *sp = w->sub + b * 2;
        if (kind == CQ4K) {
            uint16x8_t s = vmovl_u8(vld1_u8(sp));
            *ws = *wsh = vmulq_f32(*d, vcvtq_f32_u32(vmovl_u16(vget_low_u16(s))));
            *wb = vnegq_f32(vmulq_f32(*dmin, vcvtq_f32_u32(vmovl_u16(vget_high_u16(s)))));
        } else {
            int16x8_t s = vmovl_s8(vld1_s8((const int8_t *)sp));
            *ws = vmulq_f32(*d, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))));
            *wsh = vmulq_f32(*d, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))));
        }
    } else {
        *ws = *wsh = f16x4(w->scale + b);
        if (kind == CQ4) *wb = f16x4(w->bias + b);
    }
}

// one block of one activation row: acc += its 4 rows' dot products, bacc += bias term
// (every row count must sum in the same order: batched outputs equal sequential ones)
static inline __attribute__((always_inline)) void block_dot(const int kind, const int8x16_t *lo, const int8x16_t *hi, float32x4_t ws, float32x4_t wsh,
                                                            float32x4_t wb, const int8_t *xp, float s, float sq, float32x4_t *acc, float32x4_t *bacc) {
    const int32x4_t z = vdupq_n_s32(0);
    const int8x16_t xl = vld1q_s8(xp), xh = vld1q_s8(xp + 16);
    const int32x4_t dl = dot4l(z, lo, xl), dh = dot4l(z, hi, xh); // [row0..row3]
    if (kind == CQ6K) // the two halves have their own scales
        *acc = vfmaq_n_f32(*acc, vfmaq_f32(vmulq_f32(vcvtq_f32_s32(dl), ws), vcvtq_f32_s32(dh), wsh), s);
    else
        *acc = vfmaq_f32(*acc, vcvtq_f32_s32(vaddq_s32(dl, dh)), vmulq_n_f32(ws, s));
    if (kind == CQ4 || kind == CQ4K) *bacc = vfmaq_n_f32(*bacc, wb, sq);
}

static inline __attribute__((always_inline)) void mm_kind(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1, const int kind, const int tile, int t_begin, int t_end) {
    const int G = w->cols / CQ_BLOCK;
    const float32x4_t zf = vdupq_n_f32(0);
    for (int rg = r0 >> 2; rg * 4 < r1; rg++) {
        float32x4_t d = zf, dmin = zf, ws, wsh, wb = zf;
        int8x16_t lo[4], hi[4];
        for (int t0 = t_begin; t0 < t_end; t0 += tile ? tile : CQ_TILE) {
            const int nt = tile ? tile : t_end - t0;
            float32x4_t acc[CQ_TILE]; // lanes: the 4 weight rows
            for (int i = 0; i < CQ_TILE; i++) acc[i] = zf;
            for (int g = 0; g < G; g++) {
                unpack(w, rg, g, kind, lo, hi, &ws, &wsh, &wb, &d, &dmin);
                for (int i = 0; i < nt; i++) {
                    const size_t xb = (size_t)(t0 + i) * G + g;
                    block_dot(kind, lo, hi, ws, wsh, wb, a->q + xb * CQ_BLOCK, a->s[xb], a->sq[xb], &acc[i], &acc[i]);
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

// Full tiles have a constant trip count so accumulators stay in registers.
// The tail shares the same block_dot and therefore the same summation order.
static inline __attribute__((always_inline)) void mm_dispatch(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1, const int kind) {
    if (a->n == 1) mm_kind(w, a, y, ldy, r0, r1, kind, 1, 0, 1);
    else {
        int full = a->n / CQ_TILE * CQ_TILE;
        if (full) mm_kind(w, a, y, ldy, r0, r1, kind, CQ_TILE, 0, full);
        if (full < a->n) mm_kind(w, a, y, ldy, r0, r1, kind, 0, full, a->n);
    }
}

void cq_matmul_rows(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1) {
    switch (w->kind) {
    case CQ4: mm_dispatch(w, a, y, ldy, r0, r1, CQ4); break;
    case CQ4S: mm_dispatch(w, a, y, ldy, r0, r1, CQ4S); break;
    case CQ4K: mm_dispatch(w, a, y, ldy, r0, r1, CQ4K); break;
    case CQ6K: mm_dispatch(w, a, y, ldy, r0, r1, CQ6K); break;
    default: mm_dispatch(w, a, y, ldy, r0, r1, CQ8); break;
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
