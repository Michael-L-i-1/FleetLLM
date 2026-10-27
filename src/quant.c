// Scalar dequantization. Block layouts and decode formulas are ported from
// ggml (ggml-quants.c, dequantize_row_*), MIT License,
// Copyright (c) 2023-2026 The ggml authors.
#include "quant.h"
#include "gguf.h"
#include <string.h>

float fp16_to_f32(uint16_t h) {
    _Float16 v;
    memcpy(&v, &h, 2);
    return (float)v;
}

uint16_t f32_to_fp16(float f) {
    _Float16 v = (_Float16)f;
    uint16_t h;
    memcpy(&h, &v, 2);
    return h;
}

static float h16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return fp16_to_f32(v); }

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

static void deq_q2_k(const uint8_t *b, float *y) {
    float d = h16(b + 80), dmin = h16(b + 82);
    const uint8_t *sc = b, *q = b + 16;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t s = sc[is++];
            float dl = d * (s & 0xF), ml = dmin * (s >> 4);
            for (int l = 0; l < 16; l++) *y++ = dl * ((q[l] >> shift) & 3) - ml;
            s = sc[is++];
            dl = d * (s & 0xF); ml = dmin * (s >> 4);
            for (int l = 0; l < 16; l++) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
            shift += 2;
        }
        q += 32;
    }
}

static void deq_q3_k(const uint8_t *b, float *y) {
    const uint32_t k1 = 0x03030303, k2 = 0x0f0f0f0f;
    const uint8_t *hm = b, *q = b + 32;
    float d_all = h16(b + 108);
    uint32_t aux[4];
    memcpy(aux, b + 96, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    const int8_t *scales = (const int8_t *)aux;
    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

static void deq_q4_k(const uint8_t *b, float *y) {
    float d = h16(b), min = h16(b + 2);
    const uint8_t *sc = b + 4, *q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        get_scale_min_k4(is, sc, &s, &m);
        float d1 = d * s, m1 = min * m;
        get_scale_min_k4(is + 1, sc, &s, &m);
        float d2 = d * s, m2 = min * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

static void deq_q5_k(const uint8_t *b, float *y) {
    float d = h16(b), min = h16(b + 2);
    const uint8_t *sc = b + 4, *qh = b + 16, *ql = b + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        get_scale_min_k4(is, sc, &s, &m);
        float d1 = d * s, m1 = min * m;
        get_scale_min_k4(is + 1, sc, &s, &m);
        float d2 = d * s, m2 = min * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

static void deq_q6_k(const uint8_t *b, float *y) {
    float d = h16(b + 208);
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static void deq_q8_0(const uint8_t *b, float *y) {
    float d = h16(b);
    const int8_t *q = (const int8_t *)(b + 2);
    for (int j = 0; j < 32; j++) y[j] = q[j] * d;
}

static void deq_q4_0(const uint8_t *b, float *y) {
    float d = h16(b);
    const uint8_t *q = b + 2;
    for (int j = 0; j < 16; j++) {
        y[j] = ((q[j] & 0xF) - 8) * d;
        y[j + 16] = ((q[j] >> 4) - 8) * d;
    }
}

static void deq_q4_1(const uint8_t *b, float *y) {
    float d = h16(b), m = h16(b + 2);
    const uint8_t *q = b + 4;
    for (int j = 0; j < 16; j++) {
        y[j] = (q[j] & 0xF) * d + m;
        y[j + 16] = (q[j] >> 4) * d + m;
    }
}

static void deq_q5_0(const uint8_t *b, float *y) {
    float d = h16(b);
    uint32_t qh;
    memcpy(&qh, b + 2, 4);
    const uint8_t *q = b + 6;
    for (int j = 0; j < 16; j++) {
        uint8_t xh0 = ((qh >> j) << 4) & 0x10, xh1 = (qh >> (j + 12)) & 0x10;
        y[j] = (((q[j] & 0xF) | xh0) - 16) * d;
        y[j + 16] = (((q[j] >> 4) | xh1) - 16) * d;
    }
}

static void deq_q5_1(const uint8_t *b, float *y) {
    float d = h16(b), m = h16(b + 2);
    uint32_t qh;
    memcpy(&qh, b + 4, 4);
    const uint8_t *q = b + 8;
    for (int j = 0; j < 16; j++) {
        uint8_t xh0 = ((qh >> j) << 4) & 0x10, xh1 = (qh >> (j + 12)) & 0x10;
        y[j] = ((q[j] & 0xF) | xh0) * d + m;
        y[j + 16] = ((q[j] >> 4) | xh1) * d + m;
    }
}

bool dequant_supported(uint32_t type) {
    switch (type) {
    case GGML_F32: case GGML_F16: case GGML_BF16: case GGML_Q4_0: case GGML_Q4_1: case GGML_Q5_0:
    case GGML_Q5_1: case GGML_Q8_0: case GGML_Q2_K: case GGML_Q3_K: case GGML_Q4_K: case GGML_Q5_K:
    case GGML_Q6_K: return true;
    default: return false;
    }
}

bool dequant_row(uint32_t type, const void *src, float *dst, size_t n) {
    const uint8_t *b = src;
    size_t be, bb;
    if (!ggml_block(type, &be, &bb) || n % be) return false;
    switch (type) {
    case GGML_F32: memcpy(dst, src, n * 4); return true;
    case GGML_F16: for (size_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, b + 2 * i, 2); dst[i] = fp16_to_f32(h); } return true;
    case GGML_BF16: for (size_t i = 0; i < n; i++) { uint32_t u = 0; memcpy((uint8_t *)&u + 2, b + 2 * i, 2); memcpy(&dst[i], &u, 4); } return true;
    default: break;
    }
    void (*fn)(const uint8_t *, float *) = NULL;
    switch (type) {
    case GGML_Q2_K: fn = deq_q2_k; break;
    case GGML_Q3_K: fn = deq_q3_k; break;
    case GGML_Q4_K: fn = deq_q4_k; break;
    case GGML_Q5_K: fn = deq_q5_k; break;
    case GGML_Q6_K: fn = deq_q6_k; break;
    case GGML_Q8_0: fn = deq_q8_0; break;
    case GGML_Q4_0: fn = deq_q4_0; break;
    case GGML_Q4_1: fn = deq_q4_1; break;
    case GGML_Q5_0: fn = deq_q5_0; break;
    case GGML_Q5_1: fn = deq_q5_1; break;
    default: return false;
    }
    for (size_t i = 0; i < n / be; i++) fn(b + i * bb, dst + i * be);
    return true;
}
