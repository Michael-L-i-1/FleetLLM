// All formats, partial row groups, activation tile tails, and the portable
// fallback without model downloads. Run scripts/cpuqcheck.sh.
#include "cpuq.h"
#include "quant.h"
#include "gguf.h"
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void scalar_matmul(const cq_mat *, const cq_act *, float *, int, int, int);

void scalar_cq_quantize(cq_act *, const float *, int, int);
void scalar_cq_quantize_group(cq_act *, const float *, int, int, int);

static int check_quant(void) {
    enum { K = 768, N = 17 };
    float x[N * K];
    cq_act a = {0}, ref = {0};
    const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    int saved = fegetround();
    for (int group = 32; group <= 256; group *= 8) {
    for (int mode = 0; mode < 4; mode++) {
        fesetround(modes[mode]);
        for (int rep = 0; rep < 16; rep++) {
            for (int i = 0; i < N * K; i++) x[i] = rep == 0 ? 0 : ldexpf((rand() % 65537 - 32768) / 32768.f, rep * 8 - 64);
            // amax 127 makes scale one: half-integers exercise rounding ties.
            for (int i = 0; i < 32; i++) x[i] = i == 31 ? 127.f : (i - 15) * 0.5f;
            cq_quantize_group(&a, x, N, K, group); scalar_cq_quantize_group(&ref, x, N, K, group);
            if (memcmp(a.q, ref.q, N * K) || memcmp(a.s, ref.s, sizeof(float) * N * K / 32) ||
                memcmp(a.sq, ref.sq, sizeof(float) * N * K / 32) ||
                memcmp(a.s16, ref.s16, sizeof(float) * N * K / 16) ||
                memcmp(a.is16, ref.is16, sizeof(int16_t) * N * K / 16) || memcmp(a.nz, ref.nz, N * K / 32)) {
                fprintf(stderr, "quantization mismatch mode %d rep %d\n", mode, rep); return 1;
            }
        }
    }
    }
    fesetround(saved);
    free(a.q); free(a.s); free(a.sq); free(a.s16); free(a.is16); free(a.nz);
    free(ref.q); free(ref.s); free(ref.sq); free(ref.s16); free(ref.is16); free(ref.nz);
    puts("quantization: scalar bitwise, four rounding modes, zero/ties/wide exponents PASS");
    return 0;
}

static int check_q2_repack(void) {
    enum { R = 7, K = 768, N = 9, RB = K / 256 * 84 };
    uint8_t packed[R * RB];
    float x[N * K], y[N * R], row[K];
    for (int i = 0; i < R * RB; i++) packed[i] = (uint8_t)(rand() % 256);
    for (int b = 0; b < R * K / 256; b++) {
        uint16_t d = f32_to_fp16((b % 7) * .013f), mn = f32_to_fp16((b % 5) * .007f);
        memcpy(packed + b * 84 + 80, &d, 2);
        memcpy(packed + b * 84 + 82, &mn, 2);
    }
    qmat src = { .type = GGML_Q2_K, .rows = R, .cols = K, .row_bytes = RB, .data = packed };
    cq_mat *w = cq_from_qmat(&src);
    if (!w || w->kind != CQ2K || cq_bytes(w) != (size_t)8 * RB || cq_check(&src, w, R) != 0) {
        fputs("Q2_K repack/size mismatch\n", stderr); return 1;
    }
    for (int i = 0; i < N * K; i++) x[i] = sinf(i * .19f);
    cq_act a = {0}; cq_quantize(&a, x, N, K);
    cq_matmul_rows(w, &a, y, R, 0, R);
    for (int r = 0; r < R; r++) {
        qmat_row(&src, r, row);
        for (int t = 0; t < N; t++) {
            double sum = 0;
            for (int j = 0; j < K; j++) sum += (double)row[j] * a.q[t * K + j] * a.s[(t * K + j) / 32];
            if (fabs(y[t * R + r] - sum) > 1e-4 * (1 + fabs(sum))) {
                fprintf(stderr, "Q2_K dense mismatch row %d token %d\n", r, t); return 1;
            }
        }
    }
    cq_free(w); free(a.q); free(a.s); free(a.sq); free(a.s16); free(a.is16); free(a.nz);
    puts("Q2_K: exact repack, original packed size, independent dense dequantized matmul PASS");
    return 0;
}

static int check_q3_repack(void) {
    enum { R = 7, K = 768, N = 9, RB = K / 256 * 110 };
    uint8_t packed[R * RB];
    float x[N * K], y[N * R], row[K];
    for (int i = 0; i < R * RB; i++) packed[i] = (uint8_t)(rand() % 256);
    for (int b = 0; b < R * K / 256; b++) {
        uint16_t d = f32_to_fp16((b % 7) * .013f);
        memcpy(packed + b * 110 + 108, &d, 2);

    }
    qmat src = { .type = GGML_Q3_K, .rows = R, .cols = K, .row_bytes = RB, .data = packed };
    cq_mat *w = cq_from_qmat(&src);
    if (!w || w->kind != CQ3K || cq_bytes(w) != (size_t)8 * (K / 256) * 116 || cq_check(&src, w, R) != 0) {
        fputs("Q3_K repack/size mismatch\n", stderr); return 1;
    }
    for (int i = 0; i < N * K; i++) x[i] = sinf(i * .19f);
    cq_act a = {0}; cq_quantize(&a, x, N, K);
    cq_matmul_rows(w, &a, y, R, 0, R);
    for (int r = 0; r < R; r++) {
        qmat_row(&src, r, row);
        for (int t = 0; t < N; t++) {
            double sum = 0;
            for (int j = 0; j < K; j++) sum += (double)row[j] * a.q[t * K + j] * a.s[(t * K + j) / 32];
            if (fabs(y[t * R + r] - sum) > 1e-4 * (1 + fabs(sum))) {
                fprintf(stderr, "Q3_K dense mismatch row %d token %d\n", r, t); return 1;
            }
        }
    }
    cq_free(w); free(a.q); free(a.s); free(a.sq); free(a.s16); free(a.is16); free(a.nz);
    puts("Q3_K: exact repack, compact 3-bit storage, independent dense dequantized matmul PASS");
    return 0;
}

int main(void) {
    if (check_quant() || check_q2_repack() || check_q3_repack()) return 1;
    enum { R = 13, K = 768, N = 17 };
    float x[N * K], y[N * R], ref[N * R], seq[R], scalar[N * R];
    for (int i = 0; i < N * K; i++) x[i] = sinf(i * 0.17f) * 0.7f;
    cq_act a = { 0 };
    for (int group = 32; group <= 256; group *= 8) {
    for (int sparse = 0; sparse <= 1; sparse++) {
    for (int i = 0; i < N * K; i++) x[i] = sparse && ((i / 32) % 3) ? 0 : sinf(i * .17f) * .7f;
    cq_quantize_group(&a, x, N, K, group);
    a.sparse = sparse;
    for (int kind = CQ4; kind <= CQ8; kind++) {
        int G = K / 32, nb = 16 * G, bytes = kind == CQ8 ? 32 : kind == CQ6K ? 24 : kind == CQ2K ? 8 : kind == CQ3K ? 12 : 16;
        cq_mat w = { .kind = kind, .rows = R, .cols = K };
        w.codes = malloc(nb * bytes); w.scale = malloc(nb * 2); w.bias = malloc(nb * 2);
        w.sub = malloc(nb * 2); w.super = malloc(nb / 8 * 4);
        for (int i = 0; i < nb * bytes; i++) w.codes[i] = (i * 73 + i / 7) % 256;
        for (int i = 0; i < nb; i++) {
            w.scale[i] = f32_to_fp16(0.001f * (1 + i % 7));
            w.bias[i] = f32_to_fp16(-0.003f * (1 + i % 5));
        }
        for (int i = 0; i < nb * 2; i++) w.sub[i] = (kind == CQ6K || kind == CQ3K) ? (uint8_t)(i % 255 - 127) : i % 64;
        for (int i = 0; i < nb / 8 * 2; i++) w.super[i] = f32_to_fp16(0.0001f * (1 + i % 11));
        for (int n = 1; n <= N; n++) {
            a.n = n;
            cq_matmul_rows(&w, &a, y, R, 0, R);
            scalar_matmul(&w, &a, scalar, R, 0, R);
            for (int r = 0; r < R; r += 4) cq_matmul_rows(&w, &a, ref, R, r, r + 4 < R ? r + 4 : R);
            if (memcmp(y, ref, n * R * sizeof(float))) { fprintf(stderr, "partition mismatch kind %d n %d\n", kind, n); return 1; }
            for (int i = 0; i < n * R; i++)
                if (!isfinite(y[i]) || fabsf(y[i] - scalar[i]) > 0.0001f * (1 + fabsf(y[i]))) {
                    fprintf(stderr, "scalar mismatch kind %d n %d index %d\n", kind, n, i); return 1;
                }
            for (int t = 0; t < n; t++) {
                cq_act one = a;
                one.n = 1; one.q += t * K; one.s += t * G; one.sq += t * G; one.s16 += t * G * 2; one.is16 += t * G * 2; one.nz += t * G;
                cq_matmul_rows(&w, &one, seq, R, 0, R);
                if (memcmp(seq, y + t * R, R * sizeof(float))) { fprintf(stderr, "batch mismatch kind %d n %d row %d\n", kind, n, t); return 1; }
            }
        }
        free(w.codes); free(w.scale); free(w.bias); free(w.sub); free(w.super);
    }
    }
    }
    free(a.q); free(a.s); free(a.sq); free(a.s16); free(a.is16); free(a.nz);
    puts("all seven dense/sparse formats: row partitions, batch sizes 1..17, scalar tolerance, padded rows PASS");
}
