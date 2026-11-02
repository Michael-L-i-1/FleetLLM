// All formats, partial row groups, activation tile tails, and the portable
// fallback without model downloads. Run scripts/cpuqcheck.sh.
#include "cpuq.h"
#include "quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void scalar_matmul(const cq_mat *, const cq_act *, float *, int, int, int);

int main(void) {
    enum { R = 13, K = 768, N = 17 };
    float x[N * K], y[N * R], ref[N * R], seq[R], scalar[N * R];
    for (int i = 0; i < N * K; i++) x[i] = sinf(i * 0.17f) * 0.7f;
    cq_act a = { 0 };
    cq_quantize(&a, x, N, K);
    for (int kind = CQ4; kind <= CQ8; kind++) {
        int G = K / 32, nb = 16 * G, bytes = kind == CQ8 ? 32 : kind == CQ6K ? 24 : 16;
        cq_mat w = { .kind = kind, .rows = R, .cols = K };
        w.codes = malloc(nb * bytes); w.scale = malloc(nb * 2); w.bias = malloc(nb * 2);
        w.sub = malloc(nb * 2); w.super = malloc(nb / 8 * 4);
        for (int i = 0; i < nb * bytes; i++) w.codes[i] = (i * 73 + i / 7) % 256;
        for (int i = 0; i < nb; i++) {
            w.scale[i] = f32_to_fp16(0.001f * (1 + i % 7));
            w.bias[i] = f32_to_fp16(-0.003f * (1 + i % 5));
        }
        for (int i = 0; i < nb * 2; i++) w.sub[i] = kind == CQ6K ? (uint8_t)(i % 255 - 127) : i % 64;
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
                one.n = 1; one.q += t * K; one.s += t * G; one.sq += t * G;
                cq_matmul_rows(&w, &one, seq, R, 0, R);
                if (memcmp(seq, y + t * R, R * sizeof(float))) { fprintf(stderr, "batch mismatch kind %d n %d row %d\n", kind, n, t); return 1; }
            }
        }
        free(w.codes); free(w.scale); free(w.bias); free(w.sub); free(w.super);
    }
    free(a.q); free(a.s); free(a.sq);
    puts("all five formats: row partitions, batch sizes 1..17, scalar tolerance, padded rows PASS");
}
