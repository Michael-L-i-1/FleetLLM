// Exact packed-weight prefetch experiment. Run under the shared Pi benchmark
// lock.
#include "cpuq.h"
#include "quant.h"
#include "threadpool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef void (*kernel)(const cq_mat *, const cq_act *, float *, int, int, int);
#ifndef STREAM_GRAIN_ONLY
#define DECLARE(NAME)                                                          \
    void NAME##_cq_matmul_rows(const cq_mat *, const cq_act *, float *, int,   \
                               int, int)
DECLARE(baseline);
DECLARE(pf4l0);
DECLARE(pf8l0);
DECLARE(pf16l0);
DECLARE(pf32l0);
DECLARE(pf4l3);
DECLARE(pf8l3);
DECLARE(pf16l3);
DECLARE(pf32l3);
#endif
static const struct {
    const char *name;
    kernel fn;
    int grain;
} variants[] = {
#ifdef STREAM_GRAIN_ONLY
    {"baseline", cq_matmul_rows, 64},    {"grain16", cq_matmul_rows, 16},
    {"grain32", cq_matmul_rows, 32},     {"grain128", cq_matmul_rows, 128},
    {"grain256", cq_matmul_rows, 256},   {"grain512", cq_matmul_rows, 512},
    {"grain1024", cq_matmul_rows, 1024}, {"grain4096", cq_matmul_rows, 4096},
    {"static4", cq_matmul_rows, 0}
#else
#define VARIANT(NAME) {#NAME, NAME##_cq_matmul_rows, 64}
    VARIANT(baseline), VARIANT(pf4l0), VARIANT(pf8l0), VARIANT(pf16l0),
    VARIANT(pf32l0),   VARIANT(pf4l3), VARIANT(pf8l3), VARIANT(pf16l3),
    VARIANT(pf32l3)
#undef VARIANT
#endif
};
typedef struct {
    const cq_mat *w;
    const cq_act *a;
    float *out;
    kernel fn;
} job;
static void run(void *v, int r0, int r1) {
    job *j = v;
    j->fn(j->w, j->a, j->out, j->w->rows, r0, r1);
}
static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
int main(int argc, char **argv) {
    if (argc < 2)
        return 1;
    tp_init(4);
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g)
        return 1;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    bool decoder_only = argc > 3 && !strcmp(argv[3], "decoder");
    if (reps < 1)
        return 1;
    puts("tensor,type,batch,round,variant,milliseconds,bitwise_equal");
    const uint32_t types[] = {GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q6_K};
    for (int type = 0; type < 4; type++) {
        const gguf_tensor *tensor = NULL;
        for (size_t i = 0; i < g->n_tensors; i++) {
            const gguf_tensor *t = &g->tensors[i];
            if (decoder_only && !strcmp(t->name, "token_embd.weight"))
                continue;
            if (t->type == types[type] && t->n_dims == 2 && t->dims[0] >= 512 &&
                t->dims[0] % 256 == 0 &&
                (!tensor || t->nbytes > tensor->nbytes))
                tensor = t;
        }
        if (!tensor)
            continue;
        size_t be, bb;
        ggml_block(tensor->type, &be, &bb);
        qmat original = {.type = tensor->type,
                         .rows = tensor->dims[1],
                         .cols = tensor->dims[0],
                         .row_bytes = tensor->dims[0] / be * bb,
                         .data = tensor->data};
        cq_mat *w = cq_from_qmat(&original);
        if (!w)
            return 2;
        float *x = malloc(4 * w->cols * sizeof(float));
        float *out = malloc(4 * w->rows * sizeof(float)),
              *reference = malloc(4 * w->rows * sizeof(float));
        for (int i = 0; i < 4 * w->cols; i++)
            x[i] = sinf(i * .13f) + cosf(i * .019f) * .2f;
        cq_act a = {0};
        for (int n = 1; n <= 4; n *= 4) {
            cq_quantize(&a, x, n, w->cols);
            job j = {w, &a, reference, variants[0].fn};
            parallel_for(w->rows, 64, run, &j);
            j.out = out;
            for (int round = 0; round < reps; round++) {
                // Rotate order to reduce thermal/frequency/order bias.
                for (int v = 0; v < 9; v++) {
                    int i = (v + round) % 9;
                    j.fn = variants[i].fn;
                    double t = seconds();
                    int grain = variants[i].grain ? variants[i].grain
                                                  : ((w->rows + 15) / 16) * 4;
                    parallel_for(w->rows, grain, run, &j);
                    t = seconds() - t;
                    int equal =
                        !memcmp(out, reference, n * w->rows * sizeof(float));
                    printf("%s,%s,%d,%d,%s,%.6f,%d\n", tensor->name,
                           cq_kind_name(w->kind), n, round, variants[i].name,
                           t * 1e3, equal);
                    fflush(stdout);
                    if (!equal)
                        return 3;
                }
            }
        }
        cq_free(w);
        free(x);
        free(out);
        free(reference);
        free(a.q);
        free(a.s);
        free(a.sq);
        free(a.s16);
        free(a.is16);
        free(a.nz);
    }
    gguf_close(g);
    return 0;
}
