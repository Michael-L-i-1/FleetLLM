// Isolated real-weight matmul sweep. Optional llama.cpp shared library is a
// benchmark dependency only, resolved at runtime (not linked into FleetLLM).
// cc -O3 -mcpu=native -Isrc bench/cpu_kernels.c build/libfleetllm.a -lpthread -lm -ldl -o build/cpu-kernels
// build/cpu-kernels MODEL.gguf [libggml-cpu.so] [repetitions]
#include "cpuq.h"
#include "quant.h"
#include "threadpool.h"
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*quant_fn)(const float *, void *, int64_t);
typedef struct {
    float d;
    int8_t qs[256];
    int16_t sums[16];
} q8k;
#ifdef FLEET_BASELINE
void baseline_cq_matmul_rows(const cq_mat *, const cq_act *, float *, int, int, int);
#endif
typedef struct {
    cq_mat *w;
    cq_act *a;
    float *out;
    int baseline;
} tiled_job;
typedef struct {
    qmat w;
    q8k *a;
    int n;
    dot_fn dot;
    float *out;
} raw_job;
static void tiled(void *v, int r0, int r1) {
    tiled_job *j = v;
#ifdef FLEET_BASELINE
    if (j->baseline)
        baseline_cq_matmul_rows(j->w, j->a, j->out, j->w->rows, r0, r1);
    else
#endif
        cq_matmul_rows(j->w, j->a, j->out, j->w->rows, r0, r1);
}
static void raw(void *v, int r0, int r1) {
    raw_job *j = v;
    for (int r = r0; r < r1; r++)
        for (int t = 0; t < j->n; t++)
            j->dot(j->w.cols, j->out + (size_t)t * j->w.rows + r, 0, j->w.data + (size_t)r * j->w.row_bytes, 0,
                   j->a + (size_t)t * (j->w.cols / 256), 0, 1);
}
static double now(void) {
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
    void *lib = argc > 2 ? dlopen(argv[2], RTLD_NOW | RTLD_LOCAL) : NULL;
    quant_fn quant = lib ? (quant_fn)dlsym(lib, "quantize_row_q8_K") : NULL;
    if (argc > 2 && !lib) {
        fprintf(stderr, "%s\n", dlerror());
        return 1;
    }
    int reps = argc > 3 ? atoi(argv[3]) : 10;
    int group = argc > 4 ? atoi(argv[4]) : 32;
    if (reps < 1 || (group != 32 && group != 256))
        return 1;
    const int types[] = {GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q6_K};
    puts("tensor,type,rows,cols,batch,activation_group,kernel,milliseconds,effective_GBps,relative_dense_error");
    for (int kind = 0; kind < 4; kind++) {
        const gguf_tensor *t = NULL;
        for (size_t i = 0; i < g->n_tensors; i++) {
            const gguf_tensor *z = &g->tensors[i];
            if (z->type == types[kind] && z->n_dims == 2 && z->dims[0] >= 512 && z->dims[0] % 256 == 0 &&
                (!t || z->nbytes > t->nbytes))
                t = z;
        }
        if (!t)
            continue;
        size_t be, bb;
        ggml_block(t->type, &be, &bb);
        qmat w = {.type = t->type,
                  .rows = t->dims[1],
                  .cols = t->dims[0],
                  .row_bytes = t->dims[0] / be * bb,
                  .data = t->data};
        cq_mat *c = cq_from_qmat(&w);
        if (!c)
            return 2;
        float *x = malloc((size_t)8 * w.cols * sizeof(float)), *out = malloc((size_t)8 * w.rows * sizeof(float)),
              *row = malloc(w.cols * sizeof(float));
        q8k *qa = malloc((size_t)8 * (w.cols / 256) * sizeof(q8k));
        cq_act a = {0};
        for (int i = 0; i < 8 * w.cols; i++)
            x[i] = sinf(i * .13f) + cosf(i * .019f) * .2f;
        char sym[80];
        snprintf(sym, sizeof sym, "ggml_vec_dot_q%d_K_q8_K", kind == 0 ? 2 : kind == 1 ? 3 : kind == 2 ? 4 : 6);
        dot_fn fn = lib ? (dot_fn)dlsym(lib, sym) : NULL;
        for (int n = 1; n <= 8; n *= 2) {
            cq_quantize_group(&a, x, n, w.cols, group);
            if (quant)
                for (int i = 0; i < n; i++)
                    quant(x + (size_t)i * w.cols, qa + (size_t)i * (w.cols / 256), w.cols);
            int methods = 2;
#ifdef FLEET_BASELINE
            methods = 3;
#endif
            for (int method = 0; method < methods; method++) {
                if (method == 1 && (!fn || !quant))
                    continue;
                tiled_job tj = {c, &a, out, method == 2};
                raw_job rj = {w, qa, n, fn, out};
                if (method == 1)
                    parallel_for(w.rows, 64, raw, &rj);
                else
                    parallel_for(w.rows, 64, tiled, &tj);
                double start = now();
                for (int rep = 0; rep < reps; rep++) {
                    if (method == 1)
                        parallel_for(w.rows, 64, raw, &rj);
                    else
                        parallel_for(w.rows, 64, tiled, &tj);
                }
                double sec = (now() - start) / reps;
                double err = 0, mx = 0;
                for (int r = 0; r < 4 && r < w.rows; r++) {
                    qmat_row(&w, r, row);
                    double ref = 0;
                    for (int k = 0; k < w.cols; k++)
                        ref += (double)row[k] * x[k];
                    err = fmax(err, fabs(out[r] - ref));
                    mx = fmax(mx, fabs(ref));
                }
                printf("%s,%s,%d,%d,%d,%d,%s,%.5f,%.3f,%.6g\n", t->name, ggml_type_name(t->type), w.rows, w.cols, n,
                       group,
                       method == 1   ? "llama-row"
                       : method == 2 ? "fleet-reference"
                                     : "fleet-tile",
                       sec * 1e3, (method == 1 ? t->nbytes : cq_bytes(c)) / sec / 1e9, err / (mx + 1e-12));
                fflush(stdout);
            }
        }
        cq_free(c);
        free(x);
        free(out);
        free(row);
        free(qa);
        free(a.q);
        free(a.s);
        free(a.sq);
        free(a.s16);
        free(a.is16);
        free(a.nz);
    }
    gguf_close(g);
    if (lib)
        dlclose(lib);
    return 0;
}
