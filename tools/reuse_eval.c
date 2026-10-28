// Quality of reusing a span's KV computed in another context (another agent
// read the same file): for a chunk X of a file and its continuation Y,
//   exact:  prefill S + B + X, score Y
//   reused: prefill S + A + X elsewhere, then S + B, move X's KV over (K rotated
//           to the new positions), recompute its first --head and last --tail
//           tokens, score Y
// A and B are different preceding contexts of different lengths. Reports mean
// nll of Y per variant over several (A, B, X) triples taken from --files.
// --same uses B as the source context too (the move must then be exact).
//   reuse_eval --model llama8b-mlx4 --kv-int8 --files src/*.c
#include "common.h"
#include <math.h>

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1);
    if (fread(b, 1, n, f) != (size_t)n) { perror(path); exit(1); }
    b[n] = 0;
    fclose(f);
    return b;
}

typedef struct { int32_t *t; int n; } toks;

static toks enc(runtime *rt, const char *s, size_t len) {
    toks r;
    r.n = tok_encode(rt->tok, s, len, &r.t);
    return r;
}

// Prefill tokens t[0..n) at positions pos0.. of pg; when score, add the nll of
// each next token in t (and of `next` after the last) to *nll.
static void prefill(runtime *rt, seq_pages *pg, const int32_t *t, int n, int pos0, const int32_t *score, int n_score, double *nll) {
    enum { CH = 128 };
    static float *logits;
    static row_t rows[CH];
    int vocab = rt->m->c.vocab;
    if (!logits) logits = malloc(sizeof(float) * CH * vocab);
    const seq_pages *pp = pg;
    for (int p0 = 0; p0 < n; p0 += CH) {
        int k = n - p0 < CH ? n - p0 : CH;
        for (int i = 0; i < k; i++) {
            if (!kv_reserve(&rt->e->kv, pg, pos0 + p0 + i)) { fprintf(stderr, "reuse_eval: KV full\n"); exit(1); }
            rows[i] = (row_t){ .seq = 0, .pos = pos0 + p0 + i, .token = t[p0 + i], .ctx_end = pos0 + p0 + i, .logits = score != NULL, .input = true };
        }
        if (rt->be->forward(rt->be, rt->m, &rt->e->kv, &pp, rows, k, logits)) exit(1);
        pg->len = pos0 + p0 + k;
        kv_trim_window(&rt->e->kv, pg);
        for (int i = 0; score && i < k; i++) {
            int j = p0 + i - (n - n_score - 1); // index in score of the token this row predicts
            if (j < 0 || j >= n_score) continue;
            const float *l = logits + (size_t)i * vocab;
            float mx = l[0];
            for (int v = 1; v < vocab; v++) if (l[v] > mx) mx = l[v];
            double z = 0;
            for (int v = 0; v < vocab; v++) z += exp((double)l[v] - mx);
            *nll += log(z) + mx - l[score[j]];
        }
    }
}

static void cat(toks *d, const toks *s) {
    d->t = realloc(d->t, sizeof(int32_t) * (d->n + s->n));
    memcpy(d->t + d->n, s->t, sizeof(int32_t) * s->n);
    d->n += s->n;
}

int main(int argc, char **argv) {
    runtime rt = load_runtime(argc, argv);
    int head = (int)arg_num(argc, argv, "--head", -1), tail = (int)arg_num(argc, argv, "--tail", 1);
    int x_chars = (int)arg_num(argc, argv, "--x-chars", 2400), y_chars = (int)arg_num(argc, argv, "--y-chars", 1200);
    const char *list[64];
    int nf = 0;
    for (int i = 1; i < argc && nf < 64; i++)
        if (!strcmp(argv[i], "--files")) for (int j = i + 1; j < argc && argv[j][0] != '-' && nf < 64; j++) list[nf++] = argv[j];
    if (nf < 3) { fprintf(stderr, "usage: reuse_eval --model M --files f1 f2 f3 ... [--head R] [--tail T]\n"); return 1; }
    kv_rope rope = { rt.m->inv_freq, rt.m->c.rope_neox, NULL, NULL };
    uint8_t fr[256] = { 0 }, sr[256] = { 0 };
    for (int l = 0; l < rt.m->c.n_layer; l++) (rt.m->layers[l].swa ? sr : fr)[rt.m->layers[l].pool_layer] = rt.m->layers[l].rope;
    rope.full_rope = fr;
    rope.swa_rope = sr;
    const char *sys = "You are a coding agent working in a C inference-engine repository. Tools: read_file(path) returns a file.\n";
    int heads[] = { 0, 4, 16, 64 }, nh = head >= 0 ? 1 : 4;
    if (head >= 0) heads[0] = head;
    double ex = 0, iso = 0, re[4] = { 0 };
    long scored = 0;
    int trials = 0;
    for (int f = 0; f + 2 < nf; f++) {
        char *xa = read_file(list[f + 2]);
        if ((int)strlen(xa) < x_chars + y_chars) { free(xa); continue; }
        char *pa = read_file(list[f]), *pb = read_file(list[f + 1]);
        bool same = arg_flag(argc, argv, "--same"); // sanity: identical source context, no shift
        if (same) { free(pa); pa = strdup(pb); }
        toks S = enc(&rt, sys, strlen(sys)), A = enc(&rt, pa, strlen(pa) < (same ? 1800 : 3000) ? strlen(pa) : (same ? 1800 : 3000)), B = enc(&rt, pb, strlen(pb) < 1800 ? strlen(pb) : 1800);
        toks X = enc(&rt, xa, x_chars), Y = enc(&rt, xa + x_chars, y_chars);
        toks pre_a = { 0 }, pre_b = { 0 };
        int32_t bos = tok_bos(rt.tok);
        toks BOS = { &bos, bos >= 0 };
        cat(&pre_a, &BOS); cat(&pre_a, &S); cat(&pre_a, &A);
        cat(&pre_b, &BOS); cat(&pre_b, &S); cat(&pre_b, &B);
        // exact: B context, X, Y
        seq_pages pg = { 0 };
        toks all = { 0 };
        cat(&all, &pre_b); cat(&all, &X); cat(&all, &Y);
        prefill(&rt, &pg, all.t, all.n, 0, Y.t, Y.n, &ex);
        kv_free_seq(&rt.e->kv, &pg);
        // no preceding context at all (how much context matters)
        seq_pages pi = { 0 };
        toks bare = { 0 };
        cat(&bare, &BOS); cat(&bare, &X); cat(&bare, &Y);
        prefill(&rt, &pi, bare.t, bare.n, 0, Y.t, Y.n, &iso);
        kv_free_seq(&rt.e->kv, &pi);
        // source: A context, X
        seq_pages src = { 0 };
        toks sa = { 0 };
        cat(&sa, &pre_a); cat(&sa, &X);
        prefill(&rt, &src, sa.t, sa.n, 0, NULL, 0, NULL);
        for (int v = 0; v < nh; v++) {
            int R = heads[v] < X.n - tail ? heads[v] : X.n - tail, mid = X.n - R - tail;
            seq_pages d = { 0 };
            toks pre = { 0 };
            cat(&pre, &pre_b);
            for (int i = 0; i < R; i++) pre.t = realloc(pre.t, sizeof(int32_t) * (pre.n + 1)), pre.t[pre.n++] = X.t[i];
            prefill(&rt, &d, pre.t, pre.n, 0, NULL, 0, NULL);
            for (int i = 0; i < mid; i++) if (!kv_reserve(&rt.e->kv, &d, pre.n + i)) { fprintf(stderr, "KV full\n"); return 1; }
            kv_move(&rt.e->kv, &src, pre_a.n + R, &d, pre.n, mid, &rope);
            d.len = pre.n + mid;
            toks rest = { 0 };
            for (int i = X.n - tail; i < X.n; i++) rest.t = realloc(rest.t, sizeof(int32_t) * (rest.n + 1)), rest.t[rest.n++] = X.t[i];
            cat(&rest, &Y);
            prefill(&rt, &d, rest.t, rest.n, pre.n + mid, Y.t, Y.n, &re[v]);
            kv_free_seq(&rt.e->kv, &d);
            free(pre.t); free(rest.t);
        }
        kv_free_seq(&rt.e->kv, &src);
        scored += Y.n;
        trials++;
        printf("trial %d: %s after %s vs %s (X %d tok, Y %d tok, shift %d)\n", trials, list[f + 2], list[f], list[f + 1], X.n, Y.n, pre_b.n - pre_a.n);
        free(xa); free(pa); free(pb); free(all.t); free(bare.t); free(sa.t); free(pre_a.t); free(pre_b.t);
        fflush(stdout);
    }
    printf("Y nll over %ld tokens, %d trials:\n  exact %.5f (ppl %.3f)\n  X with no preceding context %.5f (ppl %.3f)\n", scored, trials, ex / scored, exp(ex / scored),
           iso / scored, exp(iso / scored));
    for (int v = 0; v < nh; v++) printf("  reused, first %d + last %d recomputed %.5f (ppl %.3f)\n", heads[v], tail, re[v] / scored, exp(re[v] / scored));
    return 0;
}
