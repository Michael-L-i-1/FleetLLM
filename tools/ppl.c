// Perplexity of a text file under teacher forcing, to measure lossy options
// (f16 dot products, KV quantization, ...) against the default path.
//   ppl --model glimmer-mlx4 --file eval.txt [--ctx 512] [--max-tokens 8192] [--chunk 64]
// The text is split into independent windows of --ctx tokens (each starting
// with BOS); every token after the first in a window is scored.
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

int main(int argc, char **argv) {
    runtime rt = load_runtime(argc, argv);
    const char *path = arg_str(argc, argv, "--file", NULL);
    if (!path) { fprintf(stderr, "usage: ppl --model M --file text.txt [--ctx 512] [--max-tokens N]\n"); return 1; }
    char *text = read_file(path);
    int32_t *ids;
    int n = tok_encode(rt.tok, text, strlen(text), &ids);
    int ctx = (int)arg_num(argc, argv, "--ctx", 512), max_tok = (int)arg_num(argc, argv, "--max-tokens", 8192);
    int chunk = (int)arg_num(argc, argv, "--chunk", 64);
    if (n > max_tok) n = max_tok;
    int vocab = rt.m->c.vocab;
    float *logits = malloc(sizeof(float) * (size_t)chunk * vocab);
    row_t *rows = malloc(sizeof(row_t) * chunk);
    int32_t *win = malloc(sizeof(int32_t) * (ctx + 1));
    double nll = 0, t0 = now_secs();
    long scored = 0;
    for (int w0 = 0; w0 < n; w0 += ctx - 1) {
        int len = 0;
        if (tok_bos(rt.tok) >= 0) win[len++] = tok_bos(rt.tok);
        for (int i = w0; i < n && len < ctx; i++) win[len++] = ids[i];
        if (len < 2) break;
        seq_pages pg = { 0 };
        const seq_pages *pp = &pg;
        for (int p0 = 0; p0 < len - 1; p0 += chunk) {
            int k = len - 1 - p0 < chunk ? len - 1 - p0 : chunk;
            for (int t = 0; t < k; t++) {
                if (!kv_reserve(&rt.e->kv, &pg, p0 + t)) { fprintf(stderr, "ppl: KV full\n"); return 1; }
                rows[t] = (row_t){ .seq = 0, .pos = p0 + t, .token = win[p0 + t], .ctx_end = p0 + t, .logits = true, .input = true };
            }
            if (rt.be->forward(rt.be, rt.m, &rt.e->kv, &pp, rows, k, logits)) return 1;
            pg.len += k;
            kv_trim_window(&rt.e->kv, &pg);
            for (int t = 0; t < k; t++) {
                const float *l = logits + (size_t)t * vocab;
                float mx = l[0];
                for (int v = 1; v < vocab; v++) if (l[v] > mx) mx = l[v];
                double z = 0;
                for (int v = 0; v < vocab; v++) z += exp((double)l[v] - mx);
                nll += log(z) + mx - l[win[p0 + t + 1]];
                scored++;
            }
        }
        kv_free_seq(&rt.e->kv, &pg);
    }
    printf("ppl %.4f  (nll %.5f over %ld tokens, ctx %d, %.1fs)\n", exp(nll / scored), nll / scored, scored, ctx, now_secs() - t0);
    return 0;
}
