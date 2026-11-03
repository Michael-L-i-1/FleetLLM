// Dump reference logits for supplied token IDs, plus sequence-state checks.
#include "common.h"
#include <assert.h>
int main(int argc, char **argv) {
    runtime rt = load_runtime(argc, argv);
    const char *out = arg_str(argc, argv, "--output", "/tmp/fleet-logits.bin");
    const char *prompt = arg_str(argc, argv, "--prompt", "The capital of France is");
    int32_t *ids;
    int n =
        arg_flag(argc, argv, "--chat") ? encode_chat(rt.tok, prompt, !arg_flag(argc, argv, "--think"), &ids) : tok_encode(rt.tok, prompt, strlen(prompt), &ids);
    seq_pages pg = {0};
    const seq_pages *pages[] = {&pg};
    float *logits = malloc(4 * (size_t)n * rt.m->c.vocab);
    row_t *rows = calloc(n, sizeof(row_t));
    for (int i = 0; i < n; i++) {
        assert(kv_reserve(&rt.e->kv, &pg, i));
        rows[i] = (row_t){.seq = 0, .pos = i, .token = ids[i], .ctx_end = i, .logits = true, .input = true};
    }
    int chunk = (int)arg_num(argc, argv, "--chunk", n);
    assert(chunk > 0);
    for (int start = 0; start < n; start += chunk) {
        int take = n - start < chunk ? n - start : chunk;
        assert(!rt.be->forward(rt.be, rt.m, &rt.e->kv, pages, rows + start, take, logits + (size_t)start * rt.m->c.vocab));
        pg.len += take;
    }
    FILE *f = fopen(out, "wb");
    assert(f);
    fwrite(logits, 4, (size_t)n * rt.m->c.vocab, f);
    fclose(f);
    printf("{\"tokens\":[");
    for (int i = 0; i < n; i++)
        printf("%s%d", i ? "," : "", ids[i]);
    printf("],\"vocab\":%d,\"argmax\":[", rt.m->c.vocab);
    for (int r = 0; r < n; r++) {
        int best = 0;
        for (int v = 1; v < rt.m->c.vocab; v++)
            if (logits[(size_t)r * rt.m->c.vocab + v] > logits[(size_t)r * rt.m->c.vocab + best]) best = v;
        printf("%s%d", r ? "," : "", best);
    }
    puts("]}");
    kv_free_seq(&rt.e->kv, &pg);
    free(logits);
    free(rows);
    free(ids);
    return 0;
}
