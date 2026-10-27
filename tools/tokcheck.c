// Print token ids for each sample of a file (samples separated by a line "=====").
// scripts/tokcheck.py compares the output with the HF reference tokenizer.
//   tokcheck model.gguf samples.txt
#include "gguf.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: tokcheck model.gguf samples.txt\n"); return 1; }
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) return 1;
    tokenizer *t = tok_load(g);
    if (!t) return 1;
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    fread(buf, 1, n, f);
    buf[n] = 0;
    fclose(f);
    const char *sep = "\n=====\n";
    char *s = buf;
    for (;;) {
        char *e = strstr(s, sep);
        size_t len = e ? (size_t)(e - s) : strlen(s);
        int32_t *ids;
        int k = tok_encode(t, s, len, &ids);
        for (int i = 0; i < k; i++) printf(i ? " %d" : "%d", ids[i]);
        printf("\n");
        char *back = tok_decode(t, ids, k);
        if (strlen(back) != len || memcmp(back, s, len) != 0) fprintf(stderr, "roundtrip mismatch in sample starting %.30s\n", s);
        free(back);
        free(ids);
        if (!e) break;
        s = e + strlen(sep);
    }
    tok_free(t);
    gguf_close(g);
    return 0;
}
