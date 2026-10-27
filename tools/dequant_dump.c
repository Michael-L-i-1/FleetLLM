// Dump the first `rows` dequantized rows of a tensor as raw f32 (for scripts/dequantcheck.py).
//   dequant_dump model.gguf tensor_name rows out.bin
#include "gguf.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: dequant_dump model.gguf tensor rows out.bin\n"); return 1; }
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) return 1;
    const gguf_tensor *t = gguf_tensor_find(g, argv[2]);
    if (!t) { fprintf(stderr, "no tensor %s\n", argv[2]); return 1; }
    size_t be, bb, cols = t->dims[0];
    ggml_block(t->type, &be, &bb);
    size_t rows = (size_t)atol(argv[3]), row_bytes = cols / be * bb;
    float *y = malloc(sizeof(float) * cols);
    FILE *f = fopen(argv[4], "wb");
    for (size_t r = 0; r < rows; r++) {
        if (!dequant_row(t->type, t->data + r * row_bytes, y, cols)) { fprintf(stderr, "unsupported type %s\n", ggml_type_name(t->type)); return 1; }
        fwrite(y, sizeof(float), cols, f);
    }
    fclose(f);
    printf("%s %s %zux%zu\n", argv[2], ggml_type_name(t->type), rows, cols);
    gguf_close(g);
    return 0;
}
