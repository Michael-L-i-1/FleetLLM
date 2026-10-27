// Compute backends. The engine (scheduler, KV bookkeeping, forking) is shared;
// a backend owns KV storage and runs one batched forward step.
#pragma once
#include "kv.h"
#include "model.h"

// One token of one sequence in a batch.
typedef struct {
    int seq;       // index into the pages[] array passed to forward
    int pos;       // position of this token
    int32_t token;
    int ctx_end;   // last position this row attends to (== pos for causal decoding)
    bool logits;   // return logits for this row
} row_t;

// A row of a speculative drafter step (DFlash). Context rows write the
// drafter's KV from main-model hidden states (src = row of the last forward's
// capture); block rows are the anchor token + mask tokens being denoised
// (src = token id) and attend bidirectionally up to ctx_end.
typedef struct { int seq, pos, src, ctx_end; } draft_row;

typedef struct backend backend;
struct backend {
    const char *name;
    int kv_elem; // bytes per KV element
    void *(*alloc_kv)(backend *b, size_t bytes, void **handle);
    // Run one step over rows (KV slots already reserved). Writes vocab floats per
    // row with logits=true, in order, to logits. Returns 0 on success.
    // logits may be NULL when only argmax is wanted.
    int (*forward)(backend *b, const model *m, kv_cache *kv, const seq_pages *const *pages, const row_t *rows, int n, float *logits);
    void (*destroy)(backend *b);
    // Per-forward extras, set by the engine before forward:
    int32_t *argmax;    // non-NULL: argmax token id per logits row (backends with argmax_ok)
    bool argmax_ok;
    int cap_layers[8];  // capture the residual stream after these layers ...
    int n_cap;          // ... (n_cap > 0) for every row, for a speculative drafter
    // Speculative drafting (NULL when unsupported): attach prepares a second
    // model's weights; draft runs one drafter step over blocks of blk_len rows
    // (anchor + masks) and writes the argmax tokens of rows 1..blk_len-1 of each.
    int (*attach)(backend *b, model *dm);
    int (*draft)(backend *b, const model *dm, const model *tm, kv_cache *dkv, const seq_pages *const *pages, const draft_row *ctx, int n_ctx,
                 const draft_row *blk, int n_blk, int blk_len, int32_t *out);
    void *impl;
};

backend *cpu_backend_new(void);
#ifdef CI_METAL
backend *metal_backend_new(model *m);
#endif
// "cpu", "metal", or "auto" (metal when available). A backend may attach
// repacked copies of weight matrices to the model (qmat.gpu).
backend *backend_create(const char *name, model *m);
