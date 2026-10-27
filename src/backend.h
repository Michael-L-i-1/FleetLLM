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

typedef struct backend backend;
struct backend {
    const char *name;
    int kv_elem; // bytes per KV element
    void *(*alloc_kv)(backend *b, size_t bytes, void **handle);
    // Run one step over rows (KV slots already reserved). Writes vocab floats per
    // row with logits=true, in order, to logits. Returns 0 on success.
    int (*forward)(backend *b, const model *m, kv_cache *kv, const seq_pages *const *pages, const row_t *rows, int n, float *logits);
    void (*destroy)(backend *b);
    void *impl;
};

backend *cpu_backend_new(void);
#ifdef CI_METAL
backend *metal_backend_new(const model *m);
#endif
// "cpu", "metal", or "auto" (metal when available)
backend *backend_create(const char *name, const model *m);
