// Paged, ref-counted KV cache with copy-on-write pages.
//
// Two pools: full-attention layers (pages live as long as the sequence) and
// sliding-window layers (pages are released once they fall out of the window).
// Forking shares pages at zero cost; the first writer into a shared partial
// page copies it. Page layout: [layer][k|v][kv_head][PAGE tokens][head_dim].
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define KV_PAGE 16
#define KV_NONE UINT32_MAX

typedef struct {
    uint8_t *data; // host-visible storage (backend-owned)
    void *handle;  // backend object (e.g. MTLBuffer)
    size_t bytes, page_elems;
    int n_layers, kv_heads, head_dim, elem;
    uint32_t *refcnt;
    uint32_t *free_list;
    uint32_t n_pages, n_free;
} kv_pool;

typedef struct {
    int len;          // tokens with KV written
    uint32_t *full;   // page ids, one per KV_PAGE tokens
    uint32_t *swa;    // KV_NONE once released (or if the model has no SWA layers)
    int n_pages, cap;
} seq_pages;

typedef void *(*kv_alloc_fn)(void *user, size_t bytes, void **handle);

typedef struct {
    kv_pool full, swa;
    int window; // 0 = no sliding-window layers
} kv_cache;

// Split budget_bytes between the pools; elem = bytes per KV element (4 f32, 2 f16).
void kv_init(kv_cache *kv, int full_layers, int swa_layers, int kv_heads, int head_dim, int window,
             size_t budget_bytes, int elem, kv_alloc_fn alloc, void *user);
// Make the slot for position pos (== len, or a COW of the current tail page) writable.
bool kv_reserve(kv_cache *kv, seq_pages *s, int pos);
// Share the pages covering [0, len) of s into a new page table.
bool kv_can_fork_at(const kv_cache *kv, const seq_pages *s, int len);
seq_pages kv_fork_at(kv_cache *kv, const seq_pages *s, int len);
void kv_free_seq(kv_cache *kv, seq_pages *s);
// Shrink s to len tokens (rejected speculative positions), releasing whole pages past it.
void kv_truncate(kv_cache *kv, seq_pages *s, int len);
// Release sliding-window pages no future query can see.
void kv_trim_window(kv_cache *kv, seq_pages *s);
// Element offset of the [PAGE x head_dim] tile for (page, layer, k=0/v=1, head).
static inline size_t kv_tile(const kv_pool *p, uint32_t page, int layer, int kv, int head) {
    return (size_t)page * p->page_elems + ((size_t)(layer * 2 + kv) * p->kv_heads + head) * KV_PAGE * p->head_dim;
}
