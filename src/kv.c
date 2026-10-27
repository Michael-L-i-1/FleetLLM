#include "kv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void pool_init(kv_pool *p, int n_layers, int kv_heads, int head_dim, uint32_t pages, int elem, kv_alloc_fn alloc, void *user) {
    memset(p, 0, sizeof *p);
    p->n_layers = n_layers;
    p->kv_heads = kv_heads;
    p->head_dim = head_dim;
    p->elem = elem;
    p->page_elems = (size_t)n_layers * 2 * kv_heads * KV_PAGE * head_dim;
    p->n_pages = pages;
    p->bytes = p->page_elems * pages * elem;
    p->data = pages ? alloc(user, p->bytes, &p->handle) : NULL;
    p->refcnt = calloc(pages ? pages : 1, sizeof(uint32_t));
    p->free_list = malloc(sizeof(uint32_t) * (pages ? pages : 1));
    for (uint32_t i = 0; i < pages; i++) p->free_list[i] = pages - 1 - i; // pop from the end: page 0 first
    p->n_free = pages;
}

void kv_init(kv_cache *kv, int full_layers, int swa_layers, int kv_heads, int head_dim, int window, size_t budget, int elem, kv_alloc_fn alloc, void *user) {
    size_t full_pb = (size_t)full_layers * 2 * kv_heads * KV_PAGE * head_dim * elem;
    size_t swa_pb = (size_t)swa_layers * 2 * kv_heads * KV_PAGE * head_dim * elem;
    uint32_t swa_pages = 0;
    if (swa_layers && window) {
        // a sequence needs one full page per 16 tokens but at most window/16+2 SWA pages;
        // give SWA up to 3/4 of the budget, capped at what 32 full-window agents need
        size_t cap = (size_t)(window / KV_PAGE + 2) * 32 * swa_pb;
        size_t b = cap < budget * 3 / 4 ? cap : budget * 3 / 4;
        swa_pages = (uint32_t)(b / swa_pb);
    }
    uint32_t full_pages = full_pb ? (uint32_t)((budget - (size_t)swa_pages * swa_pb) / full_pb) : 0;
    pool_init(&kv->full, full_layers, kv_heads, head_dim, full_pages, elem, alloc, user);
    pool_init(&kv->swa, swa_layers, kv_heads, head_dim, swa_pages, elem, alloc, user);
    kv->window = swa_pages ? window : 0;
}

static bool palloc(kv_pool *p, uint32_t *out) {
    if (!p->n_free) return false;
    *out = p->free_list[--p->n_free];
    p->refcnt[*out] = 1;
    return true;
}

static void prelease(kv_pool *p, uint32_t page) {
    if (page == KV_NONE) return;
    if (--p->refcnt[page] == 0) p->free_list[p->n_free++] = page;
}

static bool pcow(kv_pool *p, uint32_t *page) {
    if (p->refcnt[*page] == 1) return true;
    uint32_t n;
    if (!palloc(p, &n)) return false;
    size_t pb = p->page_elems * p->elem;
    memcpy(p->data + (size_t)n * pb, p->data + (size_t)*page * pb, pb);
    prelease(p, *page);
    *page = n;
    return true;
}

static void grow(seq_pages *s) {
    if (s->n_pages < s->cap) return;
    s->cap = s->cap ? s->cap * 2 : 16;
    s->full = realloc(s->full, sizeof(uint32_t) * s->cap);
    s->swa = realloc(s->swa, sizeof(uint32_t) * s->cap);
}

bool kv_reserve(kv_cache *kv, seq_pages *s, int pos) {
    int pi = pos / KV_PAGE;
    if (pi == s->n_pages) {
        grow(s);
        uint32_t f, w = KV_NONE;
        if (!palloc(&kv->full, &f)) return false;
        if (kv->window && !palloc(&kv->swa, &w)) { prelease(&kv->full, f); return false; }
        s->full[pi] = f;
        s->swa[pi] = w;
        s->n_pages++;
        return true;
    }
    if (pi > s->n_pages) return false;
    if (!pcow(&kv->full, &s->full[pi])) return false;
    if (kv->window && s->swa[pi] != KV_NONE && !pcow(&kv->swa, &s->swa[pi])) return false;
    return true;
}

bool kv_can_fork_at(const kv_cache *kv, const seq_pages *s, int len) {
    if (len > s->len) return false;
    if (!kv->window || len == 0) return true;
    int first = (len + 1 > kv->window ? len + 1 - kv->window : 0) / KV_PAGE;
    int last = (len + KV_PAGE - 1) / KV_PAGE;
    for (int pi = first; pi < last; pi++) if (s->swa[pi] == KV_NONE) return false;
    return true;
}

seq_pages kv_fork_at(kv_cache *kv, const seq_pages *s, int len) {
    seq_pages c = {0};
    int np = (len + KV_PAGE - 1) / KV_PAGE;
    c.cap = np > 16 ? np : 16;
    c.full = malloc(sizeof(uint32_t) * c.cap);
    c.swa = malloc(sizeof(uint32_t) * c.cap);
    c.n_pages = np;
    c.len = len;
    for (int i = 0; i < np; i++) {
        c.full[i] = s->full[i];
        c.swa[i] = s->swa[i];
        kv->full.refcnt[c.full[i]]++;
        if (c.swa[i] != KV_NONE) kv->swa.refcnt[c.swa[i]]++;
    }
    return c;
}

void kv_free_seq(kv_cache *kv, seq_pages *s) {
    for (int i = 0; i < s->n_pages; i++) {
        prelease(&kv->full, s->full[i]);
        if (kv->window) prelease(&kv->swa, s->swa[i]);
    }
    free(s->full);
    free(s->swa);
    memset(s, 0, sizeof *s);
}

void kv_trim_window(kv_cache *kv, seq_pages *s) {
    if (!kv->window || s->len < kv->window + KV_PAGE) return;
    int first_needed = s->len - (kv->window - 1); // future q >= len attends k >= q - window + 1
    for (int pi = 0; pi < first_needed / KV_PAGE; pi++) {
        if (s->swa[pi] != KV_NONE) { prelease(&kv->swa, s->swa[pi]); s->swa[pi] = KV_NONE; }
    }
}
