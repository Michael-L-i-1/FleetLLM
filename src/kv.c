#include "kv.h"
#include "quant.h"
#include <math.h>
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
    p->scale_off = p->page_elems;
    p->page_stride = p->page_elems + (elem == 1 ? (size_t)n_layers * 2 * kv_heads * KV_PAGE * 2 : 0);
    p->n_pages = pages;
    p->bytes = p->page_stride * pages * elem;
    p->data = pages ? alloc(user, p->bytes, &p->handle) : NULL;
    p->refcnt = calloc(pages ? pages : 1, sizeof(uint32_t));
    p->free_list = malloc(sizeof(uint32_t) * (pages ? pages : 1));
    for (uint32_t i = 0; i < pages; i++) p->free_list[i] = pages - 1 - i; // pop from the end: page 0 first
    p->n_free = pages;
}

void kv_init(kv_cache *kv, int full_layers, int swa_layers, int kv_heads, int head_dim, int window, size_t budget, int elem, kv_alloc_fn alloc, void *user) {
    size_t per = (size_t)2 * kv_heads * KV_PAGE * (head_dim * elem + (elem == 1 ? 2 : 0)); // bytes per layer per page
    size_t full_pb = full_layers * per, swa_pb = swa_layers * per;
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
    kv->recurrent_bytes = 0;
    kv->recurrent_limit = kv->recurrent_used = 0;
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
    size_t pb = p->page_stride * p->elem;
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
    if (kv->recurrent_bytes && !s->recurrent) {
        if (kv->recurrent_used >= kv->recurrent_limit) return false;
        s->recurrent_bytes = (kv->recurrent_bytes + 16383) & ~(size_t)16383;
        if (posix_memalign((void **)&s->recurrent, 16384, s->recurrent_bytes)) return false;
        memset(s->recurrent, 0, s->recurrent_bytes);
        kv->recurrent_used++;
    }
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

bool kv_has_fork_at(const kv_cache *kv, const seq_pages *s, int len) {
    if (len < 0 || len > s->len) return false;
    if (kv->recurrent_bytes && len && (!s->recurrent || len != s->recurrent_len)) return false;
    if (!kv->window || len == 0) return true;
    int first = (len + 1 > kv->window ? len + 1 - kv->window : 0) / KV_PAGE;
    int last = (len + KV_PAGE - 1) / KV_PAGE;
    for (int pi = first; pi < last; pi++) if (s->swa[pi] == KV_NONE) return false;
    return true;
}

bool kv_can_fork_at(const kv_cache *kv, const seq_pages *s, int len) {
    return kv_has_fork_at(kv, s, len) && (!kv->recurrent_bytes || !len || kv->recurrent_used < kv->recurrent_limit);
}

static seq_pages fork_attention_at(kv_cache *kv, const seq_pages *s, int len) {
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

seq_pages kv_fork_at(kv_cache *kv, const seq_pages *s, int len) {
    if (!kv_can_fork_at(kv, s, len)) {
        fprintf(stderr, "KV: unavailable fork snapshot\n");
        abort();
    }
    seq_pages c = fork_attention_at(kv, s, len);
    if (len && s->recurrent) {
        c.recurrent_bytes = s->recurrent_bytes;
        if (posix_memalign((void **)&c.recurrent, 16384, c.recurrent_bytes)) abort();
        memcpy(c.recurrent, s->recurrent, c.recurrent_bytes);
        c.recurrent_len = len;
        kv->recurrent_used++;
    }
    return c;
}

seq_pages kv_fork_at_move(kv_cache *kv, seq_pages *s, int len) {
    if (!len || !s->recurrent || !kv->recurrent_bytes || !kv_has_fork_at(kv, s, len)) {
        fprintf(stderr, "KV: unavailable recurrent move snapshot\n");
        abort();
    }
    seq_pages c = fork_attention_at(kv, s, len);
    c.recurrent = s->recurrent;
    c.recurrent_bytes = s->recurrent_bytes;
    c.recurrent_len = s->recurrent_len;
    c.recurrent_handle = s->recurrent_handle;
    c.release_recurrent_handle = s->release_recurrent_handle;
    s->recurrent = NULL;
    s->recurrent_bytes = 0;
    s->recurrent_len = 0;
    s->recurrent_handle = NULL;
    s->release_recurrent_handle = NULL;
    return c;
}

void kv_free_seq(kv_cache *kv, seq_pages *s) {
    for (int i = 0; i < s->n_pages; i++) {
        prelease(&kv->full, s->full[i]);
        if (kv->window) prelease(&kv->swa, s->swa[i]);
    }
    if (s->recurrent_handle) s->release_recurrent_handle(s->recurrent_handle);
    if (s->recurrent) kv->recurrent_used--;
    free(s->recurrent);
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

void kv_truncate(kv_cache *kv, seq_pages *s, int len) {
    if (s->recurrent && len < s->recurrent_len) {
        fprintf(stderr, "recurrent rollback requires a checkpoint\n");
        abort();
    }
    int keep = (len + KV_PAGE - 1) / KV_PAGE;
    for (int i = keep; i < s->n_pages; i++) {
        prelease(&kv->full, s->full[i]);
        if (kv->window) prelease(&kv->swa, s->swa[i]);
    }
    if (keep < s->n_pages) s->n_pages = keep;
    if (len < s->len) s->len = len;
}

void kv_release_flagged(kv_cache *kv, seq_pages *s, const uint8_t *flags) {
    for (int i = 0; i < s->n_pages; i++) {
        if (flags[i] & 1) { prelease(&kv->full, s->full[i]); s->full[i] = KV_NONE; }
        if (flags[i] & 2) { prelease(&kv->swa, s->swa[i]); s->swa[i] = KV_NONE; }
    }
}

bool kv_fill_flagged(kv_cache *kv, seq_pages *s, const uint8_t *flags) {
    uint32_t nf = 0, nw = 0;
    for (int i = 0; i < s->n_pages; i++) { nf += flags[i] & 1; nw += (flags[i] >> 1) & 1; }
    if (kv->full.n_free < nf || (nw && kv->swa.n_free < nw)) return false;
    for (int i = 0; i < s->n_pages; i++) {
        if (flags[i] & 1) palloc(&kv->full, &s->full[i]);
        if (flags[i] & 2) palloc(&kv->swa, &s->swa[i]);
    }
    return true;
}

// One token's K or V vector of one (layer, head): load as floats / store back
// (8-bit: per-token-per-head f16 scale, requantized on store).
void kv_vec_load(const kv_pool *p, uint32_t page, int layer, int kv, int head, int slot, float *x) {
    size_t at = kv_tile(p, page, layer, kv, head) + (size_t)slot * p->head_dim;
    if (p->elem == 4) memcpy(x, (const float *)p->data + at, sizeof(float) * p->head_dim);
    else if (p->elem == 2)
        for (int i = 0; i < p->head_dim; i++) x[i] = fp16_to_f32(((const uint16_t *)p->data)[at + i]);
    else {
        const uint8_t *pg = p->data + (size_t)page * p->page_stride;
        uint16_t sh;
        memcpy(&sh, pg + p->scale_off + (((size_t)layer * 2 + kv) * p->kv_heads + head) * KV_PAGE * 2 + (size_t)slot * 2, 2);
        float sc = fp16_to_f32(sh);
        for (int i = 0; i < p->head_dim; i++) x[i] = (float)(int8_t)p->data[at + i] * sc;
    }
}

void kv_vec_store(kv_pool *p, uint32_t page, int layer, int kv, int head, int slot, const float *x) {
    size_t at = kv_tile(p, page, layer, kv, head) + (size_t)slot * p->head_dim;
    if (p->elem == 4) memcpy((float *)p->data + at, x, sizeof(float) * p->head_dim);
    else if (p->elem == 2)
        for (int i = 0; i < p->head_dim; i++) ((uint16_t *)p->data)[at + i] = f32_to_fp16(x[i]);
    else { // as kv_write8: scale = absmax / 127, stored as f16
        float mx = 0;
        for (int i = 0; i < p->head_dim; i++) mx = fmaxf(mx, fabsf(x[i]));
        uint16_t sh = f32_to_fp16(mx > 0 ? mx / 127.0f : 1.0f);
        float sc = fp16_to_f32(sh);
        for (int i = 0; i < p->head_dim; i++) {
            float q = rintf(x[i] / sc);
            p->data[at + i] = (uint8_t)(int8_t)(q > 127 ? 127 : q < -127 ? -127 : q);
        }
        memcpy(p->data + (size_t)page * p->page_stride + p->scale_off + (((size_t)layer * 2 + kv) * p->kv_heads + head) * KV_PAGE * 2 + (size_t)slot * 2, &sh, 2);
    }
}

// Exact copy of one token's K or V vector (values and, for 8-bit, its scale).
static void vec_copy(kv_pool *p, uint32_t ps, int sslot, uint32_t pd, int dslot, int layer, int kv, int head) {
    size_t eb = (size_t)p->elem, hd = (size_t)p->head_dim;
    memcpy(p->data + (kv_tile(p, pd, layer, kv, head) + (size_t)dslot * hd) * eb, p->data + (kv_tile(p, ps, layer, kv, head) + (size_t)sslot * hd) * eb, hd * eb);
    if (p->elem == 1) {
        size_t so = p->scale_off + (((size_t)layer * 2 + kv) * p->kv_heads + head) * KV_PAGE * 2;
        memcpy(p->data + (size_t)pd * p->page_stride + so + (size_t)dslot * 2, p->data + (size_t)ps * p->page_stride + so + (size_t)sslot * 2, 2);
    }
}

static void pool_move(kv_pool *p, const uint32_t *sp, int src_pos, const uint32_t *dp, int dst_pos, int n, const uint8_t *rope, const kv_rope *r) {
    int hd = p->head_dim, half = hd / 2;
    float *x = malloc(sizeof(float) * hd), *y = malloc(sizeof(float) * hd), *c = malloc(sizeof(float) * half), *s = malloc(sizeof(float) * half);
    double d = (double)dst_pos - src_pos;
    for (int i = 0; i < half; i++) { c[i] = (float)cos(d * r->inv_freq[i]); s[i] = (float)sin(d * r->inv_freq[i]); }
    for (int t = 0; t < n; t++) {
        int a = src_pos + t, b = dst_pos + t;
        uint32_t ps = sp[a / KV_PAGE], pd = dp[b / KV_PAGE];
        if (ps == KV_NONE || pd == KV_NONE) continue;
        for (int l = 0; l < p->n_layers; l++)
            for (int h = 0; h < p->kv_heads; h++) {
                if (rope[l] && d != 0) { // K: rotate by the position change
                    kv_vec_load(p, ps, l, 0, h, a % KV_PAGE, x);
                    for (int i = 0; i < half; i++) {
                        int i0 = r->neox ? i : 2 * i, i1 = r->neox ? i + half : 2 * i + 1;
                        y[i0] = x[i0] * c[i] - x[i1] * s[i];
                        y[i1] = x[i0] * s[i] + x[i1] * c[i];
                    }
                    kv_vec_store(p, pd, l, 0, h, b % KV_PAGE, y);
                } else vec_copy(p, ps, a % KV_PAGE, pd, b % KV_PAGE, l, 0, h);
                vec_copy(p, ps, a % KV_PAGE, pd, b % KV_PAGE, l, 1, h); // V: unchanged
            }
    }
    free(x); free(y); free(c); free(s);
}

void kv_move(kv_cache *kv, const seq_pages *src, int src_pos, const seq_pages *dst, int dst_pos, int n, const kv_rope *r) {
    if (kv->full.n_layers) pool_move(&kv->full, src->full, src_pos, dst->full, dst_pos, n, r->full_rope, r);
    if (kv->window && kv->swa.n_layers) pool_move(&kv->swa, src->swa, src_pos, dst->swa, dst_pos, n, r->swa_rope, r);
}
