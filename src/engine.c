#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double now_secs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void append(int32_t **v, int *n, int *cap, const int32_t *x, int k) {
    if (*n + k > *cap) {
        *cap = (*n + k) * 2 + 16;
        *v = realloc(*v, sizeof(int32_t) * *cap);
    }
    memcpy(*v + *n, x, sizeof(int32_t) * k);
    *n += k;
}

static void *be_alloc(void *user, size_t bytes, void **handle) {
    backend *b = user;
    return b->alloc_kv(b, bytes, handle);
}

engine *engine_new(model *m, backend *be, const tokenizer *tok, size_t kv_budget) {
    engine *e = calloc(1, sizeof *e);
    e->m = m;
    e->be = be;
    e->tok = tok;
    e->row_budget = 128;
    e->min_share = 16;
    const model_cfg *c = &m->c;
    kv_init(&e->kv, c->n_full, c->n_swa, c->n_kv, c->hd, c->window, kv_budget, be->kv_elem, be_alloc, be);
    return e;
}

void engine_free(engine *e) {
    for (int i = 0; i < e->n_seqs; i++) if (e->seqs[i].live) engine_release(e, i);
    free(e->seqs);
    free(e->cached);
    free(e->rows);
    free(e->logits);
    free(e);
}

static int new_seq(engine *e) {
    for (int i = 0; i < e->n_seqs; i++) if (!e->seqs[i].live) { memset(&e->seqs[i], 0, sizeof(seq_t)); e->seqs[i].live = true; return i; }
    e->seqs = realloc(e->seqs, sizeof(seq_t) * (e->n_seqs + 1));
    memset(&e->seqs[e->n_seqs], 0, sizeof(seq_t));
    e->seqs[e->n_seqs].live = true;
    return e->n_seqs++;
}

int engine_create_fresh(engine *e, const int32_t *prompt, int n) {
    int id = new_seq(e);
    seq_t *s = &e->seqs[id];
    append(&s->pending, &s->n_pending, &s->cap_pending, prompt, n);
    s->t_start = now_secs();
    return id;
}

int engine_fork_at(engine *e, int parent, int len, const int32_t *suffix, int n) {
    int id = new_seq(e); // may realloc seqs: take pointers after
    seq_t *p = &e->seqs[parent], *s = &e->seqs[id];
    s->pages = kv_fork_at(&e->kv, &p->pages, len);
    append(&s->tokens, &s->n_tokens, &s->cap_tokens, p->tokens, len);
    append(&s->pending, &s->n_pending, &s->cap_pending, suffix, n);
    s->t_start = now_secs();
    e->st.shared_tokens += len;
    return id;
}

static void touch(engine *e, int id) {
    for (int i = 0; i < e->n_cached; i++)
        if (e->cached[i] == id) {
            memmove(&e->cached[i], &e->cached[i + 1], sizeof(int) * (e->n_cached - i - 1));
            e->cached[e->n_cached - 1] = id;
            return;
        }
}

int engine_create(engine *e, const int32_t *prompt, int n) {
    int best = -1, best_len = 0;
    for (int i = 0; i < e->n_seqs; i++) {
        seq_t *s = &e->seqs[i];
        if (!s->live) continue;
        int l = 0;
        while (l < s->n_tokens && l < n && s->tokens[l] == prompt[l]) l++;
        if (l > n - 1) l = n - 1; // keep one token to produce logits from
        if (l >= e->min_share && l > best_len && kv_can_fork_at(&e->kv, &s->pages, l)) { best = i; best_len = l; }
    }
    if (best < 0) return engine_create_fresh(e, prompt, n);
    touch(e, best);
    return engine_fork_at(e, best, best_len, prompt + best_len, n - best_len);
}

void engine_start(engine *e, int id, sample_params p) {
    seq_t *s = &e->seqs[id];
    if (s->n_pending == 0 && s->n_tokens > 0) {
        // fully prefilled (e.g. it served as a shared prefix): rewind one position and
        // re-feed the last token to get logits; rewriting the same KV slot is idempotent
        int32_t last = s->tokens[--s->n_tokens];
        s->pages.len--;
        append(&s->pending, &s->n_pending, &s->cap_pending, &last, 1);
    }
    s->p = p;
    s->rng = (p.seed + 1) * 0x9E3779B97F4A7C15ULL | 1;
    s->generating = true;
    s->done = false;
    s->n_gen = 0;
    s->t_first = 0;
}

void engine_release(engine *e, int id) {
    seq_t *s = &e->seqs[id];
    if (!s->live) return;
    for (int i = 0; i < e->n_cached; i++)
        if (e->cached[i] == id) { memmove(&e->cached[i], &e->cached[i + 1], sizeof(int) * (e->n_cached - i - 1)); e->n_cached--; break; }
    kv_free_seq(&e->kv, &s->pages);
    free(s->tokens);
    free(s->pending);
    free(s->gen);
    memset(s, 0, sizeof *s);
}

void engine_retire(engine *e, int id) {
    seq_t *s = &e->seqs[id];
    s->generating = false;
    s->done = true;
    s->cached = true;
    e->cached = realloc(e->cached, sizeof(int) * (e->n_cached + 1));
    e->cached[e->n_cached++] = id;
}

// Evict retired sequences until both pools can take `pages` more pages.
static void make_room(engine *e, uint32_t pages) {
    while (e->n_cached && (e->kv.full.n_free < pages || (e->kv.window && e->kv.swa.n_free < pages))) engine_release(e, e->cached[0]);
}

typedef struct { int seq, take; } plan_item;

bool engine_step(engine *e, const int *ids, int n) {
    plan_item *plan = malloc(sizeof(plan_item) * (n ? n : 1));
    int np = 0, budget = e->row_budget, n_decode = 0;
    for (int i = 0; i < n && budget > 0; i++) // decode rows first
        if (e->seqs[ids[i]].n_pending == 1) { plan[np++] = (plan_item){ ids[i], 1 }; budget--; n_decode++; }
    for (int i = 0; i < n && budget > 0; i++) {
        seq_t *s = &e->seqs[ids[i]];
        if (s->n_pending > 1) { int take = s->n_pending < budget ? s->n_pending : budget; plan[np++] = (plan_item){ ids[i], take }; budget -= take; }
    }
    if (!np) { free(plan); return false; }
    uint32_t need = 0;
    int n_rows = 0;
    for (int i = 0; i < np; i++) { need += plan[i].take / KV_PAGE + 2; n_rows += plan[i].take; }
    make_room(e, need);
    if (n_rows > e->cap_rows) { e->cap_rows = n_rows * 2; e->rows = realloc(e->rows, sizeof(row_t) * e->cap_rows); }
    const seq_pages **pages = malloc(sizeof(seq_pages *) * np);
    int r = 0, n_logits = 0;
    for (int i = 0; i < np; i++) {
        seq_t *s = &e->seqs[plan[i].seq];
        int base = s->pages.len;
        for (int t = 0; t < plan[i].take; t++)
            if (!kv_reserve(&e->kv, &s->pages, base + t)) { fprintf(stderr, "engine: KV cache full (raise --budget-gb or lower concurrency)\n"); exit(1); }
        bool wants = plan[i].take == s->n_pending && s->generating && !s->done;
        for (int t = 0; t < plan[i].take; t++) {
            bool lg = wants && t == plan[i].take - 1;
            e->rows[r++] = (row_t){ .seq = i, .pos = base + t, .token = s->pending[t], .ctx_end = base + t, .logits = lg };
            n_logits += lg;
        }
        pages[i] = &s->pages;
    }
    size_t need_l = (size_t)n_logits * e->m->c.vocab;
    if (need_l > e->cap_logits) { e->cap_logits = need_l; e->logits = realloc(e->logits, sizeof(float) * need_l); }
    double t0 = now_secs();
    if (e->be->forward(e->be, e->m, &e->kv, pages, e->rows, n_rows, e->logits) != 0) { fprintf(stderr, "engine: forward failed\n"); exit(1); }
    double t1 = now_secs();
    e->st.forward_secs += t1 - t0;
    e->st.steps++;
    e->st.decode_rows += n_decode;
    e->st.prefill_rows += n_rows - n_decode;
    int li = 0;
    for (int i = 0; i < np; i++) {
        seq_t *s = &e->seqs[plan[i].seq];
        int take = plan[i].take;
        bool finishes = take == s->n_pending;
        s->pages.len += take;
        append(&s->tokens, &s->n_tokens, &s->cap_tokens, s->pending, take);
        memmove(s->pending, s->pending + take, sizeof(int32_t) * (s->n_pending - take));
        s->n_pending -= take;
        if (finishes && s->generating && !s->done) {
            int32_t tok = sample(e->logits + (size_t)li++ * e->m->c.vocab, e->m->c.vocab, &s->p, &s->rng);
            append(&s->gen, &s->n_gen, &s->cap_gen, &tok, 1);
            if (s->n_gen == 1) s->t_first = t1;
            e->st.gen_tokens++;
            if ((!s->p.ignore_eos && tok_is_stop(e->tok, tok)) || s->n_gen >= s->p.max_new) s->done = true;
            else append(&s->pending, &s->n_pending, &s->cap_pending, &tok, 1);
        }
        kv_trim_window(&e->kv, &s->pages);
    }
    free(pages);
    free(plan);
    return true;
}

void engine_generate(engine *e, const int *ids, int n) {
    int *active = malloc(sizeof(int) * (n ? n : 1));
    for (;;) {
        int k = 0;
        for (int i = 0; i < n; i++) if (!e->seqs[ids[i]].done) active[k++] = ids[i];
        if (!k || !engine_step(e, active, k)) break;
    }
    free(active);
}
