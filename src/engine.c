#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

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

static bool swap_in(engine *e, int id);
static void swap_discard(engine *e, seq_t *s);

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
    free(e->amax);
    free(e->dctx);
    ng_global_free(&e->ng);
    free(e);
}

bool engine_set_drafter(engine *e, model *dm, size_t kv_budget, int spec_k) {
    if (!e->be->draft || !e->be->attach || !e->be->argmax_ok) { fprintf(stderr, "engine: backend %s cannot run a drafter\n", e->be->name); return false; }
    if (dm->block_size > 32 || dm->n_target > 8 || dm->c.d != e->m->c.d) { fprintf(stderr, "engine: drafter does not match the target\n"); return false; }
    if (e->be->attach(e->be, dm)) return false;
    const model_cfg *c = &dm->c;
    kv_init(&e->dkv, c->n_full, 0, c->n_kv, c->hd, 0, kv_budget, e->be->kv_elem, be_alloc, e->be);
    e->draft = dm;
    e->spec_k = spec_k > 0 && spec_k < dm->block_size ? spec_k : dm->block_size - 1;
    if (!e->spec_rows) e->spec_rows = 32;
    e->be->n_cap = dm->n_target;
    memcpy(e->be->cap_layers, dm->target_layers, sizeof(int) * dm->n_target);
    return true;
}

static int new_seq(engine *e) {
    int id = -1;
    for (int i = 0; i < e->n_seqs && id < 0; i++) if (!e->seqs[i].live) id = i;
    if (id < 0) { e->seqs = realloc(e->seqs, sizeof(seq_t) * (e->n_seqs + 1)); id = e->n_seqs++; }
    memset(&e->seqs[id], 0, sizeof(seq_t));
    e->seqs[id].live = true;
    e->seqs[id].ng_gen = ++e->ng_gen;
    return id;
}

void engine_set_ngram(engine *e, bool on) {
    e->ngram = on;
    if (on && !e->ng.t) ng_global_init(&e->ng, 20);
    if (!e->spec_rows) e->spec_rows = 32;
    if (!e->spec_k) e->spec_k = 15;
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
    if (e->draft) s->dpages = kv_fork_at(&e->dkv, &p->dpages, len < p->dpages.len ? len : p->dpages.len);
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
    int best = -1, best_len = 0, lead = -1, lead_len = 0;
    for (int i = 0; i < e->n_seqs; i++) {
        seq_t *s = &e->seqs[i];
        if (!s->live) continue;
        int l = 0;
        while (l < s->n_tokens && l < n && s->tokens[l] == prompt[l]) l++;
        int w = l; // match continuing into the queued (not yet written) prompt
        if (l == s->n_tokens && !s->wait_len) while (w < s->n_tokens + s->n_pending && w < n && s->pending[w - s->n_tokens] == prompt[w]) w++;
        if (l > n - 1) l = n - 1; // keep one token to produce logits from
        if (w > n - 1) w = n - 1;
        bool forkable = s->swapped ? l <= s->swap_len : kv_can_fork_at(&e->kv, &s->pages, l);
        if (l >= e->min_share && l > best_len && forkable) { best = i; best_len = l; }
        if (w > lead_len) { lead = i; lead_len = w; }
    }
    if (lead >= 0 && lead_len >= best_len + e->min_share) { // worth waiting for a leader still prefilling
        int id = engine_create_fresh(e, prompt, n);
        e->seqs[id].wait_for = lead;
        e->seqs[id].wait_len = lead_len;
        return id;
    }
    if (best >= 0 && e->seqs[best].swapped && (!swap_in(e, best) || !kv_can_fork_at(&e->kv, &e->seqs[best].pages, best_len))) best = -1;
    if (best < 0) return engine_create_fresh(e, prompt, n);
    touch(e, best);
    int id = engine_fork_at(e, best, best_len, prompt + best_len, n - best_len);
    // A session's next turn: the new sequence holds (shares) all of a retired one,
    // so the retired copy is superseded; dropping it keeps memory and swap for
    // live state (other agents can still match the prefix in the new sequence).
    if (e->seqs[best].cached && best_len + 64 >= e->seqs[best].n_tokens) engine_release(e, best);
    return id;
}

// A waiting sequence forks its leader once the shared prefix is written; it
// gives up (and prefills itself) if the leader is gone or cannot be forked.
static void resolve_wait(engine *e, int id, bool force) {
    seq_t *s = &e->seqs[id];
    if (!s->wait_len) return;
    seq_t *l = &e->seqs[s->wait_for];
    int len = s->wait_len;
    bool same = l->live && !l->swapped && l->n_tokens >= len && !memcmp(l->tokens, s->pending, sizeof(int32_t) * len);
    if (same && kv_can_fork_at(&e->kv, &l->pages, len)) {
        s->pages = kv_fork_at(&e->kv, &l->pages, len);
        if (e->draft) s->dpages = kv_fork_at(&e->dkv, &l->dpages, len < l->dpages.len ? len : l->dpages.len);
        append(&s->tokens, &s->n_tokens, &s->cap_tokens, l->tokens, len);
        memmove(s->pending, s->pending + len, sizeof(int32_t) * (s->n_pending - len));
        s->n_pending -= len;
        e->st.shared_tokens += len;
        e->st.deduped_tokens += len;
        s->wait_len = 0;
    } else if (force || !l->live || l->swapped || l->n_tokens >= len || l->wait_len) {
        s->wait_len = 0;
    }
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
    if (e->draft) kv_free_seq(&e->dkv, &s->dpages);
    swap_discard(e, s);
    ng_seq_free(&s->ng);
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

// ---- KV swap (agents' KV is state, not a cache: restoring beats recomputing)

static bool swap_out(engine *e, int id) {
    seq_t *s = &e->seqs[id];
    if (!e->swap_dir || s->swapped || !s->pages.n_pages) return false;
    const kv_cache *kv = &e->kv;
    size_t fb = kv_page_bytes(&kv->full), wb = kv->window ? kv_page_bytes(&kv->swa) : 0, size = 0;
    uint8_t *has = malloc(s->pages.n_pages);
    for (int i = 0; i < s->pages.n_pages; i++) {
        has[i] = kv->window && s->pages.swa[i] != KV_NONE;
        size += fb + (has[i] ? wb : 0);
    }
    if (e->swap_used + size > e->swap_limit) { free(has); return false; }
    char path[512];
    snprintf(path, sizeof path, "%s/cinfer-swap-%d-%d-%d.kv", e->swap_dir, (int)getpid(), id, s->ng_gen);
    double t0 = now_secs();
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) { free(has); return false; }
    off_t off = 0;
    bool ok = true;
    for (int i = 0; i < s->pages.n_pages && ok; i++) {
        ok = pwrite(fd, kv->full.data + (size_t)s->pages.full[i] * fb, fb, off) == (ssize_t)fb;
        off += fb;
        if (has[i] && ok) { ok = pwrite(fd, kv->swa.data + (size_t)s->pages.swa[i] * wb, wb, off) == (ssize_t)wb; off += wb; }
    }
    close(fd);
    if (!ok) { unlink(path); free(has); return false; }
    s->swap_path = strdup(path);
    s->swap_pages = s->pages.n_pages;
    s->swap_len = s->pages.len;
    s->swap_has_swa = has;
    s->swap_size = size;
    s->swapped = true;
    s->n_drafts = 0;
    kv_free_seq(&e->kv, &s->pages);
    if (e->draft) kv_free_seq(&e->dkv, &s->dpages); // drafter context is dropped (no drafts for this sequence)
    e->swap_used += size;
    e->st.swap_outs++;
    e->st.swap_secs += now_secs() - t0;
    return true;
}

static void swap_discard(engine *e, seq_t *s) {
    if (!s->swapped) return;
    unlink(s->swap_path);
    free(s->swap_path);
    free(s->swap_has_swa);
    e->swap_used -= s->swap_size;
    s->swap_path = NULL;
    s->swap_has_swa = NULL;
    s->swapped = false;
}

static void make_room(engine *e, uint32_t pages);

static bool swap_in(engine *e, int id) {
    seq_t *s = &e->seqs[id];
    if (!s->swapped) return true;
    uint32_t nw = 0;
    for (int i = 0; i < s->swap_pages; i++) nw += s->swap_has_swa[i];
    make_room(e, (uint32_t)s->swap_pages > nw ? (uint32_t)s->swap_pages : nw);
    seq_pages pg;
    if (!kv_alloc_pages(&e->kv, &pg, s->swap_pages, s->swap_has_swa, s->swap_len)) return false;
    const kv_cache *kv = &e->kv;
    size_t fb = kv_page_bytes(&kv->full), wb = kv->window ? kv_page_bytes(&kv->swa) : 0;
    double t0 = now_secs();
    int fd = open(s->swap_path, O_RDONLY);
    off_t off = 0;
    bool ok = fd >= 0;
    for (int i = 0; i < s->swap_pages && ok; i++) {
        ok = pread(fd, kv->full.data + (size_t)pg.full[i] * fb, fb, off) == (ssize_t)fb;
        off += fb;
        if (s->swap_has_swa[i] && ok) { ok = pread(fd, kv->swa.data + (size_t)pg.swa[i] * wb, wb, off) == (ssize_t)wb; off += wb; }
    }
    if (fd >= 0) close(fd);
    if (!ok) { kv_free_seq(&e->kv, &pg); return false; }
    swap_discard(e, s);
    s->pages = pg;
    e->st.swap_ins++;
    e->st.swap_secs += now_secs() - t0;
    return true;
}

// Make both pools able to take `pages` more pages: retired sequences go to swap
// (or are released if swap is off or full), oldest first.
static void make_room(engine *e, uint32_t pages) {
    for (int i = 0; i < e->n_cached && (e->kv.full.n_free < pages || (e->kv.window && e->kv.swa.n_free < pages));) {
        int id = e->cached[i];
        if (e->seqs[id].swapped) { i++; continue; }
        if (swap_out(e, id)) { i++; continue; }
        engine_release(e, id); // removes it from cached[]
    }
}

typedef struct { int seq, take, spec; } plan_item; // spec: draft rows after the decode row

static bool fits(const engine *e, uint32_t pages) {
    return e->kv.full.n_free >= pages && (!e->kv.window || e->kv.swa.n_free >= pages);
}

// Free a sequence's KV and queue all of its tokens to be recomputed later
// (vLLM-style recompute preemption): its generated output is kept.
static void preempt(engine *e, int id) {
    seq_t *s = &e->seqs[id];
    int n = s->n_tokens + s->n_pending;
    int32_t *all = malloc(sizeof(int32_t) * (n + 16));
    memcpy(all, s->tokens, sizeof(int32_t) * s->n_tokens);
    memcpy(all + s->n_tokens, s->pending, sizeof(int32_t) * s->n_pending);
    kv_free_seq(&e->kv, &s->pages);
    if (e->draft) kv_free_seq(&e->dkv, &s->dpages);
    free(s->pending);
    s->pending = all;
    s->n_pending = n;
    s->cap_pending = n + 16;
    s->n_tokens = 0;
    s->n_drafts = 0;
    e->st.preemptions++;
}

static void add_ctx(engine *e, int seq, int pos, int src) {
    if (e->n_dctx == e->cap_dctx) { e->cap_dctx = e->cap_dctx ? e->cap_dctx * 2 : 256; e->dctx = realloc(e->dctx, sizeof(draft_row) * e->cap_dctx); }
    e->dctx[e->n_dctx++] = (draft_row){ seq, pos, src, pos };
}

// Drafter step: ingest the context rows of the last forward (every live
// sequence), then draft a block for each greedy decoding sequence among ids
// whose drafter context is complete.
static void draft_phase(engine *e, const int *ids, int n) {
    const model *dm = e->draft;
    int B = dm->block_size;
    int *loc = malloc(sizeof(int) * (e->n_seqs ? e->n_seqs : 1)), nl = 0;
    int *glob = malloc(sizeof(int) * (e->n_seqs ? e->n_seqs : 1));
    int *nctx = calloc(e->n_seqs ? e->n_seqs : 1, sizeof(int));
    for (int i = 0; i < e->n_seqs; i++) loc[i] = -1;
    draft_row *ctx = malloc(sizeof(draft_row) * (e->n_dctx ? e->n_dctx : 1)), *blk = malloc(sizeof(draft_row) * (n * B + 1));
    int nc = 0, nb = 0;
    // Drafts per sequence so the step stays near spec_rows rows (the batch size
    // the matmuls run at nearly flat cost); none at all when that is < 1.
    int nd = 0;
    for (int i = 0; i < n; i++) {
        const seq_t *s = &e->seqs[ids[i]];
        nd += s->generating && !s->done && s->n_pending == 1 && !s->wait_len;
    }
    int kmax = nd ? e->spec_rows / nd - 1 : 0;
    if (kmax > e->spec_k) kmax = e->spec_k;
    // The drafter denoises only the positions that will be verified: a block of
    // the anchor + kmax masks (DFlash trains with block_size; CINFER_FULL_BLOCK=1
    // keeps the full block).
    static int full = -1;
    if (full < 0) full = getenv("CINFER_FULL_BLOCK") != NULL;
    if (!full && kmax >= 1) B = kmax + 1;
    bool *bad = calloc(e->n_seqs ? e->n_seqs : 1, 1);
    for (int i = 0; i < e->n_dctx; i++) {
        draft_row r = e->dctx[i];
        seq_t *s = &e->seqs[r.seq];
        if (!s->live || bad[r.seq] || r.pos != s->dpages.len + nctx[r.seq] || !kv_reserve(&e->dkv, &s->dpages, r.pos)) { bad[r.seq] = true; continue; }
        if (loc[r.seq] < 0) { loc[r.seq] = nl; glob[nl++] = r.seq; }
        nctx[r.seq]++;
        r.seq = loc[r.seq];
        ctx[nc++] = r;
    }
    e->n_dctx = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        seq_t *s = &e->seqs[id];
        s->n_drafts = 0;
        int k = kmax < s->p.max_new - s->n_gen - 1 ? kmax : s->p.max_new - s->n_gen - 1;
        if (!s->generating || s->done || s->n_pending != 1 || s->wait_len || s->swapped || bad[id] || k <= 0) continue;
        if (s->dpages.len + nctx[id] != s->n_tokens) continue; // drafter context incomplete (e.g. forked mid-step)
        bool ok = true;
        for (int j = 0; j < B && ok; j++) ok = kv_reserve(&e->dkv, &s->dpages, s->n_tokens + j);
        if (!ok) continue;
        if (loc[id] < 0) { loc[id] = nl; glob[nl++] = id; }
        for (int j = 0; j < B; j++) blk[nb++] = (draft_row){ loc[id], s->n_tokens + j, j ? dm->mask_token : s->pending[0], s->n_tokens + B - 1 };
        s->n_drafts = k;
    }
    if (nc || nb) {
        const seq_pages **pg = malloc(sizeof(seq_pages *) * (nl ? nl : 1));
        for (int i = 0; i < nl; i++) pg[i] = &e->seqs[glob[i]].dpages;
        int32_t *out = malloc(sizeof(int32_t) * (nb / B * (B - 1) + 1));
        double t0 = now_secs();
        if (e->be->draft(e->be, dm, e->m, &e->dkv, pg, ctx, nc, blk, nb, B, out)) { fprintf(stderr, "engine: drafter failed\n"); exit(1); }
        e->st.draft_secs += now_secs() - t0;
        for (int b = 0; b < nb / B; b++) {
            seq_t *s = &e->seqs[glob[blk[b * B].seq]];
            memcpy(s->drafts, out + b * (B - 1), sizeof(int32_t) * s->n_drafts);
        }
        free(out);
        free(pg);
    }
    for (int i = 0; i < nl; i++) { // context is now in the drafter KV; drop the block's temporary slots
        seq_t *s = &e->seqs[glob[i]];
        s->dpages.len += nctx[glob[i]];
        kv_truncate(&e->dkv, &s->dpages, s->dpages.len);
    }
    free(loc); free(glob); free(nctx); free(ctx); free(blk); free(bad);
}

static ng_view seq_view(void *u, int id) {
    engine *e = u;
    if (id < 0 || id >= e->n_seqs || !e->seqs[id].live) return (ng_view){ NULL, 0, 0 };
    return (ng_view){ e->seqs[id].tokens, e->seqs[id].n_tokens, e->seqs[id].ng_gen };
}

typedef struct { int id, k, match; } ng_cand;

// Repeated-text drafts for decoding sequences: the step's spare rows (spec_rows
// minus one row per decoding sequence) go to the longest matches first. Under a
// drafter, a repeated-text match of >= NG_TRUST tokens replaces its drafts.
#define NG_TRUST 8
static void ngram_phase(engine *e, const int *ids, int n) {
    ng_cand *c = malloc(sizeof(ng_cand) * (n ? n : 1));
    int nc = 0, nd = 0;
    int32_t buf[32];
    for (int i = 0; i < n; i++) {
        seq_t *s = &e->seqs[ids[i]];
        bool decoding = s->generating && !s->done && s->n_pending == 1 && !s->wait_len && !s->swapped;
        nd += decoding;
        if (!decoding) { // keep prompts indexed too: they are draft sources for everyone
            if (s->n_tokens) ng_update(&s->ng, &e->ng, ids[i], s->ng_gen, s->tokens, s->n_tokens);
            continue;
        }
        // context = written tokens + the pending (sampled, not yet written) token
        if (s->n_tokens + 1 > s->cap_tokens) { s->cap_tokens = (s->n_tokens + 1) * 2 + 16; s->tokens = realloc(s->tokens, sizeof(int32_t) * s->cap_tokens); }
        s->tokens[s->n_tokens] = s->pending[0];
        int nctx = s->n_tokens + 1;
        ng_update(&s->ng, &e->ng, ids[i], s->ng_gen, s->tokens, nctx);
        int max = s->p.max_new - s->n_gen - 1;
        if (max > e->spec_k) max = e->spec_k;
        if (max > 31) max = 31;
        int m = 0, k = ng_draft(&s->ng, &e->ng, ids[i], s->tokens, nctx, seq_view, e, buf, max, &m);
        if (e->draft && s->n_drafts && m < NG_TRUST) continue; // keep the drafter's proposals
        if (k > 0) {
            memcpy(s->drafts, buf, sizeof(int32_t) * k);
            c[nc++] = (ng_cand){ ids[i], k, m };
            s->n_drafts = 0; // set below within the budget
        }
    }
    int budget = e->spec_rows - nd;
    for (int i = 1; i < nc; i++) { // longest match first
        ng_cand v = c[i];
        int j = i - 1;
        while (j >= 0 && c[j].match < v.match) { c[j + 1] = c[j]; j--; }
        c[j + 1] = v;
    }
    for (int i = 0; i < nc; i++) {
        seq_t *s = &e->seqs[c[i].id];
        int k = c[i].k < budget ? c[i].k : budget;
        s->n_drafts = k > 0 ? k : 0;
        s->ng_drafts = s->n_drafts;
        if (k > 0) budget -= k;
    }
    free(c);
}

static void emit(engine *e, seq_t *s, int32_t tok, double t) {
    append(&s->gen, &s->n_gen, &s->cap_gen, &tok, 1);
    if (s->n_gen == 1) s->t_first = t;
    e->st.gen_tokens++;
    bool stop = false;
    if (s->p.n_stop) for (int i = 0; i < s->p.n_stop; i++) stop |= s->p.stop[i] == tok;
    else stop = tok_is_stop(e->tok, tok);
    if ((!s->p.ignore_eos && stop) || s->n_gen >= s->p.max_new) s->done = true;
}

bool engine_step(engine *e, const int *ids, int n) {
    if (e->draft) draft_phase(e, ids, n);
    else for (int i = 0; i < n; i++) e->seqs[ids[i]].n_drafts = 0;
    for (int i = 0; i < n; i++) e->seqs[ids[i]].ng_drafts = 0;
    if (e->ngram) ngram_phase(e, ids, n);
    plan_item *plan = malloc(sizeof(plan_item) * (n ? n : 1));
    int np = 0, budget = e->row_budget, n_decode = 0;
    for (int i = 0; i < n; i++) resolve_wait(e, ids[i], false);
    // paused (swapped-out) sequences come back, oldest first, while their pages fit
    for (int i = 0; i < n; i++) {
        seq_t *s = &e->seqs[ids[i]];
        if (s->swapped && !s->cached && !swap_in(e, ids[i])) break;
    }
    for (int pass = 0; pass < 2 && !np; pass++) {
        if (pass) for (int i = 0; i < n; i++) resolve_wait(e, ids[i], true); // leaders not in this step: stop waiting
        budget = e->row_budget;
        for (int i = 0; i < n && budget > 0; i++) { // decode rows (plus their drafts) first; drafts do not eat the prefill budget
            seq_t *s = &e->seqs[ids[i]];
            if (s->n_pending == 1 && !s->wait_len && !s->swapped) {
                int k = s->generating && !s->done ? s->n_drafts : 0;
                plan[np++] = (plan_item){ ids[i], 1 + k, k };
                budget--;
                n_decode++;
            }
        }
        for (int i = 0; i < n && budget > 0; i++) {
            seq_t *s = &e->seqs[ids[i]];
            if (s->n_pending > 1 && !s->wait_len && !s->swapped) { int take = s->n_pending < budget ? s->n_pending : budget; plan[np++] = (plan_item){ ids[i], take, 0 }; budget -= take; }
        }
    }
    if (!np) { free(plan); return false; }
    // The GEMMs compute whole 32-row tiles: rather than leave the last tile part
    // empty (decode rows and drafts on top of a full prefill chunk), fill it with
    // more pending prefill rows (free compute).
    int planned = 0;
    for (int i = 0; i < np; i++) planned += plan[i].take;
    for (int i = 0, extra = planned > 16 ? (planned + 31) / 32 * 32 - planned : 0; i < np && extra > 0; i++) {
        seq_t *s = &e->seqs[plan[i].seq];
        if (plan[i].spec || s->n_pending <= 1) continue;
        int more = s->n_pending - plan[i].take < extra ? s->n_pending - plan[i].take : extra;
        plan[i].take += more;
        extra -= more;
    }
    uint32_t need = 0;
    int n_rows = 0;
    for (int i = 0; i < np; i++) need += plan[i].take / KV_PAGE + 2;
    make_room(e, need);
    // Memory pressure: postpone the last planned items; if one item alone does not
    // fit, preempt the most recently started other sequences.
    while (!fits(e, need) && np > 1) { np--; need -= plan[np].take / KV_PAGE + 2; }
    while (!fits(e, need)) {
        int victim = -1;
        for (int i = 0; i < e->n_seqs; i++) {
            seq_t *s = &e->seqs[i];
            bool planned = false;
            for (int j = 0; j < np; j++) planned |= plan[j].seq == i;
            if (!s->live || s->cached || s->swapped || !s->pages.n_pages || planned) continue;
            if (victim < 0 || s->t_start > e->seqs[victim].t_start) victim = i;
        }
        if (victim < 0) { fprintf(stderr, "engine: KV cache too small for one sequence (raise --budget-gb)\n"); exit(1); }
        if (!swap_out(e, victim)) preempt(e, victim);
    }
    for (int i = 0; i < np; i++) n_rows += plan[i].take;
    if (n_rows > e->cap_rows) { e->cap_rows = n_rows * 2; e->rows = realloc(e->rows, sizeof(row_t) * e->cap_rows); }
    const seq_pages **pages = malloc(sizeof(seq_pages *) * np);
    int r = 0, n_logits = 0;
    bool need_logits = !e->be->argmax_ok;
    for (int i = 0; i < np; i++) {
        seq_t *s = &e->seqs[plan[i].seq];
        int base = s->pages.len;
        for (int t = 0; t < plan[i].take; t++)
            if (!kv_reserve(&e->kv, &s->pages, base + t)) { fprintf(stderr, "engine: KV cache full (raise --budget-gb or lower concurrency)\n"); exit(1); }
        bool wants = (plan[i].spec || plan[i].take == s->n_pending) && s->generating && !s->done;
        if (wants && s->p.temperature > 0) need_logits = true;
        for (int t = 0; t < plan[i].take; t++) {
            bool lg = wants && (plan[i].spec || t == plan[i].take - 1);
            int32_t tok = t == 0 || !plan[i].spec ? s->pending[t] : s->drafts[t - 1];
            // decode / verify items carry the last sampled token (plus drafts); a lone
            // pending token before anything was sampled is a prompt tail
            bool input = !plan[i].spec && !(s->n_pending == 1 && s->n_gen > 0);
            e->rows[r++] = (row_t){ .seq = i, .pos = base + t, .token = tok, .ctx_end = base + t, .logits = lg, .input = input };
            n_logits += lg;
        }
        pages[i] = &s->pages;
    }
    size_t need_l = need_logits ? (size_t)n_logits * e->m->c.vocab : 0;
    if (need_l > e->cap_logits) { e->cap_logits = need_l; e->logits = realloc(e->logits, sizeof(float) * need_l); }
    if (n_logits > e->cap_amax) { e->cap_amax = n_logits * 2; e->amax = realloc(e->amax, sizeof(int32_t) * e->cap_amax); }
    e->be->argmax = e->be->argmax_ok ? e->amax : NULL;
    double t0 = now_secs();
    if (e->be->forward(e->be, e->m, &e->kv, pages, e->rows, n_rows, need_logits ? e->logits : NULL) != 0) { fprintf(stderr, "engine: forward failed\n"); exit(1); }
    double t1 = now_secs();
    e->st.forward_secs += t1 - t0;
    e->st.steps++;
    int n_spec = 0;
    for (int i = 0; i < np; i++) n_spec += plan[i].spec;
    e->st.decode_rows += n_decode + n_spec;
    e->st.prefill_rows += n_rows - n_decode - n_spec;
    int li = 0;
    r = 0;
    for (int i = 0; i < np; i++) {
        seq_t *s = &e->seqs[plan[i].seq];
        int take = plan[i].take, base = s->pages.len, r0 = r;
        r += take;
        if (plan[i].spec) {
            // verify: the target's choice after the anchor and each accepted draft (argmax for
            // greedy rows; the exact rejection rule for sampled rows, see sample_verify)
            int k = plan[i].spec, acc = 0;
            int32_t a[33];
            bool greedy = s->p.temperature <= 0 && !need_logits;
            for (int t = 0; t <= k; t++) {
                bool ok = false;
                if (greedy) a[t] = e->amax[li + t], ok = t < k && a[t] == s->drafts[t];
                else if (t < k) a[t] = sample_verify(e->logits + (size_t)(li + t) * e->m->c.vocab, e->m->c.vocab, &s->p, &s->rng, s->drafts[t], &ok);
                else a[t] = sample(e->logits + (size_t)(li + t) * e->m->c.vocab, e->m->c.vocab, &s->p, &s->rng);
                if (!ok) break;
                acc++;
            }
            li += take;
            e->st.spec_steps++;
            e->st.spec_drafted += k;
            e->st.spec_accepted += acc;
            if (s->ng_drafts) { e->st.ng_steps++; e->st.ng_drafted += k; e->st.ng_accepted += acc; }
            int keep = 1 + acc; // anchor + accepted drafts stay in the KV
            s->pages.len = base + keep;
            kv_truncate(&e->kv, &s->pages, s->pages.len);
            append(&s->tokens, &s->n_tokens, &s->cap_tokens, s->pending, 1);
            append(&s->tokens, &s->n_tokens, &s->cap_tokens, s->drafts, acc);
            s->n_pending = 0;
            if (e->draft) for (int t = 0; t < keep; t++) add_ctx(e, plan[i].seq, base + t, r0 + t);
            for (int t = 0; t <= acc && !s->done; t++) emit(e, s, a[t], t1); // accepted drafts equal a[0..acc-1], then the bonus token
            if (!s->done) append(&s->pending, &s->n_pending, &s->cap_pending, &a[acc], 1);
            kv_trim_window(&e->kv, &s->pages);
            continue;
        }
        bool finishes = take == s->n_pending;
        s->pages.len += take;
        append(&s->tokens, &s->n_tokens, &s->cap_tokens, s->pending, take);
        memmove(s->pending, s->pending + take, sizeof(int32_t) * (s->n_pending - take));
        s->n_pending -= take;
        if (e->draft) for (int t = 0; t < take; t++) add_ctx(e, plan[i].seq, base + t, r0 + t);
        if (finishes && s->generating && !s->done) {
            int32_t tok = need_logits ? sample(e->logits + (size_t)li * e->m->c.vocab, e->m->c.vocab, &s->p, &s->rng) : e->amax[li];
            li++;
            emit(e, s, tok, t1);
            if (!s->done) append(&s->pending, &s->n_pending, &s->cap_pending, &tok, 1);
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
