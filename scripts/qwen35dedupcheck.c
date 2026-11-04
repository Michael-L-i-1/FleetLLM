// Scheduler checks with a deterministic recurrent reference backend. No weights,
// Python packages or GPU required. Each token updates a sequence-owned state.
//   make NO_METAL=1 BUILD=/tmp/fleet-dedup-check /tmp/fleet-dedup-check/libfleetllm.a
//   cc -O1 -g -fsanitize=address,undefined -Isrc scripts/qwen35dedupcheck.c \
//      /tmp/fleet-dedup-check/libfleetllm.a -lpthread -lm -o /tmp/qwen35dedupcheck
#include "engine.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 32
#define OUTPUT 12

static uint32_t update(uint32_t state, int32_t token, int pos) { return state * 31u + (uint32_t)token + (uint32_t)pos * 17u; }
static void *allocate(backend *b, size_t bytes, void **handle) { return *handle = calloc(1, bytes ? bytes : 1); }
static int forward(backend *b, const model *m, kv_cache *kv, const seq_pages *const *pages, const row_t *rows, int n, float *logits) {
    assert(backend_recurrent_rows_valid(m, pages, rows, n));
    int out = 0;
    for (int i = 0; i < n; i++) {
        seq_pages *p = (seq_pages *)pages[rows[i].seq];
        uint32_t state;
        memcpy(&state, p->recurrent, sizeof state);
        state = update(state, rows[i].token, rows[i].pos);
        memcpy(p->recurrent, &state, sizeof state);
        p->recurrent_len = rows[i].pos + 1;
        if (rows[i].logits) {
            int best = (int)(state % VOCAB);
            if (b->argmax) b->argmax[out] = best;
            if (logits)
                for (int v = 0; v < VOCAB; v++)
                    logits[(size_t)out * VOCAB + v] = -(float)abs(v - best);
            out++;
        }
    }
    return 0;
}
static model fixture(void) {
    model m = {0};
    m.c = (model_cfg){
        .arch = ARCH_QWEN35, .n_layer = 1, .n_linear = 1, .n_full = 1, .n_head = 1, .n_kv = 1, .hd = 4, .d = 4, .vocab = VOCAB, .recurrent_floats = 4};
    return m;
}
static backend compute(void) { return (backend){.name = "deterministic recurrent", .kv_elem = 4, .alloc_kv = allocate, .forward = forward, .argmax_ok = true}; }
static sample_params params(void) {
    sample_params p = sample_defaults();
    p.temperature = 0;
    p.max_new = OUTPUT;
    p.ignore_eos = true;
    p.n_stop = 1;
    p.stop[0] = -1;
    return p;
}
static void release_engine(engine *e) {
    void *full = e->kv.full.data, *swa = e->kv.swa.data;
    void *fr = e->kv.full.refcnt, *ff = e->kv.full.free_list, *sr = e->kv.swa.refcnt, *sf = e->kv.swa.free_list;
    engine_free(e);
    free(full);
    free(swa);
    free(fr);
    free(ff);
    free(sr);
    free(sf);
}
static void expected(const int32_t *prompt, int n, int32_t *out) {
    uint32_t h = 0;
    for (int i = 0; i < n; i++)
        h = update(h, prompt[i], i);
    for (int i = 0; i < OUTPUT; i++) {
        out[i] = (int32_t)(h % VOCAB);
        h = update(h, out[i], n + i);
    }
}
static void finish(engine *e, const int *ids, int n) {
    int steps = 0;
    while (1) {
        int pending = 0, active[16];
        assert(n <= 16);
        for (int i = 0; i < n; i++)
            if (!e->seqs[ids[i]].done) active[pending++] = ids[i];
        if (!pending) break;
        assert(++steps < 2000);
        assert(engine_step(e, active, pending));
    }
}
static void check_outputs(engine *e, const int *ids, int n, int32_t prompts[][100], const int *lengths) {
    for (int i = 0; i < n; i++) {
        int32_t want[OUTPUT];
        expected(prompts[i], lengths[i], want);
        assert(e->seqs[ids[i]].n_gen == OUTPUT);
        assert(!memcmp(want, e->seqs[ids[i]].gen, sizeof want));
    }
}
static long fanout(bool dedup, bool checkpoints, int slots) {
    model m = fixture();
    backend b = compute();
    engine *e = engine_new(&m, &b, NULL, 2u << 20);
    e->row_budget = 17;
    e->recurrent_prefix_dedup = dedup;
    e->prompt_checkpoints = checkpoints;
    if (slots) e->kv.recurrent_limit = slots;
    int32_t p[4][100];
    int lens[] = {100, 82, 95, 60}, prefix[] = {100, 41, 73, 41}, ids[4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < lens[i]; j++)
            p[i][j] = j < prefix[i] ? (j * 7 + 3) % VOCAB : (j * 11 + 13 + i) % VOCAB;
        // Explicitly ensure the first differing token really differs.
        if (i) p[i][prefix[i]] = (p[0][prefix[i]] + i) % VOCAB;
        ids[i] = engine_create(e, p[i], lens[i]);
        engine_start(e, ids[i], params());
        if (checkpoints) e->seqs[ids[i]].checkpoint_at = prefix[i] > 20 ? 20 : 0;
    }
    if (dedup && (!slots || slots >= 2)) assert(e->seqs[ids[1]].wait_len == 41 && e->seqs[ids[2]].wait_len == 73);
    finish(e, ids, 4);
    check_outputs(e, ids, 4, p, lens);
    long prefill = e->st.prefill_rows, shared = e->st.deduped_tokens;
    if (dedup && !slots) assert(shared == 155);
    if (!dedup) assert(shared == 0);
    printf("fanout dedup=%d checkpoints=%d slots=%d: prefill=%ld shared=%ld, outputs match fresh mathematical reference\n", dedup, checkpoints,
           e->kv.recurrent_limit, prefill, shared);
    long rows = e->st.prefill_rows + e->st.decode_rows;
    release_engine(e);
    return rows;
}
static void abandoned_leader(bool recycle, bool retire, bool absent_from_step) {
    model m = fixture();
    backend b = compute();
    engine *e = engine_new(&m, &b, NULL, 2u << 20);
    e->recurrent_prefix_dedup = true;
    e->row_budget = 17;
    int32_t prompt[100];
    for (int i = 0; i < 100; i++)
        prompt[i] = (i * 3 + 7) % VOCAB;
    int a = engine_create(e, prompt, 100);
    engine_start(e, a, params());
    int child = engine_create(e, prompt, 100);
    engine_start(e, child, params());
    assert(e->seqs[child].wait_len == 99);
    if (retire)
        engine_retire(e, a);
    else if (!absent_from_step) {
        engine_release(e, a);
        if (recycle) {
            int32_t other[] = {9, 8, 7};
            int replacement = engine_create_fresh(e, other, 3);
            assert(replacement == a);
            engine_start(e, replacement, params());
        }
    }
    finish(e, &child, 1);
    int32_t out[OUTPUT];
    expected(prompt, 100, out);
    assert(!memcmp(out, e->seqs[child].gen, sizeof out));
    assert(e->seqs[child].wait_len == 0);
    printf("abandoned leader recycle=%d retired=%d off-step=%d: exact fresh fallback PASS\n", recycle, retire, absent_from_step);
    release_engine(e);
}
static void slot_exhaustion_at_boundary(void) {
    model m = fixture();
    backend b = compute();
    engine *e = engine_new(&m, &b, NULL, 2u << 20);
    e->recurrent_prefix_dedup = true;
    e->row_budget = 17;
    e->kv.recurrent_limit = 2;
    int32_t prompt[100];
    for (int i = 0; i < 100; i++)
        prompt[i] = (i * 3 + 7) % VOCAB;
    int leader = engine_create(e, prompt, 100);
    engine_start(e, leader, params());
    int child = engine_create(e, prompt, 100);
    engine_start(e, child, params());
    int32_t other[] = {5, 9, 11, 3};
    int occupied = engine_create_fresh(e, other, 4);
    engine_start(e, occupied, params());
    // Retain the other active state's slot while leader reaches the common endpoint.
    while (e->seqs[occupied].n_gen < 1)
        assert(engine_step(e, &occupied, 1));
    while (e->seqs[leader].n_tokens < 99)
        assert(engine_step(e, &leader, 1));
    assert(e->seqs[child].wait_len == 0 && e->seqs[child].n_tokens == 0);
    finish(e, &child, 1);
    int32_t out[OUTPUT];
    expected(prompt, 100, out);
    assert(!memcmp(out, e->seqs[child].gen, sizeof out));
    puts("state slot exhaustion at endpoint: no invalid clone, exact recomputation PASS");
    release_engine(e);
}
static void cancelled_waiter(void) {
    model m = fixture();
    backend b = compute();
    engine *e = engine_new(&m, &b, NULL, 2u << 20);
    e->recurrent_prefix_dedup = true;
    e->row_budget = 17;
    int32_t prompt[100], other[100];
    for (int i = 0; i < 100; i++)
        prompt[i] = other[i] = (i * 3 + 7) % VOCAB;
    other[41] = (prompt[41] + 1) % VOCAB;
    int leader = engine_create(e, prompt, 100);
    engine_start(e, leader, params());
    int child = engine_create(e, other, 100);
    engine_start(e, child, params());
    assert(e->seqs[child].wait_len == 41);
    engine_retire(e, child);
    assert(e->seqs[child].wait_len == 0);
    assert(engine_step(e, &leader, 1));
    assert(e->seqs[leader].n_tokens == 32);
    assert(engine_step(e, &leader, 1));
    assert(e->seqs[leader].n_tokens == 64);
    assert(!e->seqs[child].pages.recurrent && e->st.deduped_tokens == 0);
    finish(e, &leader, 1);
    puts("cancelled waiter: no stale boundary or state allocation PASS");
    release_engine(e);
}
static int released_handles;
static void release_fake_handle(void *p) {
    assert(p);
    released_handles++;
    free(p);
}

static void full_slot_state_move(void) {
    model m = fixture();
    backend b = compute();
    engine *e = engine_new(&m, &b, NULL, 2u << 20);
    e->row_budget = 17;
    int32_t prompt[101];
    for (int i = 0; i < 100; i++)
        prompt[i] = (i * 3 + 7) % VOCAB;
    prompt[100] = 19;
    int source = engine_create_fresh(e, prompt, 100);
    sample_params one = params();
    one.max_new = 1;
    engine_start(e, source, one);
    finish(e, &source, 1);
    float *state = e->seqs[source].pages.recurrent;
    void *handle = malloc(8);
    e->seqs[source].pages.recurrent_handle = handle;
    e->seqs[source].pages.release_recurrent_handle = release_fake_handle;
    released_handles = 0;
    int32_t sibling_token = 11;
    int sibling = engine_fork_at(e, source, 100, &sibling_token, 1);
    e->kv.recurrent_limit = 2;
    assert(e->kv.recurrent_used == 2);
    // A completed but unretired source still belongs to its caller.
    e->recurrent_state_move = true;
    int fresh = engine_create(e, prompt, 101);
    assert(!e->seqs[fresh].pages.recurrent && e->seqs[source].pages.recurrent == state);
    assert(e->st.state_moves == 0 && e->st.fullslot_match_misses == 1);
    engine_release(e, fresh);
    engine_retire(e, source);
    // Cached metadata alone must never authorize stealing an active sequence.
    e->seqs[source].generating = true;
    e->seqs[source].done = false;
    fresh = engine_create(e, prompt, 101);
    assert(!e->seqs[fresh].pages.recurrent && e->seqs[source].pages.recurrent == state);
    assert(e->st.state_moves == 0 && e->st.fullslot_match_misses == 2);
    engine_release(e, fresh);
    e->seqs[source].generating = false;
    e->seqs[source].done = true;
    assert(kv_has_fork_at(&e->kv, &e->seqs[source].pages, 100));
    assert(!kv_can_fork_at(&e->kv, &e->seqs[source].pages, 100));
    // Protect an endpoint another live waiter still references.
    int waiter = engine_create_fresh(e, prompt, 101);
    e->seqs[waiter].wait_for = source;
    e->seqs[waiter].wait_gen = e->seqs[source].ng_gen;
    e->seqs[waiter].wait_len = 100;
    fresh = engine_create(e, prompt, 101);
    assert(!e->seqs[fresh].pages.recurrent && e->seqs[source].live);
    assert(e->st.state_moves == 0 && e->st.fullslot_match_misses == 3);
    engine_release(e, fresh);
    engine_release(e, waiter);
    e->recurrent_state_move = false;
    uint32_t recompute_need = engine_need_pages(e, prompt, 101, OUTPUT);
    fresh = engine_create(e, prompt, 101);
    assert(!e->seqs[fresh].pages.recurrent && e->st.fullslot_match_misses == 4);
    engine_release(e, fresh);
    e->recurrent_state_move = true;
    uint32_t reuse_need = engine_need_pages(e, prompt, 101, OUTPUT);
    assert(reuse_need < recompute_need);
    uint32_t shared_page = e->seqs[sibling].pages.full[100 / KV_PAGE];
    assert(e->kv.full.refcnt[shared_page] == 2);
    int moved = engine_create(e, prompt, 101);
    assert(!e->seqs[source].live);
    assert(e->seqs[moved].pages.recurrent == state && e->seqs[moved].pages.recurrent_handle == handle);
    assert(e->seqs[moved].pages.release_recurrent_handle == release_fake_handle);
    assert(e->seqs[moved].n_tokens == 100 && e->seqs[moved].prefix_cached == 100);
    assert(e->st.state_moves == 1 && e->kv.recurrent_used == 2 && released_handles == 0);
    assert(e->kv.full.refcnt[shared_page] == 2);
    // A skipped checkpoint is counted without allocating or damaging moved state.
    e->prompt_checkpoints = true;
    e->seqs[moved].checkpoint_at = 101;
    engine_start(e, moved, params());
    finish(e, &moved, 1);
    int32_t out[OUTPUT];
    expected(prompt, 101, out);
    assert(!memcmp(out, e->seqs[moved].gen, sizeof out));
    assert(e->seqs[moved].pages.full[100 / KV_PAGE] != e->seqs[sibling].pages.full[100 / KV_PAGE]);
    assert(e->st.skipped_checkpoints == 1 && e->st.slot_preemptions == 0);
    engine_release(e, moved);
    assert(released_handles == 1 && e->kv.recurrent_used == 1);
    engine_release(e, sibling);
    assert(released_handles == 1 && e->kv.recurrent_used == 0);
    puts("full-slot move: exact continuation, protected owners/waiters, COW, one handle release, unchanged slots, counters PASS");
    release_engine(e);
}

static void seed_phase_cache(engine *e, int *endpoint, int *checkpoint) {
    int32_t prompt[60];
    for (int i = 0; i < 60; i++)
        prompt[i] = (i * 3 + 7) % VOCAB;
    e->prompt_checkpoints = true;
    *endpoint = engine_create_fresh(e, prompt, 60);
    e->seqs[*endpoint].checkpoint_at = 20;
    sample_params one = params();
    one.max_new = 1;
    engine_start(e, *endpoint, one);
    finish(e, endpoint, 1);
    assert(e->st.checkpoints == 1 && e->n_cached == 1);
    *checkpoint = e->cached[0];
    assert(e->seqs[*checkpoint].is_prompt_checkpoint);
    engine_retire(e, *endpoint);
    assert(!e->seqs[*endpoint].is_prompt_checkpoint);
    assert(e->cached[0] == *checkpoint && e->cached[1] == *endpoint);
    e->prompt_checkpoints = false;
}

static void phase_priority_capture(bool priority) {
    model m = fixture();
    backend b = compute();
    engine *e = engine_new(&m, &b, NULL, 2u << 20);
    e->row_budget = 17;
    e->kv.recurrent_limit = 4;
    e->recurrent_cache_priority = priority;
    int endpoint, checkpoint;
    seed_phase_cache(e, &endpoint, &checkpoint);
    int checkpoint_gen = e->seqs[checkpoint].ng_gen;
    e->seqs[endpoint].pages.recurrent_handle = malloc(8);
    e->seqs[endpoint].pages.release_recurrent_handle = release_fake_handle;
    released_handles = 0;
    int32_t prompt[100];
    for (int i = 0; i < 100; i++)
        prompt[i] = (i * 5 + 9) % VOCAB;
    int active = engine_create_fresh(e, prompt, 100);
    engine_start(e, active, params());
    assert(engine_step(e, &active, 1));
    float *active_state = e->seqs[active].pages.recurrent;
    e->seqs[active].pages.recurrent_handle = malloc(8);
    e->seqs[active].pages.release_recurrent_handle = release_fake_handle;
    int arriving = engine_create_fresh(e, prompt, 100);
    engine_start(e, arriving, params());
    e->seqs[arriving].checkpoint_at = 41;
    e->prompt_checkpoints = true;
    assert(engine_step(e, &arriving, 1));
    assert(e->kv.recurrent_used == 4);
    assert(engine_step(e, &arriving, 1));
    assert(e->seqs[arriving].n_tokens == 41);
    assert(e->seqs[checkpoint].live && e->seqs[checkpoint].ng_gen == checkpoint_gen);
    assert(e->seqs[active].pages.recurrent == active_state && e->seqs[active].generating);
    assert(e->kv.recurrent_used == 4 && e->st.checkpoint_evictions == 0 && e->st.slot_preemptions == 0);
    if (priority) {
        assert(e->st.checkpoints == 2 && e->st.skipped_checkpoints == 0 && e->st.endpoint_evictions == 1 && released_handles == 1);
        int found = 0;
        for (int i = 0; i < e->n_cached; i++)
            found += e->seqs[e->cached[i]].is_prompt_checkpoint && e->seqs[e->cached[i]].n_tokens == 41;
        assert(found == 1);
    } else
        assert(e->st.checkpoints == 1 && e->st.skipped_checkpoints == 1 && e->st.endpoint_evictions == 0 && released_handles == 0);
    finish(e, &arriving, 1);
    int32_t out[OUTPUT];
    expected(prompt, 100, out);
    assert(!memcmp(out, e->seqs[arriving].gen, sizeof out));
    release_engine(e);
    assert(released_handles == 2);
    printf("phase capture priority=%d: bounded slots, protected active/checkpoint, callback ownership, exact output PASS\n", priority);
}

static void phase_priority_prefill(bool priority, bool waited_endpoint) {
    model m = fixture();
    backend b = compute();
    engine *e = engine_new(&m, &b, NULL, 2u << 20);
    e->row_budget = 17;
    e->kv.recurrent_limit = 3;
    e->recurrent_cache_priority = priority;
    int endpoint, checkpoint;
    seed_phase_cache(e, &endpoint, &checkpoint);
    e->seqs[endpoint].pages.recurrent_handle = malloc(8);
    e->seqs[endpoint].pages.release_recurrent_handle = release_fake_handle;
    released_handles = 0;
    int32_t prompt[100];
    for (int i = 0; i < 100; i++)
        prompt[i] = (i * 5 + 9) % VOCAB;
    int active = engine_create_fresh(e, prompt, 100);
    engine_start(e, active, params());
    assert(engine_step(e, &active, 1));
    float *active_state = e->seqs[active].pages.recurrent;
    if (waited_endpoint) {
        int waiter = engine_create_fresh(e, prompt, 100);
        e->seqs[waiter].wait_for = endpoint;
        e->seqs[waiter].wait_gen = e->seqs[endpoint].ng_gen;
        e->seqs[waiter].wait_len = 60;
    }
    int arriving = engine_create_fresh(e, prompt, 100);
    engine_start(e, arriving, params());
    assert(engine_step(e, &arriving, 1));
    assert(e->kv.recurrent_used == 3 && e->seqs[active].pages.recurrent == active_state && e->st.slot_preemptions == 0);
    if (priority && !waited_endpoint) {
        assert(e->seqs[checkpoint].live && !e->seqs[endpoint].live);
        assert(e->st.endpoint_evictions == 1 && e->st.checkpoint_evictions == 0 && released_handles == 1);
    } else {
        assert(!e->seqs[checkpoint].live && e->seqs[endpoint].live);
        assert(e->st.endpoint_evictions == 0 && e->st.checkpoint_evictions == 1 && released_handles == 0);
    }
    finish(e, &arriving, 1);
    int32_t out[OUTPUT];
    expected(prompt, 100, out);
    assert(!memcmp(out, e->seqs[arriving].gen, sizeof out));
    release_engine(e);
    assert(released_handles == 1);
    printf("phase prefill priority=%d waited-endpoint=%d: FIFO/default, endpoint preference, waiter protection PASS\n", priority, waited_endpoint);
}

static int phase_checkpoint_count(const engine *e, int at) {
    int count = 0;
    for (int i = 0; i < e->n_seqs; i++)
        count += e->seqs[i].live && e->seqs[i].cached && e->seqs[i].is_prompt_checkpoint && e->seqs[i].n_tokens == at;
    return count;
}

static void checkpoint_lifecycle(bool move, bool priority, bool growing, bool pressure) {
    model m = fixture();
    backend b = compute();
    engine *e = engine_new(&m, &b, NULL, 2u << 20);
    e->prompt_checkpoints = true;
    e->recurrent_state_move = move;
    e->recurrent_cache_priority = priority;
    e->row_budget = 17;
    if (pressure) e->kv.recurrent_limit = 2;
    int32_t prompt[160];
    for (int i = 0; i < 160; i++)
        prompt[i] = (i * 3 + 7) % VOCAB;
    released_handles = 0;
    int reused[4];
    for (int turn = 0; turn < 4; turn++) {
        int length = growing ? 100 + 20 * turn : 100, at = length - 20;
        int id = engine_create(e, prompt, length);
        reused[turn] = e->seqs[id].n_tokens;
        assert(e->seqs[id].n_tokens == (turn ? (growing ? at - 20 : at) : 0));
        // Match the server lifecycle: attach this boundary AFTER create reused an endpoint.
        e->seqs[id].checkpoint_at = at;
        engine_start(e, id, params());
        finish(e, &id, 1);
        engine_retire(e, id);
        int32_t out[OUTPUT];
        expected(prompt, length, out);
        assert(!memcmp(out, e->seqs[id].gen, sizeof out));
        assert(phase_checkpoint_count(e, at) == 1 && e->st.checkpoints == turn + 1 && e->st.skipped_checkpoints == 0);
        if (pressure) {
            assert(e->kv.recurrent_used == 2 && e->st.endpoint_evictions == turn && e->st.checkpoint_evictions == 0 && released_handles == turn);
            assert(!e->seqs[id].pages.recurrent_handle);
            e->seqs[id].pages.recurrent_handle = malloc(8);
            e->seqs[id].pages.release_recurrent_handle = release_fake_handle;
        }
    }
    // Existing later-boundary captures do the same work; identical requests keep
    // their warm checkpoint instead of alternating fresh and reused prefixes.
    assert(e->st.prefill_rows + e->st.decode_rows == (growing ? 264 : 204));
    release_engine(e);
    if (pressure) assert(released_handles == 4);
    printf("checkpoint lifecycle move=%d priority=%d growing=%d pressure=%d reused=%d,%d,%d,%d: exact replies, bounded capture PASS\n", move, priority, growing,
           pressure, reused[0], reused[1], reused[2], reused[3]);
}

static void fractions(void) {
    model m = fixture();
    m.c.recurrent_floats = 8192;
    backend b = compute();
    int last = 0;
    const double values[] = {.05, .25, .50, .65, .90};
    for (int i = 0; i < 5; i++) {
        double f = values[i];
        engine *e = engine_new_with_recurrent_fraction(&m, &b, NULL, 2u << 20, f);
        assert(e->kv.recurrent_limit > last);
        last = e->kv.recurrent_limit;
        assert(e->kv.full.bytes + e->kv.swa.bytes + e->kv.recurrent_bytes * e->kv.recurrent_limit <= (2u << 20));
        printf("fraction %.2f: %d slots, same total budget PASS\n", f, last);
        release_engine(e);
    }
}
int main(void) {
    long plain = fanout(false, false, 0), dedup = fanout(true, false, 0);
    assert(plain - dedup == 155);
    fanout(true, true, 0);
    fanout(true, false, 1);
    fanout(true, false, 2);
    abandoned_leader(false, false, false);
    abandoned_leader(true, false, false);
    abandoned_leader(false, true, false);
    abandoned_leader(false, false, true);
    slot_exhaustion_at_boundary();
    cancelled_waiter();
    full_slot_state_move();
    phase_priority_capture(false);
    phase_priority_capture(true);
    phase_priority_prefill(false, false);
    phase_priority_prefill(true, false);
    phase_priority_prefill(true, true);
    checkpoint_lifecycle(false, false, false, false);
    checkpoint_lifecycle(true, false, false, false);
    checkpoint_lifecycle(true, true, false, false);
    checkpoint_lifecycle(true, true, true, false);
    checkpoint_lifecycle(true, true, false, true);
    fractions();
    puts("ALL DEDUP SCHEDULER CHECKS PASS");
    return 0;
}
