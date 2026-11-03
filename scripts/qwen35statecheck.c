// Real-checkpoint checks for recurrent ownership, forks, resets and batching.
#include "../tools/common.h"
#include <assert.h>
#include <math.h>
static void prefill(runtime *rt, int id) {
    while (rt->e->seqs[id].n_pending)
        assert(engine_step(rt->e, &id, 1));
}
static void check_same(const seq_t *a, const seq_t *b) {
    assert(a->n_gen == b->n_gen);
    assert(!memcmp(a->gen, b->gen, a->n_gen * 4));
}
int main(int argc, char **argv) {
    runtime rt = load_runtime(argc, argv);
    engine *e = rt.e;
    assert(rt.m->c.n_linear);
    sample_params p = sample_defaults();
    p.temperature = 0;
    p.max_new = 12;
    p.ignore_eos = true;
    int32_t *prompt;
    int n = encode_chat(rt.tok, "Write a Python function that adds two integers.", true, &prompt);
    int parent = engine_create_fresh(e, prompt, n);
    prefill(&rt, parent);
    assert(e->seqs[parent].pages.recurrent_len == n);
    assert(!kv_can_fork_at(&e->kv, &e->seqs[parent].pages, n - 1));
    assert(kv_can_fork_at(&e->kv, &e->seqs[parent].pages, n));
    int32_t suffix[] = {tok_lookup(rt.tok, "\n")};
    if (suffix[0] < 0) suffix[0] = prompt[n - 1];
    int child = engine_fork_at(e, parent, n, suffix, 1);
    assert(e->seqs[parent].pages.recurrent != e->seqs[child].pages.recurrent);
    assert(!memcmp(e->seqs[parent].pages.recurrent, e->seqs[child].pages.recurrent, rt.m->c.recurrent_floats * 4));
    float *snapshot = malloc(e->kv.recurrent_bytes);
    memcpy(snapshot, e->seqs[parent].pages.recurrent, e->kv.recurrent_bytes);
    int32_t *whole = malloc((n + 1) * 4);
    memcpy(whole, prompt, n * 4);
    whole[n] = suffix[0];
    int fresh = engine_create_fresh(e, whole, n + 1);
    engine_start(e, child, p);
    engine_start(e, fresh, p);
    int batch[] = {child, fresh};
    engine_generate(e, batch, 2);
    check_same(&e->seqs[child], &e->seqs[fresh]);
    assert(!memcmp(snapshot, e->seqs[parent].pages.recurrent, e->kv.recurrent_bytes));
    int32_t generated[12];
    memcpy(generated, e->seqs[fresh].gen, sizeof generated);
    engine_release(e, child);
    engine_release(e, fresh);
    engine_release(e, parent);
    assert(e->kv.recurrent_used == 0);
    // Released slots must start with zero recurrent state.
    fresh = engine_create_fresh(e, whole, n + 1);
    engine_start(e, fresh, p);
    engine_generate(e, &fresh, 1);
    assert(!memcmp(generated, e->seqs[fresh].gen, sizeof generated));
    engine_release(e, fresh);
    // Unavailable intermediate fork falls back to exact recomputation.
    parent = engine_create_fresh(e, prompt, n);
    prefill(&rt, parent);
    child = engine_fork_at(e, parent, n - 1, suffix, 1);
    assert(!e->seqs[child].pages.recurrent);
    whole[n - 1] = suffix[0];
    fresh = engine_create_fresh(e, whole, n);
    engine_start(e, child, p);
    engine_start(e, fresh, p);
    int pair[] = {child, fresh};
    engine_generate(e, pair, 2);
    check_same(&e->seqs[child], &e->seqs[fresh]);
    engine_release(e, child);
    engine_release(e, fresh);
    engine_release(e, parent);
    // Exercise recompute preemption with two available state slots and three agents.
    assert(e->kv.recurrent_limit >= 2);
    e->kv.recurrent_limit = 2;
    int ids[3];
    for (int i = 0; i < 3; i++) {
        ids[i] = engine_create_fresh(e, whole, n);
        engine_start(e, ids[i], p);
    }
    for (int i = 0; i < 3; i++) {
        engine_generate(e, &ids[i], 1);
        if (i) check_same(&e->seqs[ids[0]], &e->seqs[ids[i]]);
    }
    assert(e->st.preemptions > 0);
    for (int i = 0; i < 3; i++)
        engine_release(e, ids[i]);
    assert(e->kv.recurrent_used == 0);
    printf(
        "PASS: endpoint fork, independent child state, concurrent/fresh equivalence, slot reset, intermediate recomputation, state-pressure preemption (%ld)\n",
        e->st.preemptions);
    free(snapshot);
    free(whole);
    free(prompt);
    engine_free(e);
    rt.be->destroy(rt.be);
    model_free(rt.m);
    tok_free(rt.tok);
    return 0;
}
