// Sequence manager + continuous-batching scheduler.
//
// Every sequence holds a queue of pending tokens (prompt not yet in the KV
// cache, or the token it just sampled). Each step packs one decode row per
// generating sequence, then prefill chunks until the row budget is used, and
// runs them as one batched forward. A new sequence forks the KV of the live or
// cached sequence that shares its longest token prefix (copy-on-write pages),
// so subagents with a common system prompt / task prefill it once. If the
// longest match is with a sequence whose prompt is still queued (agents fanned
// out at the same moment), the newcomer waits for that leader to prefill the
// shared part and then forks it, instead of computing it a second time.
#pragma once
#include "backend.h"
#include "sampler.h"
#include "tokenizer.h"
#include "ngram.h"

typedef struct {
    bool live, cached, generating, done;
    seq_pages pages;
    int32_t *tokens; int n_tokens, cap_tokens;    // tokens whose KV is written
    int32_t *pending; int n_pending, cap_pending; // queued input
    int32_t *gen; int n_gen, cap_gen;             // generated output
    int wait_for, wait_len; // waiting for seqs[wait_for] to write wait_len shared tokens (wait_len 0: not waiting)
    seq_pages dpages;        // drafter KV (speculative decoding): context of the first dpages.len tokens
    int32_t drafts[32]; int n_drafts; // proposals to verify in the next step
    ng_seq ng; int ng_gen;            // repeated-text draft index (see ngram.h)
    int ng_drafts;                    // drafts this step that came from repeated text
    sample_params p;
    uint64_t rng;
    double t_start, t_first; // creation time, first generated token (seconds)
} seq_t;

typedef struct {
    long steps, decode_rows, prefill_rows, shared_tokens, gen_tokens, deduped_tokens;
    long spec_steps, spec_drafted, spec_accepted; // verify steps, drafts checked, drafts accepted
    long ng_steps, ng_drafted, ng_accepted;       // the subset drafted from repeated text
    long preemptions; // sequences whose KV was freed under memory pressure (recomputed later)
    double draft_secs;
    double forward_secs;
} engine_stats;

typedef struct {
    model *m;
    backend *be;
    const tokenizer *tok;
    kv_cache kv;
    seq_t *seqs; int n_seqs;
    int *cached; int n_cached; // retired sequences kept for prefix reuse, oldest first
    int row_budget;            // max rows per forward step (prefill chunking)
    int min_share;             // minimum common prefix worth forking
    engine_stats st;
    // speculative decoding (DFlash): drafter, its KV, drafts per verify step,
    // and the drafter context rows produced by the last forward
    model *draft;
    kv_cache dkv;
    int spec_k;        // max drafts verified per sequence per step
    int spec_rows;     // target decode + draft rows per step: drafts per sequence = spec_rows / decoding - 1
    draft_row *dctx; int n_dctx, cap_dctx;
    // model-free speculation from repeated text (any model/backend; see ngram.h)
    bool ngram;
    ng_global ng;
    int ng_gen;
    // step scratch
    row_t *rows; int cap_rows;
    int32_t *amax; int cap_amax;
    float *logits; size_t cap_logits;
} engine;

engine *engine_new(model *m, backend *be, const tokenizer *tok, size_t kv_budget_bytes);
void engine_free(engine *e);
// Enable speculative decoding with a DFlash drafter (greedy sequences only);
// its KV cache gets kv_budget_bytes. Returns false if the backend cannot draft.
bool engine_set_drafter(engine *e, model *dm, size_t kv_budget_bytes, int spec_k);
// Enable drafting from repeated text in a sequence's own context and in every
// other sequence's (exact for greedy and sampled decoding).
void engine_set_ngram(engine *e, bool on);

// New sequence for prompt; shares KV with the best prefix match (>= min_share tokens).
int engine_create(engine *e, const int32_t *prompt, int n);
// New sequence that never shares (baseline).
int engine_create_fresh(engine *e, const int32_t *prompt, int n);
// Child sharing parent's first len positions, continuing with suffix.
int engine_fork_at(engine *e, int parent, int len, const int32_t *suffix, int n);
void engine_start(engine *e, int id, sample_params p);
void engine_release(engine *e, int id); // free KV now
void engine_retire(engine *e, int id);  // keep KV for prefix reuse until memory is needed

// One batched forward over the given sequences; returns false if nothing ran.
bool engine_step(engine *e, const int *ids, int n);
// Run steps until every listed sequence finished generating.
void engine_generate(engine *e, const int *ids, int n);
double now_secs(void);
