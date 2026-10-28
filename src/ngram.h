// Model-free drafts from repeated text (prompt lookup / SuffixDecoding-style).
//
// Agents repeat what they have seen: they quote files into edits, echo tool
// arguments and paths, re-emit structure, fall into loops, and subagents doing
// similar work write similar things. The draft for a sequence is what followed
// the most recent earlier occurrence of its current tail, found in its own
// context or in any other sequence's (agents share what they read and write).
// The match length sets how far to trust the continuation.
//
// Each sequence keeps a small hash of its NG-token windows; a global table maps
// a window hash to the latest (sequence, position) anywhere. Positions are
// verified against the actual tokens, so stale entries are harmless.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#define NG_N 3 // window length

// per-sequence index: window hash -> position just after its latest occurrence
typedef struct {
    int32_t *pos; // -1 empty
    uint32_t *key;
    uint32_t mask, used;
    int indexed;  // windows ending before this position are in the table
} ng_seq;

typedef struct { uint32_t key; int32_t seq, pos, gen; } ng_slot;
typedef struct { ng_slot *t; uint32_t mask; } ng_global;

typedef struct { const int32_t *tok; int n; int gen; } ng_view; // a sequence's tokens

void ng_seq_free(ng_seq *x);
void ng_global_init(ng_global *g, int log2_slots);
void ng_global_free(ng_global *g);

// Index windows of ctx[0..n) that end before n-1 (the tail itself is the query).
void ng_update(ng_seq *x, ng_global *g, int seq, int gen, const int32_t *ctx, int n);

// Draft up to max tokens continuing ctx[0..n). view(seq) returns another
// sequence's tokens (or n=0 if gone). Returns the count; *match = matched length.
int ng_draft(const ng_seq *x, const ng_global *g, int seq, const int32_t *ctx, int n, ng_view (*view)(void *u, int seq), void *u,
             int32_t *out, int max, int *match);
