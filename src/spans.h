// Content-addressed KV reuse across sequences ("this text already has KV").
//
// Agents working in one environment see the same text again and again: the
// same file read by several subagents, the same search result, the same tool
// output; a session whose harness dropped an old message re-sends everything
// after it at shifted positions. Prefix sharing covers none of this (the text
// sits at a different position after different context). The span index maps
// every 16-token block of a sequence's written tokens (block-aligned in that
// sequence) to its latest (sequence, position); a new sequence's pending input
// is scanned for long runs that some other resident sequence already holds,
// whose KV can then be moved over (kv_move: K re-rotated to the new positions)
// instead of recomputed. Lossy: that KV was computed after different preceding
// text, so the run's first and last tokens are recomputed (see reuse_eval).
#pragma once
#include "ngram.h"
#include <stdint.h>

#define SPAN_BLOCK 16

typedef struct { uint32_t key; int32_t seq, pos, gen; } span_slot;
typedef struct { span_slot *t; uint32_t mask; } span_index;

void span_init(span_index *x, int log2_slots);
void span_free(span_index *x);
// Index the blocks of tok that end in (from, n]; returns the new indexed length.
int span_add(span_index *x, int seq, int gen, const int32_t *tok, int from, int n);

typedef struct { int off, len, seq, pos; } span_run; // pend[off, off+len) == tokens of seq at [pos, pos+len)

// Longest run of pend[0..n) of at least min_len tokens held by another sequence
// (view(seq) gives its written tokens, n = 0 if not resident). false if none.
bool span_find(const span_index *x, int self, const int32_t *pend, int n, int min_len, ng_view (*view)(void *u, int seq), void *u, span_run *out);
