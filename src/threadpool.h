// Persistent worker pool with a blocking parallel_for over [0, n).
#pragma once

typedef void (*pf_fn)(void *ctx, int begin, int end);

void tp_init(int threads); // 0 = number of performance cores
int tp_threads(void);
// Run fn over [0, n) split into chunks of `grain` items; returns when all are done.
void parallel_for(int n, int grain, pf_fn fn, void *ctx);
