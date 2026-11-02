#include "threadpool.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// Workers spin on the job counter for a while after each job before sleeping on
// the condition variable: a decode step issues a few hundred short jobs, and a
// condvar wake-up costs tens of microseconds each.
#define SPIN 20000

static struct {
    int n;
    pthread_t *th;
    pthread_mutex_t mu;
    pthread_cond_t go;
    int sleepers; // workers blocked on go (under mu)
    _Atomic uint64_t gen;
    // current job (published by the release increment of gen)
    pf_fn fn;
    void *ctx;
    int total, grain;
    atomic_int next;
    atomic_int active; // workers still in the job
} P;

static inline void pause_cpu(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield");
#elif defined(__x86_64__)
    __asm__ volatile("pause");
#endif
}

static void run_chunks(void) {
    for (;;) {
        int b = atomic_fetch_add(&P.next, P.grain);
        if (b >= P.total) break;
        int e = b + P.grain < P.total ? b + P.grain : P.total;
        P.fn(P.ctx, b, e);
    }
}

static void *worker(void *arg) {
    (void)arg;
    uint64_t seen = 0;
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&P.gen, memory_order_acquire) == seen && spins < SPIN) { pause_cpu(); spins++; }
        if (atomic_load_explicit(&P.gen, memory_order_acquire) == seen) {
            pthread_mutex_lock(&P.mu);
            P.sleepers++;
            while (atomic_load(&P.gen) == seen) pthread_cond_wait(&P.go, &P.mu);
            P.sleepers--;
            pthread_mutex_unlock(&P.mu);
        }
        seen = atomic_load_explicit(&P.gen, memory_order_acquire);
        run_chunks();
        atomic_fetch_sub_explicit(&P.active, 1, memory_order_release);
    }
    return NULL;
}

static int default_threads(void) {
#ifdef __APPLE__
    int perf = 0;
    size_t len = sizeof perf;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, NULL, 0) == 0 && perf > 0) return perf;
#endif
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 4;
}

void tp_init(int threads) {
    if (P.th) return;
    P.n = threads > 0 ? threads : default_threads();
    pthread_mutex_init(&P.mu, NULL);
    pthread_cond_init(&P.go, NULL);
    P.th = calloc(P.n, sizeof(pthread_t));
    for (int i = 1; i < P.n; i++) pthread_create(&P.th[i], NULL, worker, NULL); // caller is thread 0
}

int tp_threads(void) { return P.n ? P.n : 1; }

void parallel_for(int n, int grain, pf_fn fn, void *ctx) {
    if (!P.th) tp_init(0);
    if (grain < 1) grain = 1;
    if (n <= grain || P.n == 1) { fn(ctx, 0, n); return; }
    P.fn = fn;
    P.ctx = ctx;
    P.total = n;
    P.grain = grain;
    atomic_store(&P.next, 0);
    atomic_store(&P.active, P.n - 1);
    atomic_fetch_add_explicit(&P.gen, 1, memory_order_release);
    pthread_mutex_lock(&P.mu); // a worker either saw the new gen or is counted in sleepers
    if (P.sleepers) pthread_cond_broadcast(&P.go);
    pthread_mutex_unlock(&P.mu);
    run_chunks();
    while (atomic_load_explicit(&P.active, memory_order_acquire) > 0) pause_cpu();
}
