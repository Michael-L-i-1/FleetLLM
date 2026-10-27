#include "threadpool.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

static struct {
    int n;
    pthread_t *th;
    pthread_mutex_t mu;
    pthread_cond_t go, done;
    uint64_t gen;
    // current job
    pf_fn fn;
    void *ctx;
    int total, grain;
    atomic_int next;
    int active; // workers still in the job
} P;

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
    pthread_mutex_lock(&P.mu);
    for (;;) {
        while (P.gen == seen) pthread_cond_wait(&P.go, &P.mu);
        seen = P.gen;
        pthread_mutex_unlock(&P.mu);
        run_chunks();
        pthread_mutex_lock(&P.mu);
        if (--P.active == 0) pthread_cond_signal(&P.done);
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
    pthread_cond_init(&P.done, NULL);
    P.th = calloc(P.n, sizeof(pthread_t));
    for (int i = 1; i < P.n; i++) pthread_create(&P.th[i], NULL, worker, NULL); // caller is thread 0
}

int tp_threads(void) { return P.n ? P.n : 1; }

void parallel_for(int n, int grain, pf_fn fn, void *ctx) {
    if (!P.th) tp_init(0);
    if (grain < 1) grain = 1;
    if (n <= grain || P.n == 1) { fn(ctx, 0, n); return; }
    pthread_mutex_lock(&P.mu);
    P.fn = fn;
    P.ctx = ctx;
    P.total = n;
    P.grain = grain;
    atomic_store(&P.next, 0);
    P.active = P.n - 1;
    P.gen++;
    pthread_cond_broadcast(&P.go);
    pthread_mutex_unlock(&P.mu);
    run_chunks();
    pthread_mutex_lock(&P.mu);
    while (P.active > 0) pthread_cond_wait(&P.done, &P.mu);
    pthread_mutex_unlock(&P.mu);
}
