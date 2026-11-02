// Concurrent agents with different tasks (bench/tasks.txt), all submitted at
// t=0, fixed output length, greedy. Same workload as the earlier vLLM runs.
//   fleet bench --model glimmer --levels 1,2,4,8,16 --gen 128 [--reps 2]
//   fleet bench --model llama1b --check     greedy batched outputs == sequential outputs
//   fleet bench --model glimmer-mlx4 --draft glimmer-draft --check-spec   speculative outputs == plain greedy outputs
//   --share: automatic prefix sharing on (new agents fork the KV of live, cached
//            or still-prefilling sequences with the same prefix; finished agents
//            stay cached), comparable to a server with prefix caching enabled
#include "common.h"

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1);
    fread(b, 1, n, f);
    b[n] = 0;
    fclose(f);
    return b;
}

typedef struct { int32_t *ids; int n; } prompt_t;

// Run k agents (prompt i % n_tasks) concurrently or one by one; returns wall seconds.
static bool share;

static double run(engine *e, const prompt_t *pr, int n_tasks, int k, int gen, bool batched, bool eos, int32_t ***outs, int **out_n, double *ttft) {
    int *ids = malloc(sizeof(int) * k);
    sample_params p = sample_defaults();
    p.temperature = 0;
    p.max_new = gen;
    p.ignore_eos = !eos;
    *outs = calloc(k, sizeof(int32_t *));
    *out_n = calloc(k, sizeof(int));
    double t0 = now_secs(), first = 0;
    for (int b = 0; b < k; b += batched ? k : 1) {
        int m = batched ? k : 1;
        for (int i = 0; i < m; i++) {
            const prompt_t *t = &pr[(b + i) % n_tasks];
            ids[i] = share ? engine_create(e, t->ids, t->n) : engine_create_fresh(e, t->ids, t->n);
            engine_start(e, ids[i], p);
        }
        engine_generate(e, ids, m);
        for (int i = 0; i < m; i++) {
            seq_t *s = &e->seqs[ids[i]];
            first += s->t_first - (batched ? t0 : s->t_start);
            (*outs)[b + i] = malloc(sizeof(int32_t) * s->n_gen);
            memcpy((*outs)[b + i], s->gen, sizeof(int32_t) * s->n_gen);
            (*out_n)[b + i] = s->n_gen;
            if (share) engine_retire(e, ids[i]);
            else engine_release(e, ids[i]);
        }
    }
    *ttft = first / k;
    free(ids);
    return now_secs() - t0;
}

int main(int argc, char **argv) {
    runtime rt = load_runtime(argc, argv);
    char *text = read_file(arg_str(argc, argv, "--tasks", "bench/tasks.txt"));
    prompt_t pr[64];
    int n_tasks = 0;
    for (char *s = text; s && n_tasks < 64;) {
        char *sep = strstr(s, "\n=====\n");
        if (sep) *sep = 0;
        pr[n_tasks].n = encode_chat(rt.tok, s, true, &pr[n_tasks].ids);
        n_tasks++;
        s = sep ? sep + 7 : NULL;
    }
    int gen = (int)arg_num(argc, argv, "--gen", 128);
    engine *e = rt.e;
    share = arg_flag(argc, argv, "--share");
    { // warm-up: pipelines, page tables, both matmul paths
        int32_t **o; int *on; double tt;
        run(e, pr, n_tasks, n_tasks < 2 ? 1 : 2, 4, true, false, &o, &on, &tt);
    }
    if (arg_flag(argc, argv, "--check-spec") && (rt.draft || e->ngram)) {
        // every task alone without drafts (spec_rows 0), then all batched with speculation
        int32_t **a, **b; int *an, *bn; double tt;
        int rows = e->spec_rows;
        e->spec_rows = 0;
        double ws = run(e, pr, n_tasks, n_tasks, gen, false, true, &a, &an, &tt);
        e->spec_rows = rows;
        double wb = run(e, pr, n_tasks, n_tasks, gen, true, true, &b, &bn, &tt);
        int same = 0;
        for (int i = 0; i < n_tasks; i++) {
            int k = 0;
            while (k < an[i] && k < bn[i] && a[i][k] == b[i][k]) k++;
            same += an[i] == bn[i] && k == an[i];
            if (k < an[i] || an[i] != bn[i]) printf("task %d: first difference at token %d of %d\n", i, k, an[i]);
        }
        printf("check-spec: %d/%d speculative outputs identical to plain greedy (plain %.2fs, speculative batched %.2fs)\n", same, n_tasks, ws, wb);
        return same == n_tasks ? 0 : 2;
    }
    if (arg_flag(argc, argv, "--check")) {
        int32_t **a, **b; int *an, *bn; double tt;
        double ws = run(e, pr, n_tasks, n_tasks, gen, false, true, &a, &an, &tt);
        double wb = run(e, pr, n_tasks, n_tasks, gen, true, true, &b, &bn, &tt);
        int same = 0;
        for (int i = 0; i < n_tasks; i++) same += an[i] == bn[i] && !memcmp(a[i], b[i], sizeof(int32_t) * an[i]);
        printf("check: %d/%d batched outputs identical to sequential (sequential %.2fs, batched %.2fs)\n", same, n_tasks, ws, wb);
        return same == n_tasks ? 0 : 2;
    }
    int reps = (int)arg_num(argc, argv, "--reps", 2);
    const char *levels = arg_str(argc, argv, "--levels", "1,2,4,8");
    printf("engine=FleetLLM backend=%s model=%s gen=%d share=%d spec=%d\n", rt.be->name, arch_name(rt.m->c.arch), gen, share, rt.draft ? rt.e->spec_k : 0);
    printf("%4s %8s %9s %10s %10s\n", "N", "wall s", "TTFT ms", "agg tok/s", "per-agent");
    char lv[256];
    snprintf(lv, sizeof lv, "%s", levels);
    for (char *tok = strtok(lv, ","); tok; tok = strtok(NULL, ",")) {
        int k = atoi(tok);
        double best = 1e30, best_ttft = 0;
        for (int r = 0; r < reps; r++) {
            int32_t **o; int *on; double tt;
            double w = run(e, pr, n_tasks, k, gen, true, false, &o, &on, &tt);
            if (w < best) { best = w; best_ttft = tt; }
        }
        double agg = (double)k * gen / best;
        printf("%4d %8.2f %9.0f %10.1f %10.1f  (prefill rows %ld, shared %ld", k, best, best_ttft * 1e3, agg, agg / k, e->st.prefill_rows, e->st.shared_tokens);
        if (e->st.spec_steps)
            printf(", spec: %.2f tokens/verify, %.0f%% of drafts accepted, drafter %.1f%% of time", 1.0 + (double)e->st.spec_accepted / e->st.spec_steps,
                   100.0 * e->st.spec_accepted / e->st.spec_drafted, 100.0 * e->st.draft_secs / (e->st.draft_secs + e->st.forward_secs));
        if (e->st.ng_steps)
            printf(", repeated-text drafts: %ld verifies, %.2f tokens/verify, %.0f%% accepted", e->st.ng_steps, 1.0 + (double)e->st.ng_accepted / e->st.ng_steps,
                   100.0 * e->st.ng_accepted / e->st.ng_drafted);
        printf(")\n");
        e->st.prefill_rows = e->st.shared_tokens = e->st.spec_steps = e->st.spec_drafted = e->st.spec_accepted = 0;
        e->st.ng_steps = e->st.ng_drafted = e->st.ng_accepted = 0;
        e->st.draft_secs = e->st.forward_secs = 0;
        fflush(stdout);
    }
    return 0;
}
