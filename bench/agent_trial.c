// Short concurrent-request trial. Build with -Itools -Isrc and libfleetllm.a.
// Decode window begins after every request has produced its first token.
#include "common.h"
#include "json.h"

typedef struct { char *id; int32_t *tokens; int n; } task;
static void quoted(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') { putchar('\\'); putchar(*p); }
        else if (*p < 32) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}
int main(int argc, char **argv) {
    const char *path = arg_str(argc, argv, "--tasks", NULL);
    int limit = (int)arg_num(argc, argv, "--gen", 32);
    if (!path || limit < 2 || limit > 4096) return 1;
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 1; }
    runtime rt = load_runtime(argc, argv);
    task tasks[8] = {0}; int nt = 0;
    char *line = NULL; size_t cap = 0;
    while (nt < 8 && getline(&line, &cap, f) > 0) {
        json *j = json_parse(line, strlen(line));
        const jnode *root = j ? json_root(j) : NULL;
        const char *id = j ? json_str(j, root, "id") : NULL;
        const char *prompt = j ? json_str(j, root, "prompt") : NULL;
        if (!id || !prompt) return 1;
        tasks[nt].id = strdup(id);
        tasks[nt].n = encode_chat(rt.tok, prompt, true, &tasks[nt].tokens);
        nt++; json_free(j);
    }
    free(line); fclose(f);
    if (nt != 8) { fprintf(stderr, "need eight task definitions\n"); return 1; }
    // Touch model pages before timing; this also separates loader swap from
    // generation-time memory pressure in the external benchmark guard.
    sample_params warm = sample_defaults();
    warm.temperature = 0; warm.max_new = 4; warm.ignore_eos = true;
    int warm_id = engine_create_fresh(rt.e, tasks[0].tokens, tasks[0].n);
    engine_start(rt.e, warm_id, warm);
    engine_generate(rt.e, &warm_id, 1);
    engine_release(rt.e, warm_id);
    fprintf(stderr, "[trial-ready] weights warmed; timed levels start\n");
    fflush(stderr);
    int32_t *reference = NULL; int reference_n = 0;
    const int levels[] = {1, 4, 8};
    for (int level = 0; level < 3; level++) {
        int n = levels[level], ids[8], active[8];
        engine_stats before = rt.e->st;
        double start = now_secs(), barrier = 0;
        int tokens_at_barrier = 0;
        sample_params params = sample_defaults();
        params.temperature = 0; params.max_new = limit; params.ignore_eos = true;
        for (int i = 0; i < n; i++) {
            ids[i] = engine_create(rt.e, tasks[i].tokens, tasks[i].n);
            engine_start(rt.e, ids[i], params);
        }
        for (;;) {
            int k = 0;
            for (int i = 0; i < n; i++) if (!rt.e->seqs[ids[i]].done) active[k++] = ids[i];
            if (!k) break;
            if (!engine_step(rt.e, active, k)) { fprintf(stderr, "engine stalled\n"); return 2; }
            bool ready = true; int generated = 0;
            for (int i = 0; i < n; i++) {
                seq_t *s = &rt.e->seqs[ids[i]];
                ready &= s->n_gen > 0; generated += s->n_gen;
            }
            if (!barrier && ready) { barrier = now_secs(); tokens_at_barrier = generated; }
        }
        double end = now_secs(); int total = 0;
        double mean_first = 0, last_first = 0;
        for (int i = 0; i < n; i++) {
            seq_t *s = &rt.e->seqs[ids[i]];
            total += s->n_gen; mean_first += s->t_first - start;
            if (s->t_first - start > last_first) last_first = s->t_first - start;
            if (s->n_gen != limit) return 2;
        }
        seq_t *first = &rt.e->seqs[ids[0]];
        bool match = !reference || (reference_n == first->n_gen &&
            !memcmp(reference, first->gen, reference_n * sizeof(int32_t)));
        if (!reference) {
            reference_n = first->n_gen;
            reference = malloc(reference_n * sizeof(int32_t));
            memcpy(reference, first->gen, reference_n * sizeof(int32_t));
        }
        double seconds = end - barrier;
        double rate = seconds > 0 ? (total - tokens_at_barrier) / seconds : 0;
        printf("{\"agents\":%d,\"tokens\":%d,\"wall_seconds\":%.6f,\"end_to_end_tok_s\":%.6f,"
               "\"mean_first_token_seconds\":%.6f,\"last_first_token_seconds\":%.6f,"
               "\"decode_window_seconds\":%.6f,\"decode_window_tokens\":%d,"
               "\"decode_aggregate_tok_s\":%.6f,\"decode_average_per_agent_tok_s\":%.6f,"
               "\"first_agent_matches_single\":%s,\"shared_tokens\":%ld,\"prefill_rows\":%ld,"
               "\"ngram_drafted\":%ld,\"ngram_accepted\":%ld,\"preemptions\":%ld,\"outputs\":[",
               n, total, end-start, total/(end-start), mean_first/n, last_first, seconds,
               total-tokens_at_barrier, rate, rate/n, match ? "true" : "false",
               rt.e->st.shared_tokens-before.shared_tokens, rt.e->st.prefill_rows-before.prefill_rows,
               rt.e->st.ng_drafted-before.ng_drafted, rt.e->st.ng_accepted-before.ng_accepted,
               rt.e->st.preemptions-before.preemptions);
        for (int i = 0; i < n; i++) {
            seq_t *s = &rt.e->seqs[ids[i]];
            char *text = tok_decode(rt.tok, s->gen, s->n_gen);
            printf("%s{\"id\":", i ? "," : ""); quoted(tasks[i].id);
            printf(",\"prompt_tokens\":%d,\"text\":", tasks[i].n); quoted(text); free(text);
            printf(",\"tokens\":[");
            for (int t = 0; t < s->n_gen; t++) printf("%s%d", t ? "," : "", s->gen[t]);
            printf("]}");
        }
        printf("]}\n"); fflush(stdout);
        fprintf(stderr, "[agents] %d: %.3f aggregate decode tok/s, %.3f per agent, %.2fs wall\n",
                n, rate, rate/n, end-start);
        for (int i = 0; i < n; i++) engine_release(rt.e, ids[i]);
        if (!match) { fprintf(stderr, "first agent changed under batching\n"); return 2; }
    }
    free(reference);
    for (int i = 0; i < nt; i++) { free(tasks[i].id); free(tasks[i].tokens); }
    engine_free(rt.e);
    return 0;
}
