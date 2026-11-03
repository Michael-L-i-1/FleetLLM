// Reproducible single-agent task sweep, one model load per configuration.
// fleet eval --model PATH --tasks bench/pi-quality.jsonl [normal runtime flags]
// JSONL results include tokens and decode-only draft/target timing counters.
#include "common.h"
#include "json.h"

static void quoted(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') {
            putchar('\\');
            putchar(*p);
        } else if (*p < 32)
            printf("\\u%04x", *p);
        else
            putchar(*p);
    }
    putchar('"');
}

int main(int argc, char **argv) {
    const char *path = arg_str(argc, argv, "--tasks", NULL);
    if (!path) {
        fprintf(stderr, "usage: fleet eval --tasks tasks.jsonl [runtime flags]\n");
        return 1;
    }
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        return 1;
    }
    runtime rt = load_runtime(argc, argv);
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    while ((len = getline(&line, &cap, f)) >= 0) {
        if (len <= 1)
            continue;
        json *j = json_parse(line, len);
        if (!j)
            return 1;
        const jnode *root = json_root(j);
        const char *id = json_str(j, root, "id"), *prompt = json_str(j, root, "prompt");
        int limit = (int)json_num(j, root, "max_tokens", 64);
        if (!id || !prompt || limit < 1 || limit > 4096) {
            fprintf(stderr, "invalid task\n");
            return 1;
        }
        int32_t *ids;
        int n = encode_chat(rt.tok, prompt, true, &ids);
        int seq = engine_create_fresh(rt.e, ids, n);
        sample_params params = sample_defaults();
        params.temperature = 0;
        params.max_new = limit;
        engine_start(rt.e, seq, params);
        engine_stats before = rt.e->st, first = before;
        bool seen = false;
        double start = now_secs();
        while (!rt.e->seqs[seq].done && engine_step(rt.e, &seq, 1)) {
            if (!seen && rt.e->seqs[seq].n_gen) {
                first = rt.e->st;
                seen = true;
            }
        }
        double end = now_secs();
        seq_t *s = &rt.e->seqs[seq];
        double prefill = seen ? s->t_first - start : end - start;
        double decode = seen ? end - s->t_first : 0;
        char *text = tok_decode(rt.tok, s->gen, s->n_gen);
        printf("{\"id\":");
        quoted(id);
        printf(",\"prompt_tokens\":%d,\"generated_tokens\":%d,\"prefill_seconds\":%.6f,\"decode_seconds\":%.6f,"
               "\"decode_tokens_per_second\":%.6f,",
               n, s->n_gen, prefill, decode, s->n_gen > 1 && decode > 0 ? (s->n_gen - 1) / decode : 0);
        printf(
            "\"decode_draft_seconds\":%.6f,\"decode_target_seconds\":%.6f,\"drafted\":%ld,\"accepted\":%ld,\"text\":",
            rt.e->st.draft_secs - first.draft_secs, rt.e->st.forward_secs - first.forward_secs,
            rt.e->st.spec_drafted - before.spec_drafted, rt.e->st.spec_accepted - before.spec_accepted);
        quoted(text);
        printf(",\"tokens\":[");
        for (int i = 0; i < s->n_gen; i++)
            printf("%s%d", i ? "," : "", s->gen[i]);
        printf("],\"input_tokens\":[");
        for (int i = 0; i < n; i++)
            printf("%s%d", i ? "," : "", ids[i]);
        printf("]}\n");
        fflush(stdout);
        fprintf(stderr, "[eval] %s: %d tokens, %.2f tok/s\n", id, s->n_gen,
                s->n_gen > 1 && decode > 0 ? (s->n_gen - 1) / decode : 0);
        free(ids);
        free(text);
        engine_release(rt.e, seq);
        json_free(j);
    }
    free(line);
    fclose(f);
    return 0;
}
