// Single-sequence generation: correctness smoke test + baseline speed.
//   fleet generate --model llama1b|qwen3|glimmer|path --backend cpu|metal --prompt "..." --n 64 [--temp 0]
#include "common.h"

int main(int argc, char **argv) {
    runtime rt = load_runtime(argc, argv);
    const char *prompt = arg_str(argc, argv, "--prompt", "What is the capital of France? Answer in one sentence.");
    int32_t *ids;
    int n = encode_chat(rt.tok, prompt, !arg_flag(argc, argv, "--think"), &ids);
    fprintf(stderr, "[prompt] %d tokens\n", n);
    sample_params p = sample_defaults();
    p.temperature = (float)arg_num(argc, argv, "--temp", 0);
    p.max_new = (int)arg_num(argc, argv, "--n", 64);
    engine *e = rt.e;
    int id = engine_create(e, ids, n);
    engine_start(e, id, p);
    double t0 = now_secs();
    int shown = 0;
    while (!e->seqs[id].done && engine_step(e, &id, 1)) {
        seq_t *s = &e->seqs[id];
        for (; shown < s->n_gen; shown++) {
            char buf[512];
            size_t k = tok_piece(rt.tok, s->gen[shown], buf, sizeof buf);
            fwrite(buf, 1, k, stdout);
        }
        fflush(stdout);
    }
    seq_t *s = &e->seqs[id];
    double ttft = s->t_first - t0, dt = now_secs() - s->t_first;
    printf("\n");
    fprintf(stderr, "[stats] prefill %d tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.2f tok/s)\n", n, ttft, n / ttft, s->n_gen - 1, dt,
            (s->n_gen - 1) / (dt > 0 ? dt : 1e-9));
    if (e->st.spec_drafted)
        fprintf(stderr, "[spec] accepted %ld/%ld drafts (%.1f%%), %.2f tokens/verify; drafter %.2fs, target %.2fs\n",
                e->st.spec_accepted, e->st.spec_drafted, 100.0 * e->st.spec_accepted / e->st.spec_drafted,
                1.0 + (double)e->st.spec_accepted / e->st.spec_steps, e->st.draft_secs, e->st.forward_secs);
    return 0;
}
