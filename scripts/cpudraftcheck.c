// Fixture driver for cpudraftcheck.py. Include the backend to inspect hidden
// states against an independent dense PyTorch reference, not just token ids.
#include "../src/cpu.c"
#include "engine.h"

static void *test_alloc(void *user, size_t bytes, void **handle) { return cpu_alloc(user, bytes, handle); }

static int approximation_check(void) {
    cpu_state s = {0}; backend b = { .forward = cpu_forward, .impl = &s };
    model m = {0}; m.c.n_layer = 4;
    if (!cpu_backend_approximate(&b, &m, .5f, "1,3") || !s.skip_layers[1] || !s.skip_layers[3]) return 1;
    const char *bad[] = {"4", "-1", "1,", "x", "0,1,2,3"};
    for (int i = 0; i < 5; i++) if (cpu_backend_approximate(&b, &m, .5f, bad[i])) return 1;
    if (cpu_backend_approximate(&b, &m, NAN, NULL) || cpu_backend_approximate(&b, &m, 1, NULL)) return 1;
    float batched[2 * 128], single[2 * 128];
    const float value[] = {1, 0, 3, 2};
    for (int i = 0; i < 256; i++) batched[i] = value[(i / 32) % 4];
    memcpy(single, batched, sizeof single);
    prune_ffn(&s, batched, 2, 128);
    for (int r = 0; r < 2; r++) prune_ffn(&s, single + r * 128, 1, 128);
    if (memcmp(single, batched, sizeof single)) return 1;
    for (int i = 0; i < 256; i++) if (batched[i] != ((i / 32) % 4 < 2 ? 0 : value[(i / 32) % 4])) return 1;
    free(s.skip_layers); free(s.prune);
    puts("approximation: option validation, deterministic energy selection and batch independence PASS");
    return 0;
}

static int capture_check(const char *path) {
    model *m = model_load(path);
    backend *be = cpu_backend_new(NULL);
    be->n_cap = 2; be->cap_layers[0] = 0; be->cap_layers[1] = 2;
    kv_cache kv; seq_pages pg = {0}; const seq_pages *pages[] = { &pg };
    kv_init(&kv, 3, 0, 2, 16, 0, 1 << 20, 4, test_alloc, be);
    row_t rows[19];
    for (int i = 0; i < 19; i++) {
        if (!kv_reserve(&kv, &pg, i)) return 1;
        pg.len = i + 1;
        rows[i] = (row_t){ .seq = 0, .pos = i, .ctx_end = i, .token = (7 * i) % 64 };
    }
    if (be->forward(be, m, &kv, pages, rows, 19, NULL)) return 1;
    cpu_state *captured = be->impl;
    for (int ci = 0; ci < 2; ci++) {
        backend *prefix_be = cpu_backend_new(NULL);
        model prefix = *m; prefix.c.n_layer = be->cap_layers[ci] + 1;
        if (prefix_be->forward(prefix_be, &prefix, &kv, pages, rows, 19, NULL)) return 1;
        cpu_state *state = prefix_be->impl;
        for (int r = 0; r < 19; r++)
            if (memcmp(captured->capture + (size_t)r * 128 + ci * 64, state->x + (size_t)r * 64, 64 * sizeof(float))) return 1;
        prefix_be->destroy(prefix_be);
    }
    be->destroy(be); model_free(m);
    puts("target hidden capture: layer selection and row layout PASS");
    return 0;
}

static int engine_check(const char *target_path, const char *draft_path, bool constant, bool reference, int elem, int k) {
    model *tm = model_load(target_path), *dm = model_load(draft_path);
    float *zero = constant ? calloc((size_t)tm->output.rows * tm->output.cols, sizeof(float)) : NULL;
    if (zero) { tm->output.type = GGML_F32; tm->output.row_bytes = tm->output.cols * sizeof(float); tm->output.data = (const uint8_t *)zero; }
    backend *plain = reference ? cpu_backend_new(NULL) : cpu_backend_new(tm);
    backend *spec = reference ? cpu_backend_new(NULL) : cpu_backend_new(tm);
    plain->kv_elem = spec->kv_elem = elem;
    engine *a = engine_new(tm, plain, NULL, 1 << 20), *b = engine_new(tm, spec, NULL, 1 << 20);
    if (!engine_set_drafter(b, dm, 1 << 20, k)) return 1;
    a->row_budget = b->row_budget = 5;
    b->spec_rows = 8;
    int ai[2], bi[2];
    sample_params p = sample_defaults();
    p.temperature = 0; p.max_new = 25; p.ignore_eos = true; p.n_stop = 1; p.stop[0] = -1;
    for (int i = 0; i < 2; i++) {
        int32_t prompt[19];
        for (int j = 0; j < 17 + i * 2; j++) prompt[j] = (j * 7 + i * 3) % tm->c.vocab;
        ai[i] = engine_create_fresh(a, prompt, 17 + i * 2);
        bi[i] = engine_create_fresh(b, prompt, 17 + i * 2);
        engine_start(a, ai[i], p); engine_start(b, bi[i], p);
    }
    engine_generate(a, ai, 2); engine_generate(b, bi, 2);
    int bad = 0;
    for (int i = 0; i < 2; i++) {
        seq_t *x = &a->seqs[ai[i]], *y = &b->seqs[bi[i]];
        bad |= x->n_gen != 25 || y->n_gen != 25 || memcmp(x->gen, y->gen, 25 * sizeof(int32_t));
    }
    bad |= b->st.spec_drafted == 0 || (constant ? b->st.spec_accepted != b->st.spec_drafted : b->st.spec_accepted == b->st.spec_drafted);
    printf("engine %s kv%d k%d %s: accepted %ld/%ld, tokens %s\n", reference ? "reference" : "native", elem, k,
           constant ? "full acceptance" : "rejection", b->st.spec_accepted, b->st.spec_drafted, bad ? "FAIL" : "PASS");
    engine_free(a); engine_free(b); plain->destroy(plain); spec->destroy(spec);
    model_free(tm); model_free(dm); free(zero);
    return bad;
}

int main(int argc, char **argv) {
    tp_init(2);
    if (argc == 4 && !strcmp(argv[1], "engine")) {
        if (approximation_check() || capture_check(argv[2])) return 1;
        uint8_t data[256] = {0};
        const int types[] = { GGML_Q5_K, GGML_Q5_0, GGML_Q5_1 };
        for (size_t i = 0; i < sizeof types / sizeof *types; i++) {
            qmat w = { .type = types[i], .rows = 1, .cols = 256, .data = data };
            if (cq_from_qmat(&w)) { fputs("compact weights unexpectedly expanded\n", stderr); return 1; }
        }
        puts("compact K-quants: no implicit 8-bit expansion PASS");
        for (int ref = 0; ref < 2; ref++) for (int elem = 1; elem <= 4; elem *= 2)
            for (int k = 1; k <= 3; k += 2) for (int constant = 0; constant < 2; constant++)
                if (engine_check(argv[2], argv[3], constant, ref, elem, k)) return 1;
        return 0;
    }
    if (argc != 7) return 2;
    model *tm = model_load(argv[1]), *dm = model_load(argv[2]);
    FILE *f = fopen(argv[3], "rb");
    int header[3];
    if (!f || fread(header, sizeof(int), 3, f) != 3) return 2;
    int nc = header[0], nb = header[1], B = header[2];
    backend *be = cpu_backend_new(NULL);
    if (be->attach(be, dm)) return 1;
    cpu_state *s = be->impl;
    s->capture_width = dm->fc.cols; s->capture_rows = nc;
    s->capture = malloc((size_t)nc * dm->fc.cols * sizeof(float));
    draft_row *ctx = malloc(nc * sizeof(draft_row)), *blk = malloc(nb * sizeof(draft_row));
    if (fread(s->capture, sizeof(float), (size_t)nc * dm->fc.cols, f) != (size_t)nc * dm->fc.cols ||
        fread(ctx, sizeof(draft_row), nc, f) != (size_t)nc || fread(blk, sizeof(draft_row), nb, f) != (size_t)nb) return 2;
    fclose(f);
    kv_cache kv; seq_pages pg[2] = {0}; const seq_pages *pages[] = { &pg[0], &pg[1] };
    kv_init(&kv, dm->c.n_layer, 0, dm->c.n_kv, dm->c.hd, 0, 1 << 20, atoi(argv[5]), test_alloc, be);
    for (int i = 0; i < nc + nb; i++) {
        const draft_row *r = i < nc ? &ctx[i] : &blk[i - nc];
        if (!kv_reserve(&kv, &pg[r->seq], r->pos)) return 2;
        pg[r->seq].len = r->pos + 1;
    }
    int32_t out[64];
    if (atoi(argv[6])) {
        if (be->draft(be, dm, tm, &kv, pages, ctx, nc, NULL, 0, B, out) ||
            be->draft(be, dm, tm, &kv, pages, NULL, 0, blk, nb, B, out)) return 1;
    } else if (be->draft(be, dm, tm, &kv, pages, ctx, nc, blk, nb, B, out)) return 1;
    f = fopen(argv[4], "wb");
    if (!f) return 2;
    fwrite(s->draft_state->x, sizeof(float), (size_t)nb * dm->c.d, f);
    fwrite(out, sizeof(int32_t), nb / B * (B - 1), f);
    fclose(f);
    be->destroy(be); model_free(tm); model_free(dm); free(ctx); free(blk);
    return 0;
}
