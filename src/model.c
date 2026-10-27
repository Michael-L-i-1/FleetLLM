#include "model.h"
#include "quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

const char *arch_name(arch_t a) {
    switch (a) {
    case ARCH_LLAMA: return "llama";
    case ARCH_QWEN2: return "qwen2";
    case ARCH_QWEN3: return "qwen3";
    case ARCH_GLIMMER: return "muse-glimmer";
    }
    return "?";
}

static qmat get_mat(const gguf_ctx *g, const char *name) {
    qmat m = {0};
    const gguf_tensor *t = gguf_tensor_find(g, name);
    if (!t) return m;
    size_t be, bb;
    ggml_block(t->type, &be, &bb);
    m.type = t->type;
    m.cols = (int)t->dims[0];
    m.rows = (int)t->dims[1];
    m.row_bytes = (size_t)m.cols / be * bb;
    m.data = t->data;
    return m;
}

static const float *get_vec(const gguf_ctx *g, const char *name, int expect) {
    const gguf_tensor *t = gguf_tensor_find(g, name);
    if (!t) return NULL;
    if (t->type != GGML_F32) { fprintf(stderr, "%s: expected F32, got %s\n", name, ggml_type_name(t->type)); exit(1); }
    if (expect && (int)t->dims[0] != expect) { fprintf(stderr, "%s: expected %d elems\n", name, expect); exit(1); }
    return (const float *)t->data;
}

static int key_u(const gguf_ctx *g, const char *arch, const char *k, int def) {
    char key[256];
    snprintf(key, sizeof key, "%s.%s", arch, k);
    uint64_t v;
    return gguf_u64(g, key, &v) ? (int)v : def;
}

static float key_f(const gguf_ctx *g, const char *arch, const char *k, float def) {
    char key[256];
    snprintf(key, sizeof key, "%s.%s", arch, k);
    float v;
    return gguf_f32(g, key, &v) ? v : def;
}

model *model_load(const char *path) {
    gguf_ctx *g = gguf_open(path);
    if (!g) return NULL;
    size_t n;
    const char *a = gguf_str(g, "general.architecture", &n);
    char arch[64] = {0};
    if (a) memcpy(arch, a, n < 63 ? n : 63);
    model *m = calloc(1, sizeof *m);
    m->g = g;
    model_cfg *c = &m->c;
    if (!strcmp(arch, "llama")) c->arch = ARCH_LLAMA;
    else if (!strcmp(arch, "qwen2")) c->arch = ARCH_QWEN2;
    else if (!strcmp(arch, "qwen3")) c->arch = ARCH_QWEN3;
    else if (!strcmp(arch, "muse-glimmer")) c->arch = ARCH_GLIMMER;
    else { fprintf(stderr, "architecture '%s' not supported\n", arch); exit(1); }
    bool glimmer = c->arch == ARCH_GLIMMER;
    c->n_layer = key_u(g, arch, "block_count", 0);
    c->d = key_u(g, arch, "embedding_length", 0);
    c->ff = key_u(g, arch, "feed_forward_length", 0);
    c->n_head = key_u(g, arch, "attention.head_count", 0);
    c->n_kv = key_u(g, arch, "attention.head_count_kv", c->n_head);
    c->hd = key_u(g, arch, "attention.key_length", c->d / c->n_head);
    c->window = key_u(g, arch, "attention.sliding_window", 0);
    c->rope_base = key_f(g, arch, "rope.freq_base", 10000.f);
    c->eps = key_f(g, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
    c->post_eps = glimmer ? 1e-8f : c->eps;
    c->logit_scale = key_f(g, arch, "logit_scale", 1.f);
    c->softcap = key_f(g, arch, "final_logit_softcapping", 0.f);
    c->embed_scale = 1.f;
    c->embed_norm = glimmer;
    c->rope_neox = c->arch == ARCH_QWEN2 || c->arch == ARCH_QWEN3;
    c->attn_scale = 1.f / sqrtf((float)c->hd);

    m->tok_embd = get_mat(g, "token_embd.weight");
    m->output = get_mat(g, "output.weight");
    if (!m->output.data) m->output = m->tok_embd; // tied embeddings
    c->vocab = m->output.rows;
    m->out_norm = get_vec(g, "output_norm.weight", c->d);

    // llama.cpp's swa pattern n: every n-th layer (i % n == n-1) is full attention
    int pattern = key_u(g, arch, "attention.sliding_window_pattern", c->window ? 4 : 1);
    m->layers = calloc(c->n_layer, sizeof(layer));
    for (int i = 0; i < c->n_layer; i++) {
        layer *l = &m->layers[i];
        char k[128];
#define T(s) (snprintf(k, sizeof k, "blk.%d.%s", i, s), k)
        l->attn_norm = get_vec(g, T("attn_norm.weight"), c->d);
        l->ffn_norm = get_vec(g, T("ffn_norm.weight"), c->d);
        l->post_attn_norm = get_vec(g, T("post_attention_norm.weight"), c->d);
        l->post_ffn_norm = get_vec(g, T("post_ffw_norm.weight"), c->d);
        l->q_norm = get_vec(g, T("attn_q_norm.weight"), c->hd);
        l->k_norm = get_vec(g, T("attn_k_norm.weight"), c->hd);
        l->bq = get_vec(g, T("attn_q.bias"), 0);
        l->bk = get_vec(g, T("attn_k.bias"), 0);
        l->bv = get_vec(g, T("attn_v.bias"), 0);
        l->wq = get_mat(g, T("attn_q.weight"));
        l->wk = get_mat(g, T("attn_k.weight"));
        l->wv = get_mat(g, T("attn_v.weight"));
        l->wg = get_mat(g, T("attn_gate.weight"));
        l->wo = get_mat(g, T("attn_output.weight"));
        l->w_gate = get_mat(g, T("ffn_gate.weight"));
        l->w_up = get_mat(g, T("ffn_up.weight"));
        l->w_down = get_mat(g, T("ffn_down.weight"));
#undef T
        if (!l->attn_norm || !l->ffn_norm || !l->wq.data || !l->wk.data || !l->wv.data || !l->wo.data || !l->w_gate.data || !l->w_up.data || !l->w_down.data) {
            fprintf(stderr, "layer %d: missing tensors\n", i);
            exit(1);
        }
        l->swa = c->window > 0 && i % pattern != pattern - 1;
        l->rope = !glimmer || l->swa; // glimmer: RoPE on SWA layers, NoPE on full layers
        l->pool_layer = l->swa ? c->n_swa++ : c->n_full++;
    }
    const float *factors = get_vec(g, "rope_freqs.weight", 0);
    m->inv_freq = malloc(sizeof(float) * c->hd / 2);
    for (int i = 0; i < c->hd / 2; i++) {
        float f = powf(c->rope_base, -(float)(2 * i) / (float)c->hd);
        m->inv_freq[i] = factors ? f / factors[i] : f;
    }
    return m;
}

void model_free(model *m) {
    if (!m) return;
    free(m->layers);
    free(m->inv_freq);
    gguf_close(m->g);
    free(m);
}

static size_t mat_bytes(const qmat *q) { return q->data ? q->row_bytes * (size_t)q->rows : 0; }

size_t model_streamed_bytes(const model *m) {
    size_t n = mat_bytes(&m->output);
    for (int i = 0; i < m->c.n_layer; i++) {
        const layer *l = &m->layers[i];
        n += mat_bytes(&l->wq) + mat_bytes(&l->wk) + mat_bytes(&l->wv) + mat_bytes(&l->wg) + mat_bytes(&l->wo) +
             mat_bytes(&l->w_gate) + mat_bytes(&l->w_up) + mat_bytes(&l->w_down);
    }
    return n;
}

static size_t pin(const qmat *q) {
    if (!q->data) return 0;
    size_t n = mat_bytes(q);
    madvise((void *)q->data, n, MADV_WILLNEED);
    volatile uint8_t s = 0;
    for (size_t i = 0; i < n; i += 16384) s += q->data[i];
    (void)s;
    return mlock(q->data, n) == 0 ? n : 0;
}

size_t model_pin(const model *m) {
    size_t n = pin(&m->output);
    for (int i = 0; i < m->c.n_layer; i++) {
        const layer *l = &m->layers[i];
        n += pin(&l->wq) + pin(&l->wk) + pin(&l->wv) + pin(&l->wg) + pin(&l->wo) + pin(&l->w_gate) + pin(&l->w_up) + pin(&l->w_down);
    }
    return n;
}

void model_embed(const model *m, const int32_t *tokens, int n, float *x) {
    const model_cfg *c = &m->c;
    for (int r = 0; r < n; r++) {
        float *xr = x + (size_t)r * c->d;
        dequant_row(m->tok_embd.type, m->tok_embd.data + (size_t)tokens[r] * m->tok_embd.row_bytes, xr, c->d);
        if (c->embed_scale != 1.f) for (int i = 0; i < c->d; i++) xr[i] *= c->embed_scale;
        if (c->embed_norm) {
            double ss = 0;
            for (int i = 0; i < c->d; i++) ss += (double)xr[i] * xr[i];
            float s = 1.f / sqrtf((float)(ss / c->d) + c->eps);
            for (int i = 0; i < c->d; i++) xr[i] *= s;
        }
    }
}
