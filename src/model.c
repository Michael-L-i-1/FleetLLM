#include "model.h"
#include "json.h"
#include "quant.h"
#include "threadpool.h"
#include <math.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#ifdef __linux__
#include <fcntl.h>
#endif

const char *arch_name(arch_t a) {
    switch (a) {
    case ARCH_LLAMA: return "llama";
    case ARCH_QWEN2: return "qwen2";
    case ARCH_QWEN3: return "qwen3";
    case ARCH_GLIMMER: return "muse-glimmer";
    case ARCH_DFLASH: return "dflash-drafter";
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

static model *model_load_st(const char *dir);

model *model_load(const char *path) {
    struct stat sb;
    if (stat(path, &sb) == 0 && S_ISDIR(sb.st_mode)) return model_load_st(path);
    gguf_ctx *g = gguf_open(path);
    if (!g) return NULL;
    size_t n;
    const char *a = gguf_str(g, "general.architecture", &n);
    char arch[64] = {0};
    if (a) memcpy(arch, a, n < 63 ? n : 63);
    model *m = calloc(1, sizeof *m);
    m->g = g;
    m->regions[m->n_regions++] = (mem_region){ g->map, g->size };
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
    if (m->g) gguf_close(m->g);
    if (m->st) st_close(m->st);
    if (m->aux) munmap(m->aux, m->aux_len);
    if (m->arena) munmap(m->arena, m->arena_len);
    if (m->host_q) munmap(m->host_q, m->host_q_len);
    free(m);
}

size_t qmat_bytes(const qmat *q) {
    if (!q->data) return 0;
    size_t n = q->row_bytes * (size_t)q->rows;
    if (q->type == QT_AFFINE) n += 2 * 2 * (size_t)q->rows * (q->cols / q->group);
    return n;
}
#define mat_bytes qmat_bytes

size_t model_streamed_bytes(const model *m) {
    size_t n = mat_bytes(&m->output) + mat_bytes(&m->fc);
    for (int i = 0; i < m->c.n_layer; i++) {
        const layer *l = &m->layers[i];
        n += mat_bytes(&l->wq) + mat_bytes(&l->wk) + mat_bytes(&l->wv) + mat_bytes(&l->wg) + mat_bytes(&l->wo) +
             mat_bytes(&l->w_gate) + mat_bytes(&l->w_up) + mat_bytes(&l->w_down);
    }
    return n;
}

static size_t pin_range(const void *p, size_t n) {
    // mlock wants page-aligned ranges
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)16383, b = ((uintptr_t)p + n + 16383) & ~(uintptr_t)16383;
    madvise((void *)a, b - a, MADV_WILLNEED);
    volatile uint8_t s = 0;
    for (uintptr_t i = a; i < b; i += 16384) s += *(const uint8_t *)i;
    (void)s;
    return mlock((void *)a, b - a) == 0 ? n : 0;
}

static size_t pin(const qmat *q) {
    if (!q->data || q->gpu) return 0;
    size_t n = pin_range(q->data, q->row_bytes * (size_t)q->rows);
    if (q->type == QT_AFFINE) {
        size_t sb = 2 * (size_t)q->rows * (q->cols / q->group);
        n += pin_range(q->scales, sb) + pin_range(q->biases, sb);
    }
    return n;
}

size_t model_pin(const model *m) {
    size_t n = pin(&m->output);
    for (int i = 0; i < m->c.n_layer; i++) {
        const layer *l = &m->layers[i];
        n += pin(&l->wq) + pin(&l->wk) + pin(&l->wv) + pin(&l->wg) + pin(&l->wo) + pin(&l->w_gate) + pin(&l->w_up) + pin(&l->w_down);
    }
    return n;
}

void model_embed_raw(const model *m, const int32_t *tokens, int n, float *x) {
    for (int r = 0; r < n; r++) qmat_row(&m->tok_embd, tokens[r], x + (size_t)r * m->c.d);
}

void model_embed(const model *m, const int32_t *tokens, int n, float *x) {
    const model_cfg *c = &m->c;
    for (int r = 0; r < n; r++) {
        float *xr = x + (size_t)r * c->d;
        qmat_row(&m->tok_embd, tokens[r], xr);
        if (c->embed_scale != 1.f) for (int i = 0; i < c->d; i++) xr[i] *= c->embed_scale;
        if (c->embed_norm) {
            double ss = 0;
            for (int i = 0; i < c->d; i++) ss += (double)xr[i] * xr[i];
            float s = 1.f / sqrtf((float)(ss / c->d) + c->eps);
            for (int i = 0; i < c->d; i++) xr[i] *= s;
        }
    }
}

static inline float bf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void qmat_row(const qmat *w, int r, float *dst) {
    const uint8_t *src = w->data + (size_t)r * w->row_bytes;
    if (w->type != QT_AFFINE) { dequant_row(w->type, src, dst, w->cols); return; }
    int ng = w->cols / w->group, bits = w->bits;
    uint32_t mask = (1u << bits) - 1;
    for (int g = 0; g < ng; g++) {
        uint16_t hs, hb; // safetensors data may be misaligned
        memcpy(&hs, w->scales + (size_t)r * ng + g, 2);
        memcpy(&hb, w->biases + (size_t)r * ng + g, 2);
        float sc = w->scales_f16 ? fp16_to_f32(hs) : bf16(hs), bi = w->scales_f16 ? fp16_to_f32(hb) : bf16(hb);
        for (int i = g * w->group; i < (g + 1) * w->group; i++) {
            // little-endian bit stream: value i occupies bits [i*bits, (i+1)*bits)
            size_t bit = (size_t)i * bits, byte = bit >> 3;
            uint32_t v = src[byte];
            if (byte + 1 < w->row_bytes) v |= (uint32_t)src[byte + 1] << 8;
            dst[i] = sc * (float)((v >> (bit & 7)) & mask) + bi;
        }
    }
}

// ---------------------------------------------------------------- safetensors / MLX

typedef struct { const st_ctx *st; const char *prefix; } names;

static const st_tensor *st_get(const names *nm, const char *suffix) {
    char k[512];
    snprintf(k, sizeof k, "%s%s", nm->prefix, suffix);
    return st_find(nm->st, k);
}

// Linear layer `base` (e.g. "model.layers.0.self_attn.q_proj") with in_features
// cols: MLX affine (weight U32 + scales + biases) or unquantized f16/bf16.
static qmat st_mat(const names *nm, const char *base, int cols) {
    qmat m = {0};
    char k[512];
    snprintf(k, sizeof k, "%s.weight", base);
    const st_tensor *w = st_get(nm, k);
    if (!w) return m;
    snprintf(k, sizeof k, "%s.scales", base);
    const st_tensor *sc = st_get(nm, k);
    snprintf(k, sizeof k, "%s.biases", base);
    const st_tensor *bi = st_get(nm, k);
    m.rows = (int)w->shape[0];
    m.data = w->data;
    if (sc && bi && w->dtype == ST_U32) {
        if (sc->dtype != bi->dtype || (sc->dtype != ST_BF16 && sc->dtype != ST_F16)) { fprintf(stderr, "%s: scales/biases must be bf16 or f16\n", base); exit(1); }
        m.type = QT_AFFINE;
        m.scales_f16 = sc->dtype == ST_F16;
        m.cols = cols;
        m.bits = (int)(w->shape[1] * 32 / m.cols);
        m.group = (int)(m.cols / sc->shape[1]);
        m.row_bytes = (size_t)w->shape[1] * 4;
        m.scales = (const uint16_t *)sc->data;
        m.biases = (const uint16_t *)bi->data;
        if (m.bits * m.cols != w->shape[1] * 32 || m.group * sc->shape[1] != m.cols) { fprintf(stderr, "%s: inconsistent quantized shape\n", base); exit(1); }
    } else if (w->dtype == ST_BF16 || w->dtype == ST_F16) {
        m.type = w->dtype == ST_BF16 ? GGML_BF16 : GGML_F16;
        m.cols = (int)w->shape[1];
        m.row_bytes = (size_t)m.cols * 2;
    } else if (w->dtype == ST_F32) {
        m.type = GGML_F32;
        m.cols = (int)w->shape[1];
        m.row_bytes = (size_t)m.cols * 4;
    } else { fprintf(stderr, "%s: unsupported dtype %s\n", base, st_dtype_name(w->dtype)); exit(1); }
    return m;
}

static float *aux_alloc(model *m, size_t n) {
    size_t bytes = (n * sizeof(float) + 63) & ~(size_t)63;
    if (m->aux_used + bytes > m->aux_len) { fprintf(stderr, "model: aux arena exhausted\n"); exit(1); }
    float *p = (float *)(m->aux + m->aux_used);
    m->aux_used += bytes;
    return p;
}

// Norm weight as f32 (optionally 1 + w for zero-centered norms); NULL if absent.
static const float *st_vec(model *m, const names *nm, const char *name, int expect, float add) {
    const st_tensor *t = st_get(nm, name);
    if (!t) return NULL;
    if (t->shape[0] != expect || t->n_dims != 1) { fprintf(stderr, "%s: expected %d elems\n", name, expect); exit(1); }
    float *v = aux_alloc(m, expect);
    for (int i = 0; i < expect; i++) {
        float x;
        if (t->dtype == ST_F32) memcpy(&x, t->data + 4 * i, 4);
        else {
            uint16_t h;
            memcpy(&h, t->data + 2 * i, 2);
            x = t->dtype == ST_BF16 ? bf16(h) : fp16_to_f32(h);
        }
        v[i] = x + add;
    }
    return v;
}

static const float *const_vec(model *m, int n, float val) {
    float *v = aux_alloc(m, n);
    for (int i = 0; i < n; i++) v[i] = val;
    return v;
}

// safetensors data is only as aligned as its JSON header length makes it; GPU
// kernels want 16-byte aligned rows, so misaligned matrices the GPU reads are
// copied once into an aligned arena (same layout). The embedding table stays
// mapped unless it doubles as the output projection.
static size_t up64(size_t n) { return (n + 63) & ~(size_t)63; }
static bool misaligned(const qmat *q) {
    return q->data && !q->gpu && (((uintptr_t)q->data | (uintptr_t)q->scales | (uintptr_t)q->biases) & 15);
}
static size_t mat_parts(const qmat *q, size_t *wb, size_t *sb) {
    *wb = q->row_bytes * (size_t)q->rows;
    *sb = q->type == QT_AFFINE ? 2 * (size_t)q->rows * (q->cols / q->group) : 0;
    return up64(*wb) + 2 * up64(*sb);
}

void model_align_mats(model *m) {
    qmat *mats[8 * 1024 + 1];
    int n = 0;
    bool tied = m->output.data == m->tok_embd.data;
    mats[n++] = &m->output;
    for (int i = 0; i < m->c.n_layer; i++) {
        layer *l = &m->layers[i];
        qmat *ls[] = { &l->wq, &l->wk, &l->wv, &l->wg, &l->wo, &l->w_gate, &l->w_up, &l->w_down };
        for (int k = 0; k < 8; k++) mats[n++] = ls[k];
    }
    size_t total = 0, wb, sb;
    for (int i = 0; i < n; i++) if (misaligned(mats[i])) total += mat_parts(mats[i], &wb, &sb);
    if (!total) return;
    m->arena_len = (total + 16383) & ~(size_t)16383;
    m->arena = mmap(NULL, m->arena_len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (m->arena == MAP_FAILED) { fprintf(stderr, "model: cannot allocate %.2f GB for aligned weights\n", m->arena_len / 1e9); exit(1); }
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        qmat *q = mats[i];
        if (!misaligned(q)) continue;
        mat_parts(q, &wb, &sb);
        uint8_t *d = m->arena + off;
        memcpy(d, q->data, wb);
        q->data = d;
        off += up64(wb);
        if (sb) {
            memcpy(m->arena + off, q->scales, sb);
            q->scales = (const uint16_t *)(m->arena + off);
            off += up64(sb);
            memcpy(m->arena + off, q->biases, sb);
            q->biases = (const uint16_t *)(m->arena + off);
            off += up64(sb);
        }
    }
    if (tied) m->tok_embd = m->output;
    m->regions[m->n_regions++] = (mem_region){ m->arena, m->arena_len };
}

static model *model_load_dflash(const char *dir, json *j);

static model *model_load_st(const char *dir) {
    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", dir);
    json *j = json_load_file(path);
    if (!j) { fprintf(stderr, "%s: cannot read\n", path); return NULL; }
    const jnode *root = json_root(j), *tc = json_get(j, root, "text_config");
    const jnode *cfg = tc ? tc : root;
    const char *mt = json_str(j, root, "model_type");
    model *m = calloc(1, sizeof *m);
    snprintf(m->dir, sizeof m->dir, "%s", dir);
    model_cfg *c = &m->c;
    if (!mt) mt = "?";
    if (!strcmp(mt, "llama")) c->arch = ARCH_LLAMA;
    else if (!strcmp(mt, "qwen2")) c->arch = ARCH_QWEN2;
    else if (!strcmp(mt, "qwen3")) c->arch = ARCH_QWEN3;
    else if (!strcmp(mt, "muse_glimmer") || !strcmp(mt, "muse_glimmer_text")) c->arch = ARCH_GLIMMER;
    else if (!strcmp(mt, "muse_glimmer_assistant")) { free(m); return model_load_dflash(dir, j); }
    else { fprintf(stderr, "model_type '%s' not supported\n", mt); exit(1); }
    bool glimmer = c->arch == ARCH_GLIMMER;
#define NUM(k, d) json_num(j, cfg, k, d)
    c->n_layer = (int)NUM("num_hidden_layers", 0);
    c->d = (int)NUM("hidden_size", 0);
    c->ff = (int)NUM("intermediate_size", 0);
    c->n_head = (int)NUM("num_attention_heads", 0);
    c->n_kv = (int)NUM("num_key_value_heads", c->n_head);
    c->hd = (int)NUM("head_dim", c->d / c->n_head);
    c->eps = (float)NUM("rms_norm_eps", 1e-6);
    c->post_eps = glimmer ? (float)NUM("post_norm_eps", 1e-8) : c->eps;
    c->rope_base = (float)NUM("rope_theta", 10000.0);
    const jnode *rp = json_get(j, cfg, "rope_parameters");
    if (rp) c->rope_base = (float)json_num(j, rp, "rope_theta", c->rope_base);
    c->window = glimmer || json_bool(j, cfg, "use_sliding_window", false) ? (int)NUM("sliding_window", 0) : 0;
    c->logit_scale = glimmer ? (float)NUM("output_multiplier", 1.0) : 1.f;
    c->softcap = (float)NUM("final_logit_softcapping", 0.0);
    c->embed_scale = 1.f;
    c->embed_norm = glimmer;
    c->rope_neox = true; // HF layout: rotate-half
    c->attn_scale = 1.f / sqrtf((float)c->hd);
    float qk_scale = glimmer ? (float)NUM("qk_scale_factor", 1.0) : 1.f;
    bool tied = json_bool(j, cfg, "tie_word_embeddings", json_bool(j, root, "tie_word_embeddings", false));

    m->st = st_open_dir(dir);
    if (!m->st) exit(1);
    for (int i = 0; i < m->st->n_files; i++) m->regions[m->n_regions++] = (mem_region){ m->st->files[i].map, m->st->files[i].size };
    m->aux_len = ((size_t)c->n_layer * (4 * c->d + 2 * c->hd + 3 * 1024) * 4 + c->d * 4 + (1 << 20) + 16383) & ~(size_t)16383;
    m->aux = mmap(NULL, m->aux_len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    m->regions[m->n_regions++] = (mem_region){ m->aux, m->aux_len };
    // Multimodal checkpoints nest the text tower under language_model.
    names nm = { m->st, st_find(m->st, "language_model.model.embed_tokens.weight") ? "language_model." : "" };
    m->tok_embd = st_mat(&nm, "model.embed_tokens", c->d);
    m->output = tied ? m->tok_embd : st_mat(&nm, "lm_head", c->d);
    if (!m->tok_embd.data || !m->output.data) { fprintf(stderr, "%s: missing embeddings / lm_head\n", dir); exit(1); }
    c->vocab = m->output.rows;
    m->out_norm = st_vec(m, &nm, "model.norm.weight", c->d, 0);

    const jnode *lt = json_get(j, cfg, "layer_types"), *lr = json_get(j, cfg, "layer_rope_theta");
    m->layers = calloc(c->n_layer, sizeof(layer));
    float center = glimmer ? 1.f : 0.f; // glimmer's per-block norms scale by (1 + w)
    for (int i = 0; i < c->n_layer; i++) {
        layer *l = &m->layers[i];
        char k[256];
#define L(s) (snprintf(k, sizeof k, "model.layers.%d.%s", i, s), k)
        l->attn_norm = st_vec(m, &nm, L("input_layernorm.weight"), c->d, center);
        if (st_get(&nm, L("pre_feedforward_layernorm.weight"))) {
            // sandwich norms (gemma-style): post_attention_layernorm normalizes the attention output
            l->post_attn_norm = st_vec(m, &nm, L("post_attention_layernorm.weight"), c->d, center);
            l->ffn_norm = st_vec(m, &nm, L("pre_feedforward_layernorm.weight"), c->d, center);
            l->post_ffn_norm = st_vec(m, &nm, L("post_feedforward_layernorm.weight"), c->d, center);
        } else {
            l->ffn_norm = st_vec(m, &nm, L("post_attention_layernorm.weight"), c->d, center);
        }
        l->q_norm = st_vec(m, &nm, L("self_attn.q_norm.weight"), c->hd, 0);
        l->k_norm = st_vec(m, &nm, L("self_attn.k_norm.weight"), c->hd, 0);
        if (glimmer) { // weightless QK-norm; q additionally scaled
            l->q_norm = const_vec(m, c->hd, qk_scale);
            l->k_norm = const_vec(m, c->hd, 1.f);
        }
        l->bq = st_vec(m, &nm, L("self_attn.q_proj.bias"), c->n_head * c->hd, 0);
        l->bk = st_vec(m, &nm, L("self_attn.k_proj.bias"), c->n_kv * c->hd, 0);
        l->bv = st_vec(m, &nm, L("self_attn.v_proj.bias"), c->n_kv * c->hd, 0);
        l->wq = st_mat(&nm, L("self_attn.q_proj"), c->d);
        l->wk = st_mat(&nm, L("self_attn.k_proj"), c->d);
        l->wv = st_mat(&nm, L("self_attn.v_proj"), c->d);
        l->wg = st_mat(&nm, L("self_attn.gate_proj"), c->d);
        l->wo = st_mat(&nm, L("self_attn.o_proj"), c->n_head * c->hd);
        l->w_gate = st_mat(&nm, L("mlp.gate_proj"), c->d);
        l->w_up = st_mat(&nm, L("mlp.up_proj"), c->d);
        l->w_down = st_mat(&nm, L("mlp.down_proj"), c->ff);
#undef L
        if (!l->attn_norm || !l->ffn_norm || !l->wq.data || !l->wk.data || !l->wv.data || !l->wo.data || !l->w_gate.data || !l->w_up.data || !l->w_down.data) {
            fprintf(stderr, "layer %d: missing tensors\n", i);
            exit(1);
        }
        const jnode *t = json_child(j, lt, i), *th = json_child(j, lr, i);
        l->swa = c->window > 0 && (t ? t->type == J_STR && !strcmp(t->s, "sliding_attention") : glimmer && i % 4 != 3);
        l->rope = th ? th->num != 0 : !glimmer || l->swa;
        l->pool_layer = l->swa ? c->n_swa++ : c->n_full++;
    }
    // inverse frequencies, with llama3-style long-context scaling when configured
    m->inv_freq = malloc(sizeof(float) * c->hd / 2);
    const jnode *rs = json_get(j, cfg, "rope_scaling");
    const char *rtype = rs ? json_str(j, rs, "rope_type") : NULL;
    if (!rtype && rs) rtype = json_str(j, rs, "type");
    for (int i = 0; i < c->hd / 2; i++) {
        double f = pow(c->rope_base, -(double)(2 * i) / c->hd);
        if (rtype && !strcmp(rtype, "llama3")) {
            double factor = json_num(j, rs, "factor", 8), lo = json_num(j, rs, "low_freq_factor", 1), hi = json_num(j, rs, "high_freq_factor", 4);
            double old = json_num(j, rs, "original_max_position_embeddings", 8192);
            double wl = 2 * M_PI / f, lo_wl = old / lo, hi_wl = old / hi;
            if (wl > lo_wl) f /= factor;
            else if (wl >= hi_wl) {
                double sm = (old / wl - lo) / (hi - lo);
                f = (1 - sm) * f / factor + sm * f;
            }
        } else if (rtype && strcmp(rtype, "default")) {
            fprintf(stderr, "rope_scaling type '%s' not supported\n", rtype);
            exit(1);
        }
        m->inv_freq[i] = (float)f;
    }
#undef NUM
    json_free(j);
    return m;
}

// ---------------------------------------------------------------- DFlash drafter

static uint16_t to_bf16(float f) { // round to nearest even
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

typedef struct { qmat src, dst; } quant_job;

// bf16/f16/f32 row-major -> MLX affine 4-bit, group 64, bf16 scales/biases
static void quant_rows(void *ctx, int r0, int r1) {
    quant_job *j = ctx;
    const qmat *a = &j->src;
    qmat *q = &j->dst;
    int ng = a->cols / 64;
    float *row = malloc(sizeof(float) * a->cols);
    for (int r = r0; r < r1; r++) {
        qmat_row(a, r, row);
        uint8_t *qr = (uint8_t *)q->data + (size_t)r * q->row_bytes;
        memset(qr, 0, q->row_bytes);
        for (int g = 0; g < ng; g++) {
            const float *w = row + g * 64;
            float lo = w[0], hi = w[0];
            for (int i = 1; i < 64; i++) { lo = w[i] < lo ? w[i] : lo; hi = w[i] > hi ? w[i] : hi; }
            uint16_t sb = to_bf16(hi > lo ? (hi - lo) / 15.f : 1e-8f), bb = to_bf16(lo);
            ((uint16_t *)q->scales)[(size_t)r * ng + g] = sb;
            ((uint16_t *)q->biases)[(size_t)r * ng + g] = bb;
            uint32_t us = (uint32_t)sb << 16, ub = (uint32_t)bb << 16;
            float sc, bi;
            memcpy(&sc, &us, 4);
            memcpy(&bi, &ub, 4);
            for (int i = 0; i < 64; i++) {
                int v = (int)lrintf((w[i] - bi) / sc);
                v = v < 0 ? 0 : v > 15 ? 15 : v;
                qr[(g * 64 + i) / 2] |= (uint8_t)(v << ((i & 1) * 4));
            }
        }
    }
    free(row);
}

static size_t quant_bytes(const qmat *a) { return (size_t)a->rows * a->cols / 2 + 2 * 2 * (size_t)a->rows * (a->cols / 64); }

// Quantize a (16-bit) matrix in place into the model's host_q arena at *off.
static void quantize_into(model *m, qmat *w, size_t *off) {
    if (!w->data) return;
    if (w->type == QT_AFFINE) return; // already quantized checkpoint
    if (w->cols % 64) { fprintf(stderr, "drafter: in-features %d not a multiple of 64\n", w->cols); exit(1); }
    quant_job j = { *w, { 0 } };
    qmat *q = &j.dst;
    q->type = QT_AFFINE;
    q->rows = w->rows;
    q->cols = w->cols;
    q->bits = 4;
    q->group = 64;
    q->row_bytes = (size_t)w->cols / 2;
    uint8_t *base = m->host_q + *off;
    q->data = base;
    q->scales = (const uint16_t *)(base + (size_t)w->rows * q->row_bytes);
    q->biases = q->scales + (size_t)w->rows * (w->cols / 64);
    *off += (quant_bytes(w) + 63) & ~(size_t)63;
    parallel_for(w->rows, 64, quant_rows, &j);
    *w = j.dst;
}

static model *model_load_dflash(const char *dir, json *j) {
    const jnode *cfg = json_root(j);
    model *m = calloc(1, sizeof *m);
    snprintf(m->dir, sizeof m->dir, "%s", dir);
    model_cfg *c = &m->c;
    c->arch = ARCH_DFLASH;
    c->n_layer = (int)json_num(j, cfg, "num_hidden_layers", 5);
    c->d = (int)json_num(j, cfg, "hidden_size", 0);
    c->ff = (int)json_num(j, cfg, "intermediate_size", 0);
    c->n_head = (int)json_num(j, cfg, "num_attention_heads", 0);
    c->n_kv = (int)json_num(j, cfg, "num_key_value_heads", c->n_head);
    c->hd = (int)json_num(j, cfg, "head_dim", c->d / c->n_head);
    c->eps = c->post_eps = (float)json_num(j, cfg, "rms_norm_eps", 1e-5);
    c->window = (int)json_num(j, cfg, "sliding_window", 0);
    c->rope_base = (float)json_num(j, json_get(j, cfg, "rope_parameters"), "rope_theta", 500000.0);
    c->rope_neox = true;
    c->attn_scale = 1.f / sqrtf((float)c->hd);
    c->logit_scale = 1.f;
    c->embed_scale = 1.f;
    m->block_size = (int)json_num(j, cfg, "block_size", 16);
    m->mask_token = (int)json_num(j, cfg, "mask_token_id", -1);
    const jnode *tl = json_get(j, cfg, "target_layer_ids");
    for (const jnode *t = json_first(j, tl); t && m->n_target < 8; t = json_next(j, t)) m->target_layers[m->n_target++] = (int)t->num;
    json_free(j);

    m->st = st_open_dir(dir);
    if (!m->st) exit(1);
    for (int i = 0; i < m->st->n_files; i++) m->regions[m->n_regions++] = (mem_region){ m->st->files[i].map, m->st->files[i].size };
    m->aux_len = ((size_t)c->n_layer * (2 * c->d + 2 * c->hd) * 4 + 2 * c->d * 4 + (1 << 20) + 16383) & ~(size_t)16383;
    m->aux = mmap(NULL, m->aux_len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    m->regions[m->n_regions++] = (mem_region){ m->aux, m->aux_len };
    names nm = { m->st, "" };
    m->fc = st_mat(&nm, "encoder.fc", c->d * m->n_target);
    m->enc_norm = st_vec(m, &nm, "encoder.output_norm_enc.weight", c->d, 0);
    m->out_norm = st_vec(m, &nm, "norm.weight", c->d, 0);
    if (!m->fc.data || !m->enc_norm || !m->out_norm || m->mask_token < 0 || !m->n_target) { fprintf(stderr, "%s: not a DFlash drafter\n", dir); exit(1); }
    m->layers = calloc(c->n_layer, sizeof(layer));
    for (int i = 0; i < c->n_layer; i++) {
        layer *l = &m->layers[i];
        char k[256];
#define L(s) (snprintf(k, sizeof k, "layers.%d.%s", i, s), k)
        l->attn_norm = st_vec(m, &nm, L("input_layernorm.weight"), c->d, 0);
        l->ffn_norm = st_vec(m, &nm, L("post_attention_layernorm.weight"), c->d, 0);
        l->q_norm = st_vec(m, &nm, L("self_attn.q_norm.weight"), c->hd, 0);
        l->k_norm = st_vec(m, &nm, L("self_attn.k_norm.weight"), c->hd, 0);
        l->wq = st_mat(&nm, L("self_attn.q_proj"), c->d);
        l->wk = st_mat(&nm, L("self_attn.k_proj"), c->d);
        l->wv = st_mat(&nm, L("self_attn.v_proj"), c->d);
        l->wo = st_mat(&nm, L("self_attn.o_proj"), c->n_head * c->hd);
        l->w_gate = st_mat(&nm, L("mlp.gate_proj"), c->d);
        l->w_up = st_mat(&nm, L("mlp.up_proj"), c->d);
        l->w_down = st_mat(&nm, L("mlp.down_proj"), c->ff);
#undef L
        if (!l->attn_norm || !l->ffn_norm || !l->wq.data || !l->wk.data || !l->wv.data || !l->wo.data || !l->w_gate.data) { fprintf(stderr, "drafter layer %d: missing tensors\n", i); exit(1); }
        l->swa = false; // drafter KV lives in one pool; the window is applied by the attention bounds
        l->rope = true;
        l->pool_layer = c->n_full++;
    }
    m->inv_freq = malloc(sizeof(float) * c->hd / 2);
    for (int i = 0; i < c->hd / 2; i++) m->inv_freq[i] = (float)pow(c->rope_base, -(double)(2 * i) / c->hd);
    // 16-bit checkpoint: quantize to 4-bit affine (draft quality only affects
    // the acceptance rate; outputs are always verified by the target)
    qmat *mats[1 + 8 * 8];
    int nq = 0;
    mats[nq++] = &m->fc;
    for (int i = 0; i < c->n_layer; i++) {
        layer *l = &m->layers[i];
        qmat *ls[] = { &l->wq, &l->wk, &l->wv, &l->wo, &l->w_gate, &l->w_up, &l->w_down };
        for (int k = 0; k < 7; k++) mats[nq++] = ls[k];
    }
    size_t total = 0, off = 0;
    for (int i = 0; i < nq; i++) if (mats[i]->type != QT_AFFINE) total += (quant_bytes(mats[i]) + 63) & ~(size_t)63;
    if (total) {
        tp_init(0);
        m->host_q_len = (total + 16383) & ~(size_t)16383;
        m->host_q = mmap(NULL, m->host_q_len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        m->regions[m->n_regions++] = (mem_region){ m->host_q, m->host_q_len };
        for (int i = 0; i < nq; i++) quantize_into(m, mats[i], &off);
#ifdef __linux__
        // Norms were copied into aux; converted weights now live in host_q.
        // Drop the checkpoint cache before loading/repacking the target, rather
        // than waiting until the CPU drafter is attached after the target.
        // Mappings stay readable for any already-quantized source tensors.
        for (int i = 0; i < m->st->n_files; i++) {
            const st_file *f = &m->st->files[i];
            madvise((void *)f->map, f->size, MADV_DONTNEED);
            posix_fadvise(f->fd, 0, 0, POSIX_FADV_DONTNEED);
        }
#endif
    }
    return m;
}

void model_drop_host_copies(model *m) {
    if (!m->host_q) return;
    bool all = !m->fc.data || m->fc.gpu;
    for (int i = 0; i < m->c.n_layer; i++) {
        const layer *l = &m->layers[i];
        const qmat *ls[] = { &l->wq, &l->wk, &l->wv, &l->wo, &l->w_gate, &l->w_up, &l->w_down };
        for (int k = 0; k < 7; k++) if (ls[k]->data && !ls[k]->gpu) all = false;
    }
    if (all) { // contents no longer read; release immediately on memory-limited Linux hosts
#ifdef __linux__
        madvise(m->host_q, m->host_q_len, MADV_DONTNEED);
#else
        madvise(m->host_q, m->host_q_len, MADV_FREE);
#endif
    }
}
