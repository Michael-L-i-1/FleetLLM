// Architecture-agnostic decoder description loaded from GGUF.
//
// One pre-norm transformer covers several families via flags:
//   x = embed(tok) [* embed_scale] [rms(x) if embed_norm]
//   per layer: h = rms(x)*attn_norm; q,k,v (+bias) = W h; [g = Wg h]
//              [q,k = rms(.)*qk_norm]; rope (NORM or NEOX) on layers that use it
//              a = Wo(sdpa(q,k,v) [* sigmoid(g)]); x += [rms(a)*post_attn_norm | a]
//              h = rms(x)*ffn_norm; f = Wdown(silu(Wgate h) * Wup h); x += [rms(f)*post_ffw_norm | f]
//   logits = softcap(logit_scale * Wout rms(x)*out_norm)
// muse-glimmer: embed_norm, qk_norm, attention gate, post norms, SWA + NoPE full layers, softcap
// llama: optional rope_freqs; qwen2: NEOX rope + q/k/v bias; qwen3: NEOX rope + qk_norm
#pragma once
#include "gguf.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t type;
    int rows, cols;      // rows = output features, cols = input features
    size_t row_bytes;
    const uint8_t *data; // NULL if the tensor is absent
} qmat;

typedef struct {
    const float *attn_norm, *ffn_norm, *post_attn_norm, *post_ffn_norm, *q_norm, *k_norm, *bq, *bk, *bv;
    qmat wq, wk, wv, wg, wo, w_gate, w_up, w_down;
    bool swa;       // sliding-window layer (KV lives in the SWA pool)
    bool rope;      // apply rotary embedding
    int pool_layer; // index within its KV pool
} layer;

typedef enum { ARCH_LLAMA, ARCH_QWEN2, ARCH_QWEN3, ARCH_GLIMMER } arch_t;

typedef struct {
    arch_t arch;
    int n_layer, d, ff, n_head, n_kv, hd, vocab;
    int window; // 0 = no sliding-window layers
    int n_full, n_swa;
    float rope_base, eps, post_eps, attn_scale, logit_scale, softcap, embed_scale;
    bool rope_neox, embed_norm;
} model_cfg;

typedef struct {
    model_cfg c;
    gguf_ctx *g;
    layer *layers;
    const float *out_norm;
    qmat tok_embd, output;
    float *inv_freq; // hd/2 entries: base^(-2i/hd) / freq_factor[i]
} model;

model *model_load(const char *path);
void model_free(model *m);
// Dequantize embeddings for n tokens into x (n x d), applying embed scale / norm.
void model_embed(const model *m, const int32_t *tokens, int n, float *x);
// Weight bytes read on every decode step (everything but the embedding table).
size_t model_streamed_bytes(const model *m);
// Page in and mlock the streamed weights; returns bytes pinned.
size_t model_pin(const model *m);
const char *arch_name(arch_t a);
