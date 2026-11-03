// Architecture-agnostic decoder description loaded from GGUF or from a Hugging
// Face / MLX safetensors directory (config.json + affine-quantized weights).
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
#include "safetensors.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// MLX affine quantization: w = scale * q + bias per group of `group` columns,
// q packed little-endian at `bits` per value, scales/biases bf16 or f16 [rows][cols/group].
#define QT_AFFINE 1000

typedef struct {
    uint32_t type;       // ggml type or QT_AFFINE
    int rows, cols;      // rows = output features, cols = input features
    size_t row_bytes;
    const uint8_t *data; // NULL if the tensor is absent
    const uint16_t *scales, *biases; // QT_AFFINE only
    int bits, group;                 // QT_AFFINE only
    bool scales_f16;                 // QT_AFFINE: scales/biases are f16 (else bf16)
    void *gpu;                       // backend-owned repacked copy (if any); data is then unused on device
} qmat;

typedef struct {
    const float *attn_norm, *ffn_norm, *post_attn_norm, *post_ffn_norm, *q_norm, *k_norm, *bq, *bk, *bv;
    qmat wq, wk, wv, wg, wo, w_gate, w_up, w_down;
    // Linear layers reuse projection slots: wq=qkv, wk=z, wv=a, wg=b, wo=out.
    bool linear;
    int recurrent_layer;
    const float *conv, *a_log, *dt_bias, *linear_norm;
    bool swa;       // sliding-window layer (KV lives in the SWA pool)
    bool rope;      // apply rotary embedding
    int pool_layer; // index within its KV pool
} layer;

typedef enum { ARCH_LLAMA, ARCH_QWEN2, ARCH_QWEN3, ARCH_GLIMMER, ARCH_DFLASH, ARCH_QWEN35 } arch_t;

typedef struct {
    arch_t arch;
    int n_layer, d, ff, n_head, n_kv, hd, vocab;
    int window; // 0 = no sliding-window layers
    int n_full, n_swa;
    int n_linear, linear_k_heads, linear_v_heads, linear_k_dim, linear_v_dim, conv_kernel, rope_dim;
    size_t recurrent_floats; // per sequence, f32 convolution + delta state

    float rope_base, eps, post_eps, attn_scale, logit_scale, softcap, embed_scale;
    bool rope_neox, embed_norm;
} model_cfg;

typedef struct { const void *base; size_t len; } mem_region;

typedef struct {
    model_cfg c;
    gguf_ctx *g;          // GGUF source, or
    st_ctx *st;           // safetensors source (dir holds config/tokenizer)
    char dir[4096];
    // Host memory the weights and norm vectors live in (for zero-copy GPU wrapping).
    mem_region regions[66];
    int n_regions;
    uint8_t *aux; size_t aux_len, aux_used; // converted f32 norm vectors (safetensors)
    uint8_t *arena; size_t arena_len;       // aligned copies of misaligned safetensors matrices
    layer *layers;
    const float *out_norm;
    qmat tok_embd, output;
    float *inv_freq; // hd/2 entries: base^(-2i/hd) / freq_factor[i]
    // DFlash drafter (ARCH_DFLASH): no embeddings / lm_head of its own (uses the
    // target's); fc projects the concatenated target hidden states.
    qmat fc;
    const float *enc_norm;
    int target_layers[8], n_target, block_size, mask_token;
    void *host_q; size_t host_q_len; // heap copies of weights quantized at load
} model;

// path: a .gguf file, or a directory with config.json + *.safetensors
model *model_load(const char *path);
// Dequantize row r of w into dst (w->cols floats).
void qmat_row(const qmat *w, int r, float *dst);
size_t qmat_bytes(const qmat *w);
// Copy misaligned safetensors matrices into an aligned arena (for backends that
// read the original layout directly); skips matrices a backend repacked.
void model_align_mats(model *m);
void model_free(model *m);
// Dequantize embeddings for n tokens into x (n x d), applying embed scale / norm
// (raw: without them, as the DFlash drafter's inputs).
void model_embed(const model *m, const int32_t *tokens, int n, float *x);
void model_embed_raw(const model *m, const int32_t *tokens, int n, float *x);
// Free heap copies of load-time-quantized weights once a backend repacked them.
void model_drop_host_copies(model *m);
// Weight bytes read on every decode step (everything but the embedding table).
size_t model_streamed_bytes(const model *m);
// Page in and mlock the streamed weights that live in the model's own memory
// (backend-repacked matrices are skipped); returns bytes pinned.
size_t model_pin(const model *m);
const char *arch_name(arch_t a);
