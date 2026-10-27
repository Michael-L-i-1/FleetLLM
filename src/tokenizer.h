// Byte-level BPE tokenizer built from GGUF metadata (tokenizer.ggml.*) or a
// Hugging Face tokenizer.json.
#pragma once
#include "gguf.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum { PRE_LLAMA3, PRE_QWEN2, PRE_LLAMA4 } pretok_kind;

typedef struct tokenizer tokenizer;

tokenizer *tok_load(const gguf_ctx *g);
// From a Hugging Face model directory (tokenizer.json + tokenizer_config.json + config.json).
tokenizer *tok_load_hf(const char *dir);
void tok_free(tokenizer *t);

// Encode UTF-8 text (special tokens such as <|eot_id|> are recognized). Returns
// the token count; *out is malloc'd and owned by the caller.
int tok_encode(const tokenizer *t, const char *text, size_t len, int32_t **out);
// Append the decoded bytes of token id to buf (cap bytes); returns bytes written.
size_t tok_piece(const tokenizer *t, int32_t id, char *buf, size_t cap);
// Decode a token array into a malloc'd NUL-terminated string.
char *tok_decode(const tokenizer *t, const int32_t *ids, int n);

int32_t tok_lookup(const tokenizer *t, const char *s); // -1 if absent
bool tok_is_stop(const tokenizer *t, int32_t id);
int tok_vocab_size(const tokenizer *t);
int32_t tok_bos(const tokenizer *t);
// The model's Jinja chat template as shipped (tokenizer_config.json or GGUF
// tokenizer.chat_template); "" if absent. Used to detect template variants.
const char *tok_chat_template(const tokenizer *t);
