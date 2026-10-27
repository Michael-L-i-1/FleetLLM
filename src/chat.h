// Chat formatting for the supported template families, detected from the
// vocabulary: Llama 3 headers, ChatML (Qwen; optional no-think block) and Muse
// Glimmer channels. Templates are built in rather than interpreted (no Jinja).
#pragma once
#include "tokenizer.h"
#include <stdbool.h>

typedef struct { const char *role, *content; } chat_msg;

// Format messages plus the assistant generation prompt. `direct` asks for an
// answer without a reasoning channel (Qwen3 no-think, Glimmer to=user).
// Returns a malloc'd string.
char *chat_format(const tokenizer *t, const chat_msg *msgs, int n, bool direct);
