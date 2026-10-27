// Chat formatting for the supported template families, detected from the
// vocabulary: Llama 3 headers, ChatML (Qwen; optional no-think block) and Muse
// Glimmer channels. Templates are built in rather than interpreted (no Jinja).
#pragma once
#include "tokenizer.h"
#include <stdbool.h>

typedef struct { const char *role, *content; } chat_msg;

typedef struct {
    bool direct;           // answer without a reasoning channel (Qwen3 no-think, Glimmer to=user)
    const char *reasoning; // Glimmer reasoning strength ("low" / "medium" / "high"); NULL: template default
    const char *date;      // "YYYY-MM-DD" for templates that state the current date; NULL: omit
} chat_opts;

typedef enum { CHAT_LLAMA3, CHAT_CHATML, CHAT_GLIMMER, CHAT_PLAIN } chat_family;
chat_family chat_detect(const tokenizer *t);

// Format messages plus the assistant generation prompt. Returns a malloc'd string.
char *chat_format_opts(const tokenizer *t, const chat_msg *msgs, int n, const chat_opts *o);
// Shorthand used by the CLI tools (Glimmer: low reasoning strength).
char *chat_format(const tokenizer *t, const chat_msg *msgs, int n, bool direct);
