#include "chat.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *s; size_t n, cap; } sbuf;

static void put(sbuf *b, const char *fmt, ...) {
    va_list ap;
    for (;;) {
        va_start(ap, fmt);
        int k = vsnprintf(b->s + b->n, b->cap - b->n, fmt, ap);
        va_end(ap);
        if ((size_t)k < b->cap - b->n) { b->n += k; return; }
        b->cap = (b->cap + k) * 2;
        b->s = realloc(b->s, b->cap);
    }
}

char *chat_format(const tokenizer *t, const chat_msg *msgs, int n, bool direct) {
    sbuf b = { malloc(256), 0, 256 };
    b.s[0] = 0;
    if (tok_lookup(t, "<|start_header_id|>") >= 0) { // Llama 3
        put(&b, "<|begin_of_text|>");
        for (int i = 0; i < n; i++) put(&b, "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", msgs[i].role, msgs[i].content);
        put(&b, "<|start_header_id|>assistant<|end_header_id|>\n\n");
    } else if (tok_lookup(t, "<|im_start|>") >= 0) { // ChatML (Qwen)
        for (int i = 0; i < n; i++) put(&b, "<|im_start|>%s\n%s<|im_end|>\n", msgs[i].role, msgs[i].content);
        put(&b, "<|im_start|>assistant\n");
        if (direct && tok_lookup(t, "<think>") >= 0) put(&b, "<think>\n\n</think>\n\n");
    } else if (tok_lookup(t, "<|message|>") >= 0) { // Muse Glimmer
        put(&b, "<|begin_of_text|>");
        int i = 0;
        if (n > 0 && !strcmp(msgs[0].role, "system")) put(&b, "<|start|>system<|message|>%s\n\n", msgs[i++].content);
        else put(&b, "<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n\n");
        put(&b, "Reasoning strength: low.\n\n# Valid recipients: \"self\", \"user\".<|eot|>");
        for (; i < n; i++) put(&b, "<|start|>%s<|message|>%s<|eot|>", msgs[i].role, msgs[i].content);
        put(&b, direct ? "<|start|>assistant to=user<|message|>" : "<|start|>assistant");
    } else { // plain
        for (int i = 0; i < n; i++) put(&b, "%s: %s\n", msgs[i].role, msgs[i].content);
        put(&b, "assistant: ");
    }
    return b.s;
}
