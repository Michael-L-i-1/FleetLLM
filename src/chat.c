#include "chat.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static void put_trimmed(sbuf *b, const char *s) { // Jinja's | trim
    while (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r') s++;
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\n' || s[n - 1] == '\t' || s[n - 1] == '\r')) n--;
    put(b, "%.*s", (int)n, s);
}

chat_family chat_detect(const tokenizer *t) {
    if (tok_lookup(t, "<|start_header_id|>") >= 0) return CHAT_LLAMA3;
    if (tok_lookup(t, "<|im_start|>") >= 0) return CHAT_CHATML;
    if (tok_lookup(t, "<|message|>") >= 0) return CHAT_GLIMMER;
    return CHAT_PLAIN;
}

char *chat_format_opts(const tokenizer *t, const chat_msg *msgs, int n, const chat_opts *o) {
    sbuf b = { malloc(256), 0, 256 };
    b.s[0] = 0;
    switch (chat_detect(t)) {
    case CHAT_LLAMA3: { // Llama 3.1+ templates always open with a dated system header and trim contents
        const char *tpl = tok_chat_template(t);
        bool dated = strstr(tpl, "Cutting Knowledge Date") != NULL;
        put(&b, "<|begin_of_text|>");
        int i = 0;
        if (dated) {
            char today[32] = "26 Jul 2024"; // the 3.1 template's default date_string
            if (strstr(tpl, "strftime_now")) { time_t now = time(NULL); strftime(today, sizeof today, "%d %b %Y", localtime(&now)); }
            put(&b, "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: %s\n\n", today);
            if (n > 0 && !strcmp(msgs[0].role, "system")) put_trimmed(&b, msgs[i++].content);
            put(&b, "<|eot_id|>");
        }
        for (; i < n; i++) {
            put(&b, "<|start_header_id|>%s<|end_header_id|>\n\n", msgs[i].role);
            if (dated) put_trimmed(&b, msgs[i].content);
            else put(&b, "%s", msgs[i].content);
            put(&b, "<|eot_id|>");
        }
        put(&b, "<|start_header_id|>assistant<|end_header_id|>\n\n");
        break;
    }
    case CHAT_CHATML:
        for (int i = 0; i < n; i++) put(&b, "<|im_start|>%s\n%s<|im_end|>\n", msgs[i].role, msgs[i].content);
        put(&b, "<|im_start|>assistant\n");
        if (o->direct && tok_lookup(t, "<think>") >= 0) put(&b, "<think>\n\n</think>\n\n");
        break;
    case CHAT_GLIMMER: { // follows the checkpoint's chat_template.jinja (no tools)
        const char *rs = o->reasoning ? o->reasoning : "high";
        put(&b, "<|begin_of_text|>");
        int i = 0;
        if (n > 0 && !strcmp(msgs[0].role, "system")) put(&b, "<|start|>system<|message|>%s\n\n", msgs[i++].content);
        else {
            put(&b, "<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.");
            if (o->date) put(&b, "\nCurrent date: %s.", o->date);
            put(&b, "\n\n");
        }
        put(&b, "Reasoning strength: %s.\n\n# Valid recipients: \"self\", \"user\".<|eot|>", rs);
        for (; i < n; i++) {
            if (!strcmp(msgs[i].role, "assistant")) put(&b, "<|start|>assistant to=user<|message|>%s<|eot|>", msgs[i].content);
            else put(&b, "<|start|>%s<|message|>%s<|eot|>", msgs[i].role, msgs[i].content);
        }
        put(&b, o->direct ? "<|start|>assistant to=user<|message|>" : "<|start|>assistant");
        break;
    }
    case CHAT_PLAIN:
        for (int i = 0; i < n; i++) put(&b, "%s: %s\n", msgs[i].role, msgs[i].content);
        put(&b, "assistant: ");
        break;
    }
    return b.s;
}

char *chat_format(const tokenizer *t, const chat_msg *msgs, int n, bool direct) {
    chat_opts o = { direct, "low", NULL };
    return chat_format_opts(t, msgs, n, &o);
}
