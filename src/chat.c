#include "chat.h"
#include "json.h"
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
        bool dated = strstr(tpl, "Cutting Knowledge Date") != NULL, tools_done = !o->n_tools;
        put(&b, "<|begin_of_text|>");
        int i = 0;
        if (dated) {
            char today[32] = "26 Jul 2024"; // the 3.1 template's default date_string
            if (strstr(tpl, "strftime_now")) { time_t now = time(NULL); strftime(today, sizeof today, "%d %b %Y", localtime(&now)); }
            put(&b, "<|start_header_id|>system<|end_header_id|>\n\n%sCutting Knowledge Date: December 2023\nToday Date: %s\n\n", o->n_tools ? "Environment: ipython\n" : "", today);
            if (n > 0 && !strcmp(msgs[0].role, "system")) put_trimmed(&b, msgs[i++].content);
            put(&b, "<|eot_id|>");
        }
        for (; i < n; i++) {
            const chat_msg *m = &msgs[i];
            put(&b, "<|start_header_id|>%s<|end_header_id|>\n\n", !strcmp(m->role, "tool") ? "ipython" : m->role);
            if (!tools_done && !strcmp(m->role, "user")) { // the template's tools_in_user_message
                put(&b, "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n"
                        "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n");
                for (int k = 0; k < o->n_tools; k++) put(&b, "%s\n\n", o->tools[k]);
                tools_done = true;
            }
            if (m->n_calls) // the template renders one call per assistant turn
                put(&b, "{\"name\": \"%s\", \"parameters\": %s}", m->calls[0].name, m->calls[0].arguments && *m->calls[0].arguments ? m->calls[0].arguments : "{}");
            else if (dated) put_trimmed(&b, m->content);
            else put(&b, "%s", m->content);
            put(&b, "<|eot_id|>");
        }
        put(&b, "<|start_header_id|>assistant<|end_header_id|>\n\n");
        break;
    }
    case CHAT_CHATML: {
        int i = 0;
        if (o->n_tools) { // Qwen 2.5 / 3 tool section in the system message
            put(&b, "<|im_start|>system\n%s", n > 0 && !strcmp(msgs[0].role, "system") ? msgs[i++].content : "");
            put(&b, "%s# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
                i ? "\n\n" : "");
            for (int k = 0; k < o->n_tools; k++) put(&b, "\n%s", o->tools[k]);
            put(&b, "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                    "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
        }
        for (; i < n; i++) {
            const chat_msg *m = &msgs[i];
            if (!strcmp(m->role, "tool")) { // consecutive tool results share one user turn
                if (i == 0 || strcmp(msgs[i - 1].role, "tool")) put(&b, "<|im_start|>user");
                put(&b, "\n<tool_response>\n%s\n</tool_response>", m->content);
                if (i + 1 == n || strcmp(msgs[i + 1].role, "tool")) put(&b, "<|im_end|>\n");
                continue;
            }
            put(&b, "<|im_start|>%s\n", m->role);
            if (tok_qwen35(t))
                put_trimmed(&b, m->content);
            else
                put(&b, "%s", m->content);
            for (int k = 0; k < m->n_calls; k++)
                put(&b, "%s<tool_call>\n{\"name\": \"%s\", \"arguments\": %s}\n</tool_call>", (*m->content || k) ? "\n" : "", m->calls[k].name,
                    m->calls[k].arguments && *m->calls[k].arguments ? m->calls[k].arguments : "{}");
            put(&b, "<|im_end|>\n");
        }
        put(&b, "<|im_start|>assistant\n");
        if (o->direct && tok_lookup(t, "<think>") >= 0) put(&b, "<think>\n\n</think>\n\n");
        else if (tok_qwen35(t))
            put(&b, "<think>\n");
        break;
    }
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
    chat_opts o = { .direct = direct, .reasoning = "low" };
    return chat_format_opts(t, msgs, n, &o);
}

// A JSON object with a "name" string and an object of arguments under args_key
// (or "arguments"); fills *c on success.
static bool parse_call(const char *p, size_t n, const char *args_key, chat_call *c) {
    json *j = json_parse(p, n);
    if (!j) return false;
    const jnode *r = json_root(j), *a = json_get(j, r, args_key);
    if (!a) a = json_get(j, r, "arguments");
    const char *name = json_str(j, r, "name");
    bool ok = r->type == J_OBJ && name && (!a || a->type == J_OBJ || a->type == J_STR);
    if (ok) {
        c->name = strdup(name);
        c->arguments = !a ? strdup("{}") : a->type == J_STR ? strdup(a->s) : json_dump(j, a);
    }
    json_free(j);
    return ok;
}

int chat_parse_calls(chat_family f, const char *reply, chat_call *calls, int max, char **rest) {
    sbuf out = { malloc(64), 0, 64 };
    out.s[0] = 0;
    int nc = 0;
    if (f == CHAT_CHATML) {
        const char *p = reply;
        for (const char *q; (q = strstr(p, "<tool_call>")) != NULL;) {
            const char *e = strstr(q, "</tool_call>");
            if (!e) break;
            put(&out, "%.*s", (int)(q - p), p);
            const char *a = q + strlen("<tool_call>");
            if (nc < max && parse_call(a, (size_t)(e - a), "arguments", &calls[nc])) nc++;
            else put(&out, "%.*s", (int)(e + strlen("</tool_call>") - q), q);
            p = e + strlen("</tool_call>");
        }
        put(&out, "%s", p);
    } else if (f == CHAT_LLAMA3) {
        const char *p = reply;
        while (*p == ' ' || *p == '\n') p++;
        if (!strncmp(p, "<|python_tag|>", 14)) p += 14;
        size_t n = strlen(p);
        while (n && (p[n - 1] == ' ' || p[n - 1] == '\n')) n--;
        if (n && p[0] == '{' && p[n - 1] == '}' && max > 0 && parse_call(p, n, "parameters", &calls[0])) nc = 1;
        else put(&out, "%s", reply);
    } else put(&out, "%s", reply);
    *rest = out.s;
    return nc;
}
