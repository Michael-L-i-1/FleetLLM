#include "chat.h"
#include "json.h"
#include <stdarg.h>
#include <math.h>
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

// A JSON string literal (UTF-8 passes through, as with ensure_ascii=False;
// C0 controls escaped). Matches the escaping of json_dump for plain strings.
static void put_json_str(sbuf *b, const char *s, size_t n) {
    put(b, "\"");
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"') put(b, "\\\"");
        else if (c == '\\') put(b, "\\\\");
        else if (c == '\n') put(b, "\\n");
        else if (c == '\r') put(b, "\\r");
        else if (c == '\t') put(b, "\\t");
        else if (c < 0x20) put(b, "\\u%04x", c);
        else put(b, "%c", c);
    }
    put(b, "\"");
}

// Jinja uses Python JSON's shortest round-trip float spelling and keeps the
// distinction between integer tokens and floats such as 1.0. This renderer is
// local to Glimmer; the general JSON serializer's behavior is unchanged.
static void put_template_json(sbuf *b, const json *j, const jnode *n) {
    if (n->type == J_NUM) {
        if (!n->is_float) { put(b, "%.0f", n->num); return; }
        if (!isfinite(n->num)) { put(b, "null"); return; }
        char text[64]; int precision;
        for (precision = 1; precision < 17; precision++) {
            snprintf(text, sizeof text, "%.*g", precision, n->num);
            if (strtod(text, NULL) == n->num) break;
        }
        snprintf(text, sizeof text, "%.*e", precision - 1, n->num);
        int exponent = atoi(strchr(text, 'e') + 1);
        if (exponent < -4 || exponent >= 16) put(b, "%s", text);
        else {
            int decimals = precision - 1 - exponent;
            if (decimals < 1) decimals = 1;
            put(b, "%.*f", decimals, n->num);
        }
    } else if (n->type == J_STR) put_json_str(b, n->s, (size_t)n->n);
    else if (n->type == J_NULL) put(b, "null");
    else if (n->type == J_BOOL) put(b, n->num ? "true" : "false");
    else {
        put(b, n->type == J_OBJ ? "{" : "[");
        for (const jnode *c = json_first(j, n); c; c = json_next(j, c)) {
            if (c != json_first(j, n)) put(b, ", ");
            if (n->type == J_OBJ) { put_json_str(b, c->key, (size_t)c->key_len); put(b, ": "); }
            put_template_json(b, j, c);
        }
        put(b, n->type == J_OBJ ? "}" : "]");
    }
}

static char *template_json(const json *j, const jnode *n) {
    sbuf b = { malloc(64), 0, 64 }; b.s[0] = 0;
    put_template_json(&b, j, n);
    return b.s;
}

chat_family chat_detect(const tokenizer *t) {
    if (tok_lookup(t, "<|start_header_id|>") >= 0) return CHAT_LLAMA3;
    if (tok_lookup(t, "<|im_start|>") >= 0) return CHAT_CHATML;
    if (tok_lookup(t, "<|message|>") >= 0) return CHAT_GLIMMER;
    return CHAT_PLAIN;
}

// ---- Glimmer tool definitions (render_tool_defs / render_system_meta) ----

typedef struct { char *name, *ns, *desc, *params; } gtool; // desc/params: JSON text ("null" when absent)

// Split "namespace.function" (Jinja: fn.name.split('.')[0]); a bare name is
// its own namespace, whose recipient renders as "name.*".
static char *tool_ns(const char *name) {
    const char *dot = strchr(name, '.');
    size_t n = dot ? (size_t)(dot - name) : strlen(name);
    char *ns = malloc(n + 1);
    memcpy(ns, name, n);
    ns[n] = 0;
    return ns;
}

// Parse one OpenAI-format tool definition ({type, function: {name,
// description, parameters}} or the bare {name, ...}). Returns false for
// malformed input (skipped) or a nameless tool.
static bool parse_tool_def(const char *text, gtool *t) {
    memset(t, 0, sizeof *t);
    json *j = json_parse(text, strlen(text));
    if (!j) return false;
    const jnode *root = json_root(j);
    const jnode *fn = json_get(j, root, "function");
    if (!fn || fn->type != J_OBJ) fn = root;
    const char *name = json_str(j, fn, "name");
    bool ok = name && *name;
    if (ok) {
        t->name = strdup(name);
        t->ns = tool_ns(name);
        const jnode *d = json_get(j, fn, "description"), *p = json_get(j, fn, "parameters");
        if (d && d->type == J_STR) { sbuf b = { malloc(32), 0, 32 }; b.s[0] = 0; put_json_str(&b, d->s, (size_t)d->n); t->desc = b.s; }
        else t->desc = strdup("null");
        t->params = (p && (p->type == J_OBJ || p->type == J_ARR)) ? template_json(j, p) : strdup("null");
    }
    json_free(j);
    return ok;
}

static void free_tool_def(gtool *t) { free(t->name); free(t->ns); free(t->desc); free(t->params); }

// The system tool preamble, verbatim from the official chat template.
static void put_tool_defs(sbuf *b, gtool *tools, int n_tools) {
    put(b, "In this environment you have access to a set of tools you can use to answer the user's question.\n\n");
    put(b, "You can invoke a function by writing a \"<atem:function_calls>\" block like the following:\n");
    put(b, "<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n");
    put(b, "String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n");
    put(b, "Here are the functions available in JSONSchema format:\n");
    put(b, "// Tool metadata\n");
    for (int i = 0; i < n_tools; i++) { // namespaces in first-seen order
        bool seen = false;
        for (int k = 0; k < i; k++) if (!strcmp(tools[k].ns, tools[i].ns)) { seen = true; break; }
        if (seen) continue;
        put(b, "{\"name\": ");
        put_json_str(b, tools[i].ns, strlen(tools[i].ns));
        put(b, ", \"description\": \"\"}\n");
    }
    put(b, "// Function schemas");
    for (int i = 0; i < n_tools; i++) {
        put(b, "\n{\"name\": ");
        put_json_str(b, tools[i].name, strlen(tools[i].name));
        put(b, ", \"description\": %s, \"parameters\": %s}", tools[i].desc, tools[i].params);
    }
    put(b, "\n\nHere's an example of how to call a function in the tool set:\n");
    put(b, "(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)\n\n");
    put(b, "to=example_tool_name.example_function_name\n\n");
    put(b, "<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n");
    put(b, "<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n");
    put(b, "<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n");
    put(b, "</atem:invoke>\n</atem:function_calls>");
}

static void put_system_meta(sbuf *b, gtool *tools, int n_tools) {
    put(b, "# Valid recipients: \"self\"");
    for (int i = 0; i < n_tools; i++) {
        bool seen = false;
        for (int k = 0; k < i; k++) if (!strcmp(tools[k].ns, tools[i].ns)) { seen = true; break; }
        if (seen) continue;
        put(b, ", \"%s.*\"", tools[i].ns);
    }
    put(b, ", \"user\".");
}

// One historical or outgoing call as an ATEM block (render_atem). Values are
// typed from the parsed arguments JSON: booleans/null literally, arrays and
// objects as JSON, strings and numbers raw (whitespace and newlines kept).
static void put_atem(sbuf *b, const char *name, const char *args) {
    put(b, "<atem:function_calls>\n<atem:invoke name=\"%s\">\n", name ? name : "");
    json *j = args && *args ? json_parse(args, strlen(args)) : NULL;
    const jnode *root = j ? json_root(j) : NULL;
    if (root && root->type == J_OBJ)
        for (const jnode *m = json_first(j, root); m; m = json_next(j, m)) {
            put(b, "<atem:parameter name=\"%.*s\">", m->key_len, m->key);
            if (m->type == J_BOOL) put(b, m->num ? "true" : "false");
            else if (m->type == J_NULL) put(b, "null");
            else if (m->type == J_ARR || m->type == J_OBJ || m->type == J_NUM) put_template_json(b, j, m);
            else put(b, "%.*s", m->n, m->s); // strings verbatim
            put(b, "</atem:parameter>\n");
        }
    json_free(j);
    put(b, "</atem:invoke>\n</atem:function_calls>");
}

// A role=tool result's tool name: the message's own name, else the name of the
// historical call with the matching id (searched over all messages, as the
// template does), else the id itself, else "".
static const char *tool_msg_name(const chat_msg *msgs, int n, const chat_msg *m) {
    if (m->name && *m->name) return m->name;
    if (m->tool_call_id && *m->tool_call_id) {
        for (int i = 0; i < n; i++)
            for (int k = 0; k < msgs[i].n_calls; k++)
                if (msgs[i].calls[k].id && !strcmp(msgs[i].calls[k].id, m->tool_call_id) &&
                    msgs[i].calls[k].name && *msgs[i].calls[k].name)
                    return msgs[i].calls[k].name;
        return m->tool_call_id;
    }
    return "";
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
    case CHAT_GLIMMER: { // the checkpoint's chat_template.jinja, tools included
        const char *rs = o->reasoning ? o->reasoning : "high";
        gtool *tools = calloc(o->n_tools ? o->n_tools : 1, sizeof(gtool));
        int n_tools = 0;
        for (int k = 0; k < o->n_tools; k++)
            if (parse_tool_def(o->tools[k], &tools[n_tools])) n_tools++;
        sbuf tail = { malloc(256), 0, 256 }; // "\n\nReasoning strength..." through the recipient list
        tail.s[0] = 0;
        put(&tail, "\n\nReasoning strength: %s.", rs);
        if (n_tools) { put(&tail, "\n\n"); sbuf td = { malloc(256), 0, 256 }; td.s[0] = 0; put_tool_defs(&td, tools, n_tools); put(&tail, "%s", td.s); free(td.s); }
        put(&tail, "\n\n");
        put_system_meta(&tail, tools, n_tools);
        put(&b, "<|begin_of_text|>");
        bool has_system = false;
        for (int i = 0; i < n; i++) if (!strcmp(msgs[i].role, "system")) { has_system = true; break; }
        if (!has_system) {
            put(&b, "<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.");
            if (o->date) put(&b, "\nCurrent date: %s.", o->date);
            put(&b, "%s<|eot|>", tail.s);
        }
        for (int i = 0; i < n; i++) {
            const chat_msg *m = &msgs[i];
            if (!strcmp(m->role, "system")) put(&b, "<|start|>system<|message|>%s%s<|eot|>", m->content, tail.s);
            else if (!strcmp(m->role, "user")) put(&b, "<|start|>user<|message|>%s<|eot|>", m->content);
            else if (!strcmp(m->role, "tool")) {
                const char *tname = tool_msg_name(msgs, n, m);
                put(&b, "<|start|>tool %s<|message|><tool_output name=\"%s\">\n%s\n</tool_output><|eot|>", tname, tname, m->content);
            } else if (!strcmp(m->role, "assistant")) {
                if (m->reasoning_content && *m->reasoning_content) put(&b, "<|start|>assistant to=self<|message|>%s<|eom|>", m->reasoning_content);
                if (m->n_calls) {
                    const char *end = (i + 1 < n && !strcmp(msgs[i + 1].role, "assistant")) ? "<|eom|>" : "<|eot|>";
                    for (int k = 0; k < m->n_calls; k++) {
                        const char *cname = m->calls[k].name ? m->calls[k].name : "";
                        put(&b, "<|start|>assistant to=%s<|message|>", cname);
                        put_atem(&b, cname, m->calls[k].arguments);
                        put(&b, "%s", k + 1 < m->n_calls ? "<|eom|>" : end);
                    }
                } else {
                    const char *rcpt = m->recipient && *m->recipient ? m->recipient : "user";
                    bool end_turn = m->end_turn > 0 ? true : m->end_turn < 0 ? false : !strcmp(rcpt, "user");
                    put(&b, "<|start|>assistant to=%s<|message|>%s%s", rcpt, m->content, end_turn ? "<|eot|>" : "<|eom|>");
                }
            } else put(&b, "<|start|>%s<|message|>%s<|eot|>", m->role, m->content);
        }
        put(&b, o->direct ? "<|start|>assistant to=user<|message|>" : "<|start|>assistant");
        for (int k = 0; k < n_tools; k++) free_tool_def(&tools[k]);
        free(tools);
        free(tail.s);
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

// ---- Glimmer ATEM parsing (mirrors the official muse_glimmer tool parser) ----
//
// The reply is segmented into channel messages on <|start|>assistant / to=
// headers and <|eom|> / <|eot|> ends; only bodies addressed to an actual tool
// recipient (not to=self reasoning, not to=user answers, not bare content)
// are scanned for <atem:invoke> blocks. An <atem:invoke> echoed inside
// reasoning or a final answer is never a call, and a block truncated before
// </atem:invoke> (length-capped generation) yields no call.

static bool alcan(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
}

static bool aword(char c) { // regex \w for \b checks
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

// First occurrence of needle in [hay, end), or NULL.
static const char *find_from(const char *hay, const char *needle, const char *end) {
    size_t n = strlen(needle), hlen = (size_t)(end - hay);
    if (!n || n > hlen) return NULL;
    for (const char *p = hay; (size_t)(end - p) >= n; p++)
        if (!memcmp(p, needle, n)) return p;
    return NULL;
}

typedef struct { const char *rcpt; size_t rn, hs, he; } gheader; // header span [hs, he)

// Match a message header ending at the "<|message|>" at m:
// [(<|start|> ws* assistant)? spaces* (to=NAME)?] <|message|>.
static gheader glimmer_header(const char *s, size_t m) {
    gheader h = { NULL, 0, m, m + 11 };
    size_t p = m, k = p;
    while (k > 0 && alcan(s[k - 1])) k--;
    if (k < p && k >= 3 && s[k - 1] == '=' && s[k - 2] == 'o' && s[k - 3] == 't') {
        h.rcpt = s + k;
        h.rn = p - k;
        p = k - 3;
    }
    while (p > 0 && (s[p - 1] == ' ' || s[p - 1] == '\t')) p--;
    if (p >= 9 && !memcmp(s + p - 9, "assistant", 9)) {
        size_t q = p - 9;
        while (q > 0 && (s[q - 1] == ' ' || s[q - 1] == '\t' || s[q - 1] == '\n' || s[q - 1] == '\r')) q--;
        if (q >= 9 && !memcmp(s + q - 9, "<|start|>", 9)) { h.hs = q - 9; return h; }
    }
    h.hs = p;
    return h;
}

// Length of the JSON value starting at s[i] (bounded by n), or 0 when the
// text there is not one complete value. Used to accept a parameter only when
// the whole raw text is valid JSON (like json.loads), keeping anything else
// byte-for-byte as a string.
static size_t jspan_ws(const char *s, size_t n, size_t i) {
    while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    return i;
}

static size_t jspan_str(const char *s, size_t n, size_t i) { // s[i] == '"'
    static const char *ok = "\"\\/bfnrt";
    i++;
    while (i < n) {
        char c = s[i];
        if (c == '"') return i + 1;
        if (c == '\\') {
            i++;
            if (i >= n) return 0;
            if (s[i] == 'u') {
                if (i + 4 >= n) return 0;
                for (int k = 1; k <= 4; k++) {
                    char d = s[i + k];
                    bool hex = (d >= '0' && d <= '9') || (d >= 'a' && d <= 'f') || (d >= 'A' && d <= 'F');
                    if (!hex) return 0;
                }
                i += 5;
            } else {
                bool known = false;
                for (const char *e = ok; *e; e++) if (*e == s[i]) { known = true; break; }
                if (!known) return 0;
                i++;
            }
        } else if ((unsigned char)c < 0x20) return 0;
        else i++;
    }
    return 0;
}

static size_t jspan_val(const char *s, size_t n, size_t i, int depth) {
    if (depth > 100) return 0;
    i = jspan_ws(s, n, i);
    if (i >= n) return 0;
    char c = s[i];
    if (c == '{' || c == '[') {
        char open = c, close = c == '{' ? '}' : ']';
        i = jspan_ws(s, n, i + 1);
        if (i < n && s[i] == close) return i + 1;
        for (;;) {
            if (open == '{') {
                if (i >= n || s[i] != '"') return 0;
                i = jspan_str(s, n, i);
                if (!i) return 0;
                i = jspan_ws(s, n, i);
                if (i >= n || s[i] != ':') return 0;
                i++;
            }
            i = jspan_val(s, n, i, depth + 1);
            if (!i) return 0;
            i = jspan_ws(s, n, i);
            if (i >= n) return 0;
            if (s[i] == ',') { i = jspan_ws(s, n, i + 1); continue; }
            if (s[i] == close) return i + 1;
            return 0;
        }
    }
    if (c == '"') return jspan_str(s, n, i);
    if (c == 't') return n - i >= 4 && !memcmp(s + i, "true", 4) ? i + 4 : 0;
    if (c == 'f') return n - i >= 5 && !memcmp(s + i, "false", 5) ? i + 5 : 0;
    if (c == 'n') return n - i >= 4 && !memcmp(s + i, "null", 4) ? i + 4 : 0;
    if (c == '-' || (c >= '0' && c <= '9') || c == 'N' || c == 'I') {
        size_t j = i;
        if (s[j] == '-') j++;
        if (j < n && s[j] == 'N') return n - j >= 3 && !memcmp(s + j, "NaN", 3) ? j + 3 : 0;
        if (j < n && s[j] == 'I') return n - j >= 8 && !memcmp(s + j, "Infinity", 8) ? j + 8 : 0;
        if (j >= n) return 0;
        if (s[j] == '0') j++;
        else if (s[j] >= '1' && s[j] <= '9') while (j < n && s[j] >= '0' && s[j] <= '9') j++;
        else return 0;
        if (j < n && s[j] == '.') {
            j++;
            if (j >= n || s[j] < '0' || s[j] > '9') return 0;
            while (j < n && s[j] >= '0' && s[j] <= '9') j++;
        }
        if (j < n && (s[j] == 'e' || s[j] == 'E')) {
            size_t k = j + 1;
            if (k < n && (s[k] == '+' || s[k] == '-')) k++;
            if (k >= n || s[k] < '0' || s[k] > '9') return 0;
            while (k < n && s[k] >= '0' && s[k] <= '9') k++;
            j = k;
        }
        return j;
    }
    return 0;
}

// A parameter value as a JSON fragment: canonical JSON when the whole raw
// text parses (typed scalars, arrays, objects), else the raw string escaped
// (whitespace and newlines preserved).
static char *decode_param(const char *raw, size_t n) {
    size_t a = 0, b = n;
    while (a < b && (raw[a] == ' ' || raw[a] == '\t' || raw[a] == '\n' || raw[a] == '\r')) a++;
    while (b > a && (raw[b - 1] == ' ' || raw[b - 1] == '\t' || raw[b - 1] == '\n' || raw[b - 1] == '\r')) b--;
    if (a < b && jspan_val(raw, b, a, 0) == b) {
        json *j = json_parse(raw + a, b - a);
        if (j) {
            char *d = template_json(j, json_root(j));
            json_free(j);
            if (d) return d;
        }
    }
    sbuf o = { malloc(16), 0, 16 };
    o.s[0] = 0;
    put_json_str(&o, raw, n);
    return o.s;
}

static bool valid_tool_name(const char *s, size_t n) {
    if (!n) return false;
    for (size_t i = 0; i < n; i++) if (!alcan(s[i])) return false;
    return true;
}

typedef struct { char *key, *val; } gparam; // val: a JSON fragment

// Written in two literals: the tool-call framing must never contain the
// closing parameter tag as one source token outside a string fragment.
static const char PEND[] = "</atem" ":parameter>";
static const char INVOKE_END[] = "</atem:invoke>";

// Parse the complete <atem:invoke>...</atem:invoke> blocks in the slice
// [base, end) of reply, appending calls (up to max) and consumed spans.
// Returns the call count. Malformed names are rejected; truncated blocks
// (no closing tag) yield nothing.
static int glimmer_scan_calls(const char *reply, const char *base, const char *end,
                              chat_call *calls, int max, int *nc, size_t *sa, size_t *sb, int *nsp, int capsp) {
    int found = 0;
    const char *p = base;
    while (*nc + found < max) {
        const char *s = find_from(p, "<atem:invoke", end);
        if (!s) break;
        if (aword(s[12])) { p = s + 12; continue; } // "<atem:invoked..." is not a block
        const char *gt = find_from(s + 12, ">", end);
        const char *close = find_from(gt ? gt + 1 : s + 12, "</atem:invoke>", end);
        if (!gt || !close || gt > close) {
            // No complete block here; a later one may still follow garbage.
            p = gt ? gt + 1 : s + 12;
            if (!close) break;
            continue;
        }
        // The name lives in the open tag.
        const char *nm = NULL;
        for (const char *q = s; (q = find_from(q, "name=\"", gt)) != NULL;) {
            if (q == s + 1 || !aword(q[-1])) { nm = q + 6; break; }
            q += 6;
        }
        const char *ne = nm ? find_from(nm, "\"", gt) : NULL;
        if (!nm || !ne || !valid_tool_name(nm, (size_t)(ne - nm))) { p = gt + 1; continue; }
        gparam *ps = NULL;
        int nps = 0;
        for (const char *q = gt + 1; (q = find_from(q, "<atem:parameter", close)) != NULL;) {
            if (aword(q[15])) { q += 15; continue; }
            const char *g2 = find_from(q + 15, ">", close);
            if (!g2) { q += 15; continue; }
            const char *kn = NULL;
            for (const char *r = q; (r = find_from(r, "name=\"", g2)) != NULL;) {
                if (r == q || !aword(r[-1])) { kn = r + 6; break; }
                r += 6;
            }
            const char *ke = kn ? find_from(kn, "\"", g2) : NULL;
            const char *c2 = find_from(g2 + 1, PEND, close);
            if (!kn || !ke || ke <= kn || !c2) { q = g2 + 1; continue; }
            char *key = malloc((size_t)(ke - kn) + 1);
            memcpy(key, kn, (size_t)(ke - kn));
            key[ke - kn] = 0;
            char *val = decode_param(g2 + 1, (size_t)(c2 - g2 - 1));
            bool dup = false;
            for (int k = 0; k < nps; k++)
                if (!strcmp(ps[k].key, key)) { free(ps[k].val); ps[k].val = val; dup = true; break; }
            if (dup) free(key);
            else { ps = realloc(ps, sizeof *ps * (size_t)(nps + 1)); ps[nps].key = key; ps[nps].val = val; nps++; }
            q = c2 + sizeof PEND - 1;
        }
        sbuf a = { malloc(32), 0, 32 };
        a.s[0] = 0;
        put(&a, "{");
        for (int k = 0; k < nps; k++) {
            if (k) put(&a, ", ");
            put_json_str(&a, ps[k].key, strlen(ps[k].key));
            put(&a, ": %s", ps[k].val);
            free(ps[k].key);
            free(ps[k].val);
        }
        free(ps);
        put(&a, "}");
        size_t nn = (size_t)(ne - nm); // preserve names until registered-tool binding
        chat_call *c = &calls[(*nc) + found++];
        char *cn = malloc(nn + 1);
        memcpy(cn, nm, nn);
        cn[nn] = 0;
        c->name = cn;
        c->arguments = a.s;
        c->id = NULL;
        if (*nsp < capsp) { sa[*nsp] = (size_t)(s - reply); sb[*nsp] = (size_t)(close - reply) + sizeof INVOKE_END - 1; (*nsp)++; }
        p = close + sizeof INVOKE_END - 1;
    }
    return found;
}

static bool rcpt_is(const gheader *h, const char *name) {
    size_t n = strlen(name);
    return h->rcpt && h->rn == n && !memcmp(h->rcpt, name, n);
}

int chat_parse_calls_tools(chat_family f, const char *reply, chat_call *calls, int max, char **rest,
                           const char *const *names, int n_names) {
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
    } else if (f == CHAT_GLIMMER) {
        size_t len = strlen(reply);
        size_t *ba = NULL, *bb = NULL; // tool-channel body slices of reply
        int nb = 0, capb = 0;
        size_t pos = 0;
        while (pos < len) {
            const char *m = find_from(reply + pos, "<|message|>", reply + len);
            if (!m) break;
            gheader h = glimmer_header(reply, (size_t)(m - reply));
            size_t he = h.he;
            const char *e1 = find_from(reply + he, "<|eom|>", reply + len);
            const char *e2 = find_from(reply + he, "<|eot|>", reply + len);
            const char *e = e1 && e2 ? (e1 < e2 ? e1 : e2) : e1 ? e1 : e2;
            const char *m2 = find_from(reply + he, "<|message|>", reply + len);
            size_t nhs = len;
            if (m2) {
                gheader h2 = glimmer_header(reply, (size_t)(m2 - reply));
                nhs = h2.hs < he ? he : h2.hs;
            }
            size_t be = e ? (size_t)(e - reply) : len;
            size_t next_pos;
            if (m2 && nhs < be) { be = nhs; next_pos = nhs; }
            else next_pos = e ? (size_t)(e - reply) + 6 : len;
            const char *st = find_from(reply + he, "<|start|>", reply + be);
            if (st) be = (size_t)(st - reply);
            if (!rcpt_is(&h, "self") && !rcpt_is(&h, "user") && h.rcpt && be > he) {
                if (nb == capb) {
                    capb = capb ? capb * 2 : 8;
                    ba = realloc(ba, sizeof *ba * (size_t)capb);
                    bb = realloc(bb, sizeof *bb * (size_t)capb);
                }
                ba[nb] = he;
                bb[nb] = be;
                nb++;
            }
            pos = next_pos <= pos ? pos + 1 : next_pos;
        }
        size_t *sa = malloc(sizeof *sa * (size_t)(max > 0 ? max : 1));
        size_t *sb = malloc(sizeof *sb * (size_t)(max > 0 ? max : 1));
        int nsp = 0;
        if (nb) {
            for (int i = 0; i < nb && nc < max; i++)
                nc += glimmer_scan_calls(reply, reply + ba[i], reply + bb[i], calls, max, &nc, sa, sb, &nsp, max);
        }
        size_t at = 0;
        for (int i = 0; i < nsp; i++) {
            put(&out, "%.*s", (int)(sa[i] - at), reply + at);
            at = sb[i];
        }
        put(&out, "%s", reply + at);
        free(sa);
        free(sb);
        free(ba);
        free(bb);
    } else put(&out, "%s", reply);
    if (f == CHAT_GLIMMER) for (int i = 0; i < nc; i++) {
        const char *name = calls[i].name, *dot = strchr(name, '.');
        bool exact = false;
        for (int k = 0; k < n_names; k++) if (!strcmp(name, names[k])) exact = true;
        if (exact || !dot || strlen(dot + 1) != (size_t)(dot - name) ||
            memcmp(name, dot + 1, (size_t)(dot - name))) continue;
        for (int k = 0; k < n_names; k++) if (!strcmp(dot + 1, names[k])) {
            char *bound = strdup(names[k]);
            free((char *)calls[i].name);
            calls[i].name = bound;
            break;
        }
    }
    *rest = out.s;
    return nc;
}

int chat_parse_calls(chat_family f, const char *reply, chat_call *calls, int max, char **rest) {
    return chat_parse_calls_tools(f, reply, calls, max, rest, NULL, 0);
}
