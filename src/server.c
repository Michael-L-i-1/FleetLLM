#include "server.h"
#include "chat.h"
#include "json.h"
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

// ---------------------------------------------------------------- byte buffer

typedef struct { char *s; size_t n, cap; } buf;

static void bput(buf *b, const char *p, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2;
        b->s = realloc(b->s, b->cap);
    }
    memcpy(b->s + b->n, p, n);
    b->n += n;
    b->s[b->n] = 0;
}
static void bstr(buf *b, const char *s) { bput(b, s, strlen(s)); }
static void bfmt(buf *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    bput(b, tmp, k < (int)sizeof tmp ? (size_t)k : sizeof tmp - 1);
}
// JSON string literal (UTF-8 passes through; control characters escaped)
static void bjson(buf *b, const char *s, size_t n) {
    bput(b, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"') bput(b, "\\\"", 2);
        else if (c == '\\') bput(b, "\\\\", 2);
        else if (c == '\n') bput(b, "\\n", 2);
        else if (c == '\r') bput(b, "\\r", 2);
        else if (c == '\t') bput(b, "\\t", 2);
        else if (c < 0x20) bfmt(b, "\\u%04x", c);
        else bput(b, (const char *)&c, 1);
    }
    bput(b, "\"", 1);
}

// ---------------------------------------------------------------- requests

enum { CH_CONTENT, CH_REASONING };
enum { PS_HEADER, PS_MESSAGE, PS_EXPECT_START }; // Glimmer channel parser

typedef struct req {
    int seq; // engine sequence id
    int32_t *prompt; int n_prompt;
    sample_params p;
    char *stop[4]; int n_stop_str;
    // output, guarded by mu
    pthread_mutex_t mu;
    pthread_cond_t cv;
    buf text[2];      // content, reasoning (complete UTF-8 only)
    buf raw[2];       // bytes not yet known to be complete UTF-8
    int n_gen;        // generated tokens
    int n_cached;     // prompt tokens whose KV was reused (prefix sharing)
    bool done, cancelled, started;
    bool tools;       // tools were offered: the reply is parsed for tool calls (and not streamed piecemeal)
    const char *finish;
    double t_submit, t_first, t_done;
    // parser (engine thread only)
    int pstate, channel, seen;
    char header[96]; int header_len;
    struct req *next;
} req;

typedef struct {
    engine *e;
    const tokenizer *tok;
    chat_family fam;
    const server_opts *o;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    req *queue;      // submitted, not yet in the engine
    req **active; int n_active, cap_active;
    long served, gen_tokens, prompt_tokens;
    double t_start;
    int32_t t_start_id, t_message, t_eom, t_eot, t_think, t_end_think, t_eos;
} server;

static double wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Append bytes to channel ch, moving complete UTF-8 sequences to text[ch].
static void out_bytes(req *r, int ch, const char *p, size_t n) {
    buf *raw = &r->raw[ch];
    bput(raw, p, n);
    size_t ok = raw->n;
    // hold back a trailing incomplete sequence
    for (size_t k = 1; k <= 3 && k <= raw->n; k++) {
        unsigned char c = (unsigned char)raw->s[raw->n - k];
        if ((c & 0xC0) == 0x80) continue;
        size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        if (need > k) ok = raw->n - k;
        break;
    }
    bput(&r->text[ch], raw->s, ok);
    memmove(raw->s, raw->s + ok, raw->n - ok);
    raw->n -= ok;
}

// Route one generated token into content / reasoning (engine thread, r->mu held).
static void route_token(server *sv, req *r, int32_t t) {
    char piece[256];
    if (sv->fam == CHAT_GLIMMER) {
        if (t == sv->t_eot || t == sv->t_eos) return;
        switch (r->pstate) {
        case PS_EXPECT_START:
            if (t == sv->t_start_id) { r->pstate = PS_HEADER; r->header_len = 0; return; }
            r->pstate = PS_MESSAGE; // unexpected: treat as text of the current channel
            break;
        case PS_HEADER:
            if (t == sv->t_message) {
                r->header[r->header_len] = 0;
                const char *to = strstr(r->header, "to=");
                r->channel = to && !strncmp(to + 3, "self", 4) ? CH_REASONING : CH_CONTENT;
                r->pstate = PS_MESSAGE;
            } else {
                size_t k = tok_piece(sv->tok, t, piece, sizeof piece);
                if (r->header_len + k < sizeof r->header - 1) { memcpy(r->header + r->header_len, piece, k); r->header_len += (int)k; }
            }
            return;
        }
        if (t == sv->t_eom) { r->pstate = PS_EXPECT_START; return; }
        if (t == sv->t_start_id || t == sv->t_message) return;
    } else if (t == sv->t_think) { r->channel = CH_REASONING; return; }
    else if (t == sv->t_end_think) { r->channel = CH_CONTENT; return; }
    if (tok_is_stop(sv->tok, t)) return;
    size_t k = tok_piece(sv->tok, t, piece, sizeof piece);
    out_bytes(r, r->channel, piece, k);
}

// Stop strings apply to the answer text; returns true (and truncates) on a hit.
static bool hit_stop_string(req *r) {
    buf *c = &r->text[CH_CONTENT];
    for (int i = 0; i < r->n_stop_str; i++) {
        char *at = strstr(c->s ? c->s : "", r->stop[i]);
        if (at) { c->n = (size_t)(at - c->s); c->s[c->n] = 0; return true; }
    }
    return false;
}

// ---------------------------------------------------------------- engine thread

static void *engine_loop(void *arg) {
    server *sv = arg;
    engine *e = sv->e;
    int *ids = NULL, cap_ids = 0;
    double t_log = now_secs();
    long gen_log = 0;
    for (;;) {
        pthread_mutex_lock(&sv->mu);
        while (!sv->n_active && !sv->queue) pthread_cond_wait(&sv->cv, &sv->mu);
        // admit queued requests (prefix sharing: a new agent forks the longest matching KV)
        while (sv->queue && (sv->o->max_active <= 0 || sv->n_active < sv->o->max_active)) {
            req *r = sv->queue;
            if (sv->n_active && sv->o->admit) { // admit only what fits beside the running requests
                if (sv->n_active > cap_ids) { cap_ids = sv->n_active * 2; ids = realloc(ids, sizeof(int) * cap_ids); }
                for (int i = 0; i < sv->n_active; i++) ids[i] = sv->active[i]->seq;
                if (engine_need_pages(e, r->prompt, r->n_prompt, r->p.max_new) + engine_committed_pages(e, ids, sv->n_active) > engine_avail_pages(e)) break;
            }
            sv->queue = r->next;
            if (sv->n_active == sv->cap_active) { sv->cap_active = sv->cap_active ? sv->cap_active * 2 : 64; sv->active = realloc(sv->active, sizeof(req *) * sv->cap_active); }
            r->seq = engine_create(e, r->prompt, r->n_prompt);
            r->n_cached = e->seqs[r->seq].n_tokens + e->seqs[r->seq].wait_len;
            engine_start(e, r->seq, r->p);
            r->started = true;
            sv->active[sv->n_active++] = r;
            sv->prompt_tokens += r->n_prompt;
        }
        pthread_mutex_unlock(&sv->mu);
        if (sv->n_active > cap_ids) { cap_ids = sv->n_active * 2; ids = realloc(ids, sizeof(int) * cap_ids); }
        int n = 0;
        for (int i = 0; i < sv->n_active; i++) {
            req *r = sv->active[i];
            if (!r->cancelled && !e->seqs[r->seq].done) ids[n++] = r->seq;
        }
        if (n) engine_step(e, ids, n);
        // deliver new tokens; finish completed / cancelled requests
        int k = 0;
        for (int i = 0; i < sv->n_active; i++) {
            req *r = sv->active[i];
            seq_t *s = &e->seqs[r->seq];
            pthread_mutex_lock(&r->mu);
            bool stopped = false;
            for (; r->seen < s->n_gen; r->seen++) {
                if (!r->n_gen) r->t_first = wall();
                r->n_gen++;
                sv->gen_tokens++;
                route_token(sv, r, s->gen[r->seen]);
                if (r->n_stop_str && hit_stop_string(r)) { stopped = true; break; }
            }
            bool fin = r->cancelled || stopped || s->done;
            if (fin) {
                r->finish = stopped || (s->done && s->n_gen < s->p.max_new) ? "stop" : "length";
                r->done = true;
                r->t_done = wall();
                engine_retire(e, r->seq); // keep the KV for prefix reuse (multi-turn agents) until memory is needed
                sv->served++;
            }
            pthread_cond_broadcast(&r->cv);
            pthread_mutex_unlock(&r->mu);
            if (!fin) sv->active[k++] = r;
        }
        sv->n_active = k;
        double t = now_secs();
        if (t - t_log > 10) {
            fprintf(stderr, "[serve] %d active | %.1f tok/s generated | %ld served | %ld preemptions | %ld shared prompt tokens | swap %ld out %ld in (%.2fs, %.2f GB held)"
                            " | %ld tokens reused in %ld spans (%.2fs) | drafts %ld/%ld accepted in %ld verifies\n",
                    sv->n_active, (sv->gen_tokens - gen_log) / (t - t_log), sv->served, e->st.preemptions, e->st.shared_tokens, e->st.swap_outs, e->st.swap_ins,
                    e->st.swap_secs, e->swap_used / 1e9, e->st.reuse_tokens, e->st.reuse_moves, e->st.reuse_secs, e->st.spec_accepted,
                    e->st.spec_drafted, e->st.spec_steps);
            t_log = t;
            gen_log = sv->gen_tokens;
        }
    }
    return NULL;
}

// ---------------------------------------------------------------- HTTP

typedef struct { int fd; server *sv; } conn;

static bool send_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t k = send(fd, p, n, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return false; }
        p += k;
        n -= (size_t)k;
    }
    return true;
}

static bool respond(int fd, int code, const char *type, const char *body, size_t n) {
    char h[256];
    const char *msg = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : "Error";
    int k = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: keep-alive\r\n\r\n", code, msg, type, n);
    return send_all(fd, h, (size_t)k) && send_all(fd, body, n);
}

static bool respond_error(int fd, int code, const char *message) {
    buf b = { 0 };
    bstr(&b, "{\"error\":{\"message\":");
    bjson(&b, message, strlen(message));
    bstr(&b, ",\"type\":\"invalid_request_error\"}}");
    bool ok = respond(fd, code, "application/json", b.s, b.n);
    free(b.s);
    return ok;
}

static bool chunk(int fd, const char *p, size_t n) { // HTTP/1.1 chunked transfer
    char h[32];
    int k = snprintf(h, sizeof h, "%zx\r\n", n);
    return send_all(fd, h, (size_t)k) && send_all(fd, p, n) && send_all(fd, "\r\n", 2);
}

static void sse(int fd, bool *ok, buf *ev) {
    buf b = { 0 };
    bstr(&b, "data: ");
    bput(&b, ev->s, ev->n);
    bstr(&b, "\n\n");
    if (*ok) *ok = chunk(fd, b.s, b.n);
    free(b.s);
    ev->n = 0;
}

// Text of a message's content: a string, or an array of {type: "text", text}.
static void content_text(const json *j, const jnode *c, buf *out) {
    if (!c) return;
    if (c->type == J_STR) { bput(out, c->s, c->n); return; }
    for (const jnode *p = c->type == J_ARR ? json_first(j, c) : NULL; p; p = json_next(j, p)) {
        const char *ty = json_str(j, p, "type"), *tx = json_str(j, p, "text");
        if (tx && (!ty || !strcmp(ty, "text"))) bstr(out, tx);
    }
}

// The fields of a finished chat reply's message (or delta): content, reasoning
// and, when tools were offered, the tool calls parsed from it. Returns the
// finish reason.
static const char *reply_fields(server *sv, req *r, buf *b) {
    static long next_call;
    const char *content = r->text[0].s ? r->text[0].s : "", *finish = r->finish;
    chat_call calls[8];
    char *rest = NULL;
    int nc = r->tools ? chat_parse_calls(sv->fam, content, calls, 8, &rest) : 0;
    if (nc) {
        const char *p = rest;
        while (*p == ' ' || *p == '\n') p++;
        if (*p) { bstr(b, "\"content\":"); bjson(b, rest, strlen(rest)); }
        else bstr(b, "\"content\":null");
        bstr(b, ",\"tool_calls\":[");
        for (int i = 0; i < nc; i++) {
            bfmt(b, "%s{\"index\":%d,\"id\":\"call_%ld\",\"type\":\"function\",\"function\":{\"name\":", i ? "," : "", i, __sync_add_and_fetch(&next_call, 1));
            bjson(b, calls[i].name, strlen(calls[i].name));
            bstr(b, ",\"arguments\":");
            bjson(b, calls[i].arguments, strlen(calls[i].arguments));
            bstr(b, "}}");
            free((char *)calls[i].name);
            free((char *)calls[i].arguments);
        }
        bstr(b, "]");
        finish = "tool_calls";
    } else {
        bstr(b, "\"content\":");
        bjson(b, content, r->text[0].n);
    }
    free(rest);
    if (r->text[1].n) { bstr(b, ",\"reasoning_content\":"); bjson(b, r->text[1].s, r->text[1].n); }
    return finish;
}

static void free_req(req *r) {
    free(r->prompt);
    for (int i = 0; i < 2; i++) { free(r->text[i].s); free(r->raw[i].s); }
    for (int i = 0; i < r->n_stop_str; i++) free(r->stop[i]);
    pthread_mutex_destroy(&r->mu);
    pthread_cond_destroy(&r->cv);
    free(r);
}

static void handle_completion(conn *c, const char *body, size_t len, bool chat) {
    server *sv = c->sv;
    json *j = json_parse(body, len);
    if (!j) { respond_error(c->fd, 400, "invalid JSON"); return; }
    const jnode *root = json_root(j);
    req *r = calloc(1, sizeof *r);
    pthread_mutex_init(&r->mu, NULL);
    pthread_cond_init(&r->cv, NULL);
    // prompt
    char *text = NULL;
    if (chat) {
        const jnode *msgs = json_get(j, root, "messages");
        if (!msgs || msgs->type != J_ARR || !msgs->n) { respond_error(c->fd, 400, "messages must be a non-empty array"); json_free(j); free_req(r); return; }
        chat_msg *cm = calloc(msgs->n, sizeof(chat_msg));
        buf *bodies = calloc(msgs->n, sizeof(buf));
        int k = 0, n_calls = 0;
        for (const jnode *m = json_first(j, msgs); m; m = json_next(j, m)) {
            const jnode *tc = json_get(j, m, "tool_calls");
            n_calls += tc && tc->type == J_ARR ? tc->n : 0;
        }
        chat_call *calls = calloc(n_calls ? n_calls : 1, sizeof(chat_call));
        n_calls = 0;
        for (const jnode *m = json_first(j, msgs); m; m = json_next(j, m), k++) {
            content_text(j, json_get(j, m, "content"), &bodies[k]);
            const char *role = json_str(j, m, "role");
            cm[k].role = role ? role : "user";
            cm[k].content = bodies[k].s ? bodies[k].s : "";
            const jnode *tc = json_get(j, m, "tool_calls"); // [{id, type, function: {name, arguments}}]
            cm[k].calls = calls + n_calls;
            for (const jnode *c = tc && tc->type == J_ARR ? json_first(j, tc) : NULL; c; c = json_next(j, c)) {
                const jnode *fn = json_get(j, c, "function");
                if (!json_str(j, fn, "name")) continue;
                calls[n_calls++] = (chat_call){ json_str(j, fn, "name"), json_str(j, fn, "arguments") };
                cm[k].n_calls++;
            }
        }
        // tool definitions, each as JSON text (tool_choice "none": not offered)
        const jnode *tl = json_get(j, root, "tools");
        const char *choice = json_str(j, root, "tool_choice");
        int n_tools = tl && tl->type == J_ARR && !(choice && !strcmp(choice, "none")) ? tl->n : 0;
        char **tools = calloc(n_tools ? n_tools : 1, sizeof(char *));
        n_tools = 0;
        for (const jnode *t = tl && tl->type == J_ARR && !(choice && !strcmp(choice, "none")) ? json_first(j, tl) : NULL; t; t = json_next(j, t)) tools[n_tools++] = json_dump(j, t);
        r->tools = n_tools > 0;
        time_t now = time(NULL);
        char date[16];
        strftime(date, sizeof date, "%Y-%m-%d", localtime(&now));
        const char *effort = json_str(j, root, "reasoning_effort");
        chat_opts o = { false, effort ? effort : sv->o->reasoning, date, (const char **)tools, n_tools };
        text = chat_format_opts(sv->tok, cm, k, &o);
        for (int i = 0; i < k; i++) free(bodies[i].s);
        for (int i = 0; i < n_tools; i++) free(tools[i]);
        free(tools);
        free(calls);
        free(bodies);
        free(cm);
    } else {
        const char *p = json_str(j, root, "prompt");
        if (!p) { respond_error(c->fd, 400, "prompt must be a string"); json_free(j); free_req(r); return; }
        text = strdup(p);
    }
    r->n_prompt = tok_encode(sv->tok, text, strlen(text), &r->prompt);
    free(text);
    // sampling
    r->p = sample_defaults();
    r->p.temperature = (float)json_num(j, root, "temperature", 1.0);
    r->p.top_p = (float)json_num(j, root, "top_p", 1.0);
    r->p.top_k = 0;
    r->p.seed = (uint64_t)json_num(j, root, "seed", 0);
    r->p.max_new = (int)json_num(j, root, "max_completion_tokens", json_num(j, root, "max_tokens", sv->o->default_max_tokens));
    if (r->p.max_new <= 0) r->p.max_new = sv->o->default_max_tokens;
    if (json_num(j, root, "n", 1) != 1) { respond_error(c->fd, 400, "only n=1 is supported"); json_free(j); free_req(r); return; }
    if (chat && sv->fam == CHAT_GLIMMER) { // reasoning ends a message (<|eom|>), not the turn
        r->p.stop[r->p.n_stop++] = sv->t_eot;
        if (sv->t_eos >= 0) r->p.stop[r->p.n_stop++] = sv->t_eos;
        r->pstate = PS_HEADER;
    }
    const jnode *st = json_get(j, root, "stop");
    if (st && st->type == J_STR && st->n) r->stop[r->n_stop_str++] = strdup(st->s);
    for (const jnode *s = st && st->type == J_ARR ? json_first(j, st) : NULL; s && r->n_stop_str < 4; s = json_next(j, s))
        if (s->type == J_STR && s->n) r->stop[r->n_stop_str++] = strdup(s->s);
    bool stream = json_bool(j, root, "stream", false);
    bool usage = json_bool(j, json_get(j, root, "stream_options"), "include_usage", false);
    json_free(j);
    r->t_submit = wall();
    pthread_mutex_lock(&sv->mu);
    req **tail = &sv->queue;
    while (*tail) tail = &(*tail)->next;
    *tail = r;
    pthread_cond_signal(&sv->cv);
    pthread_mutex_unlock(&sv->mu);

    char id[48];
    static long counter;
    snprintf(id, sizeof id, "%s-%ld", chat ? "chatcmpl" : "cmpl", __sync_add_and_fetch(&counter, 1));
    long created = (long)time(NULL);
    const char *obj = chat ? "chat.completion" : "text_completion";
    bool ok = true;
    if (stream && r->tools) { // a tool call is only known once the reply is complete: send it whole
        const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n";
        ok = send_all(c->fd, h, strlen(h));
        pthread_mutex_lock(&r->mu);
        while (!r->done) pthread_cond_wait(&r->cv, &r->mu);
        pthread_mutex_unlock(&r->mu);
        buf ev = { 0 };
        bfmt(&ev, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, created);
        bjson(&ev, sv->o->model_id, strlen(sv->o->model_id));
        bstr(&ev, ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",");
        const char *finish = reply_fields(sv, r, &ev);
        bstr(&ev, "},\"finish_reason\":null}]}");
        sse(c->fd, &ok, &ev);
        bfmt(&ev, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, created);
        bjson(&ev, sv->o->model_id, strlen(sv->o->model_id));
        bfmt(&ev, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}", finish);
        sse(c->fd, &ok, &ev);
        if (usage) {
            bfmt(&ev, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, created);
            bjson(&ev, sv->o->model_id, strlen(sv->o->model_id));
            bfmt(&ev, ",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":%d}}}", r->n_prompt, r->n_gen,
                 r->n_prompt + r->n_gen, r->n_cached);
            sse(c->fd, &ok, &ev);
        }
        const char *done = "data: [DONE]\n\n";
        if (ok) ok = chunk(c->fd, done, strlen(done)) && send_all(c->fd, "0\r\n\r\n", 5);
        free(ev.s);
    } else if (stream) {
        const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n";
        ok = send_all(c->fd, h, strlen(h));
        size_t sent[2] = { 0, 0 };
        buf ev = { 0 };
        if (chat) {
            bfmt(&ev, "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, created);
            bjson(&ev, sv->o->model_id, strlen(sv->o->model_id));
            bstr(&ev, ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}");
            sse(c->fd, &ok, &ev);
        }
        pthread_mutex_lock(&r->mu);
        for (;;) {
            while (ok && !r->done && r->text[0].n == sent[0] && r->text[1].n == sent[1]) pthread_cond_wait(&r->cv, &r->mu);
            for (int ch = 0; ch < 2 && ok; ch++) {
                if (r->text[ch].n == sent[ch]) continue;
                if (ch == CH_REASONING && !chat) { sent[ch] = r->text[ch].n; continue; }
                bfmt(&ev, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", id, chat ? "chat.completion.chunk" : "text_completion", created);
                bjson(&ev, sv->o->model_id, strlen(sv->o->model_id));
                if (chat) {
                    bstr(&ev, ch == CH_CONTENT ? ",\"choices\":[{\"index\":0,\"delta\":{\"content\":" : ",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":");
                    bjson(&ev, r->text[ch].s + sent[ch], r->text[ch].n - sent[ch]);
                    bstr(&ev, "},\"finish_reason\":null}]}");
                } else {
                    bstr(&ev, ",\"choices\":[{\"index\":0,\"text\":");
                    bjson(&ev, r->text[ch].s + sent[ch], r->text[ch].n - sent[ch]);
                    bstr(&ev, ",\"finish_reason\":null}]}");
                }
                sent[ch] = r->text[ch].n;
                pthread_mutex_unlock(&r->mu);
                sse(c->fd, &ok, &ev);
                pthread_mutex_lock(&r->mu);
            }
            if (!ok) { r->cancelled = true; break; } // client went away
            if (r->done && r->text[0].n == sent[0] && r->text[1].n == sent[1]) break;
        }
        while (!r->done) pthread_cond_wait(&r->cv, &r->mu);
        pthread_mutex_unlock(&r->mu);
        bfmt(&ev, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", id, chat ? "chat.completion.chunk" : "text_completion", created);
        bjson(&ev, sv->o->model_id, strlen(sv->o->model_id));
        bfmt(&ev, chat ? ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}" : ",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"%s\"}]}", r->finish);
        sse(c->fd, &ok, &ev);
        if (usage) {
            bfmt(&ev, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", id, chat ? "chat.completion.chunk" : "text_completion", created);
            bjson(&ev, sv->o->model_id, strlen(sv->o->model_id));
            bfmt(&ev, ",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":%d}}}", r->n_prompt, r->n_gen,
                 r->n_prompt + r->n_gen, r->n_cached);
            sse(c->fd, &ok, &ev);
        }
        const char *done = "data: [DONE]\n\n";
        if (ok) ok = chunk(c->fd, done, strlen(done)) && send_all(c->fd, "0\r\n\r\n", 5);
        free(ev.s);
    } else {
        pthread_mutex_lock(&r->mu);
        while (!r->done) pthread_cond_wait(&r->cv, &r->mu);
        pthread_mutex_unlock(&r->mu);
        buf b = { 0 };
        bfmt(&b, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", id, obj, created);
        bjson(&b, sv->o->model_id, strlen(sv->o->model_id));
        if (chat) {
            bstr(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",");
            const char *finish = reply_fields(sv, r, &b);
            bfmt(&b, "},\"finish_reason\":\"%s\"}]", finish);
        } else {
            bstr(&b, ",\"choices\":[{\"index\":0,\"text\":");
            bjson(&b, r->text[0].s ? r->text[0].s : "", r->text[0].n);
            bfmt(&b, ",\"finish_reason\":\"%s\"}]", r->finish);
        }
        bfmt(&b, ",\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":%d}}}", r->n_prompt, r->n_gen,
             r->n_prompt + r->n_gen, r->n_cached);
        ok = respond(c->fd, 200, "application/json", b.s, b.n);
        free(b.s);
    }
    fprintf(stderr, "[serve] %s: %d prompt + %d generated tokens, first token %.2fs, total %.2fs, %s\n", id, r->n_prompt, r->n_gen,
            r->t_first ? r->t_first - r->t_submit : 0, r->t_done - r->t_submit, r->finish);
    free_req(r);
}

static void *conn_loop(void *arg) {
    conn *c = arg;
    buf in = { 0 };
    for (;;) {
        // read headers
        char *end;
        while (!(in.s && (end = strstr(in.s, "\r\n\r\n")))) {
            char tmp[65536];
            ssize_t k = recv(c->fd, tmp, sizeof tmp, 0);
            if (k <= 0) goto done;
            bput(&in, tmp, (size_t)k);
        }
        size_t hlen = (size_t)(end - in.s) + 4;
        char method[8] = { 0 }, path[256] = { 0 };
        sscanf(in.s, "%7s %255s", method, path);
        size_t clen = 0;
        for (char *h = in.s; h < end; h = strstr(h, "\r\n") + 2) {
            if (!strncasecmp(h, "Content-Length:", 15)) clen = strtoul(h + 15, NULL, 10);
            if (!strstr(h, "\r\n")) break;
        }
        while (in.n < hlen + clen) {
            char tmp[65536];
            ssize_t k = recv(c->fd, tmp, sizeof tmp, 0);
            if (k <= 0) goto done;
            bput(&in, tmp, (size_t)k);
        }
        char *q = strchr(path, '?');
        if (q) *q = 0;
        const char *body = in.s + hlen;
        if (!strcmp(method, "GET") && (!strcmp(path, "/health") || !strcmp(path, "/v1/health"))) respond(c->fd, 200, "application/json", "{\"status\":\"ok\"}", 15);
        else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
            buf b = { 0 };
            bstr(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
            bjson(&b, c->sv->o->model_id, strlen(c->sv->o->model_id));
            bstr(&b, ",\"object\":\"model\",\"owned_by\":\"FleetLLM\"}]}");
            respond(c->fd, 200, "application/json", b.s, b.n);
            free(b.s);
        } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/chat/completions")) handle_completion(c, body, clen, true);
        else if (!strcmp(method, "POST") && !strcmp(path, "/v1/completions")) handle_completion(c, body, clen, false);
        else respond_error(c->fd, 404, "unknown endpoint");
        memmove(in.s, in.s + hlen + clen, in.n - hlen - clen);
        in.n -= hlen + clen;
        in.s[in.n] = 0;
    }
done:
    close(c->fd);
    free(in.s);
    free(c);
    return NULL;
}

int server_run(engine *e, const server_opts *o) {
    signal(SIGPIPE, SIG_IGN);
    server *sv = calloc(1, sizeof *sv);
    sv->e = e;
    sv->tok = e->tok;
    sv->fam = chat_detect(e->tok);
    sv->o = o;
    sv->t_start = now_secs();
    pthread_mutex_init(&sv->mu, NULL);
    pthread_cond_init(&sv->cv, NULL);
    sv->t_start_id = tok_lookup(e->tok, "<|start|>");
    sv->t_message = tok_lookup(e->tok, "<|message|>");
    sv->t_eom = tok_lookup(e->tok, "<|eom|>");
    sv->t_eot = tok_lookup(e->tok, "<|eot|>");
    sv->t_eos = tok_lookup(e->tok, "<|end_of_text|>");
    sv->t_think = tok_lookup(e->tok, "<think>");
    sv->t_end_think = tok_lookup(e->tok, "</think>");
    pthread_t th;
    pthread_create(&th, NULL, engine_loop, sv);
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = { 0 };
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)o->port);
    a.sin_addr.s_addr = inet_addr(o->host ? o->host : "127.0.0.1");
    if (bind(fd, (struct sockaddr *)&a, sizeof a) || listen(fd, 256)) { perror("serve: bind/listen"); return 1; }
    fprintf(stderr, "[serve] listening on http://%s:%d/v1 (model \"%s\")\n", o->host ? o->host : "127.0.0.1", o->port, o->model_id);
    for (;;) {
        int cfd = accept(fd, NULL, NULL);
        if (cfd < 0) { if (errno == EINTR) continue; perror("serve: accept"); return 1; }
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        conn *c = calloc(1, sizeof *c);
        c->fd = cfd;
        c->sv = sv;
        pthread_t ct;
        pthread_create(&ct, NULL, conn_loop, c);
        pthread_detach(ct);
    }
}
