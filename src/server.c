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
#include <stdatomic.h>
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

enum { CH_CONTENT, CH_REASONING, CH_TOOL }; // Glimmer tool-recipient channel
enum { PS_HEADER, PS_MESSAGE, PS_EXPECT_START }; // Glimmer channel parser

typedef struct req {
    int seq; // engine sequence id
    int32_t *prompt; int n_prompt;
    sample_params p;
    char *stop[4]; int n_stop_str;
    // output, guarded by mu
    pthread_mutex_t mu;
    pthread_cond_t cv;
    buf text[3];      // content, reasoning, tool-channel (complete UTF-8 only)
    buf raw[3];       // bytes not yet known to be complete UTF-8
    int n_gen;        // generated tokens
    int checkpoint_at;
    int n_cached;     // prompt tokens whose KV was reused (prefix sharing)
    bool done, started;
    atomic_bool cancelled;
    bool tools;       // tools were offered: the reply is parsed for tool calls (and not streamed piecemeal)
    char **tool_names; int n_tool_names;
    const char *finish;
    double t_submit, t_first, t_done;
    // parser (engine thread only)
    int pstate, channel, seen;
    char header[256]; int header_len;
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
    // Engine-owned counters, copied here by the engine thread after every
    // step under mu so telemetry readers never race the engine.
    struct {
        int active;
        long served, prompt_tokens, generated_tokens;
        size_t kv_total_bytes, kv_free_bytes;
        uint32_t full_free, full_total, swa_free, swa_total;
        int recurrent_used, recurrent_limit;
        long shared_tokens, prefill_rows, decode_rows, gen_tokens, deduped_tokens;
        long preemptions, swap_outs, swap_ins;
    } snap;
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

// Classify a Glimmer message header ("<|start|>assistant to=NAME" or the bare
// " to=NAME" opening a turn). Only an exact "self" is reasoning and only an
// exact "user" (or no recipient) is content; anything else addresses a tool,
// so its ATEM is parsed for calls while echoes in other channels never are.
static int glimmer_channel(const char *header) {
    const char *to = strstr(header, "to=");
    if (!to) return CH_CONTENT;
    to += 3;
    size_t n = 0;
    while ((to[n] >= 'A' && to[n] <= 'Z') || (to[n] >= 'a' && to[n] <= 'z') || (to[n] >= '0' && to[n] <= '9') ||
           to[n] == '_' || to[n] == '.' || to[n] == '-') n++;
    if (n == 4 && !memcmp(to, "self", 4)) return CH_REASONING;
    if (n == 0 || (n == 4 && !memcmp(to, "user", 4))) return CH_CONTENT;
    return CH_TOOL;
}

// Route one generated token into content / reasoning / tool text (engine thread, r->mu held).
static void route_token(server *sv, req *r, int32_t t) {
    char piece[256];
    if (sv->fam == CHAT_GLIMMER) {
        if (t == sv->t_eot || t == sv->t_eos) {
            if (r->channel == CH_TOOL) out_bytes(r, CH_TOOL, "<|eot|>", 7);
            return;
        }
        if (t == sv->t_start_id && r->channel == CH_TOOL)
            out_bytes(r, CH_TOOL, "<|eom|>", 7);
        switch (r->pstate) {
        case PS_EXPECT_START:
            if (t == sv->t_start_id) { r->pstate = PS_HEADER; r->header_len = 0; return; }
            r->pstate = PS_MESSAGE; // unexpected: treat as text of the current channel
            break;
        case PS_HEADER:
            if (t == sv->t_message) {
                r->header[r->header_len] = 0;
                r->channel = glimmer_channel(r->header);
                if (r->channel == CH_TOOL) {
                    // Preserve channel boundaries for the strict ATEM parser;
                    // stripped/unframed markup is never executable.
                    const char *to = strstr(r->header, "to=");
                    out_bytes(r, CH_TOOL, "<|start|>assistant ", 19);
                    out_bytes(r, CH_TOOL, to, strlen(to));
                    out_bytes(r, CH_TOOL, "<|message|>", 11);
                }
                r->pstate = PS_MESSAGE;
            } else {
                size_t k = tok_piece(sv->tok, t, piece, sizeof piece);
                if (r->header_len + k < sizeof r->header - 1) { memcpy(r->header + r->header_len, piece, k); r->header_len += (int)k; }
            }
            return;
        default:
            // A fresh header inside a message (the direct-answer prefix ends
            // with <|message|>, but the model may still open channels).
            if (t == sv->t_start_id) { r->pstate = PS_HEADER; r->header_len = 0; return; }
            break;
        }
        if (t == sv->t_eom) {
            if (r->channel == CH_TOOL) out_bytes(r, CH_TOOL, "<|eom|>", 7);
            r->pstate = PS_EXPECT_START; return;
        }
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

// Copy engine-owned counters into the telemetry snapshot (mu held).
static void snap_engine(server *sv) {
    engine *e = sv->e;
    sv->snap.active = sv->n_active;
    sv->snap.served = sv->served;
    sv->snap.prompt_tokens = sv->prompt_tokens;
    sv->snap.generated_tokens = sv->gen_tokens;
    sv->snap.kv_total_bytes = e->kv.full.bytes + e->kv.swa.bytes;
    sv->snap.kv_free_bytes = (size_t)e->kv.full.n_free * kv_page_bytes(&e->kv.full) +
                            (size_t)e->kv.swa.n_free * kv_page_bytes(&e->kv.swa);
    sv->snap.full_free = e->kv.full.n_free;
    sv->snap.full_total = e->kv.full.n_pages;
    sv->snap.swa_free = e->kv.window ? e->kv.swa.n_free : 0;
    sv->snap.swa_total = e->kv.window ? e->kv.swa.n_pages : 0;
    sv->snap.recurrent_used = e->kv.recurrent_used;
    sv->snap.recurrent_limit = e->kv.recurrent_limit;
    sv->snap.shared_tokens = e->st.shared_tokens;
    sv->snap.prefill_rows = e->st.prefill_rows;
    sv->snap.decode_rows = e->st.decode_rows;
    sv->snap.gen_tokens = e->st.gen_tokens;
    sv->snap.deduped_tokens = e->st.deduped_tokens;
    sv->snap.preemptions = e->st.preemptions;
    sv->snap.swap_outs = e->st.swap_outs;
    sv->snap.swap_ins = e->st.swap_ins;
}

// sv->mu held. Detach before acknowledgment: waiters may free the request
// immediately after done. Cancellation must not wait for an admission slot.
static void drain_cancelled(server *sv) {
    req **at = &sv->queue;
    while (*at) {
        req *r = *at;
        if (!r->cancelled) { at = &r->next; continue; }
        *at = r->next;
        pthread_mutex_lock(&r->mu);
        r->finish = "cancelled";
        r->t_done = wall();
        r->done = true;
        pthread_cond_broadcast(&r->cv);
        pthread_mutex_unlock(&r->mu);
    }
}

static void *engine_loop(void *arg) {
    server *sv = arg;
    engine *e = sv->e;
    int *ids = NULL, cap_ids = 0;
    double t_log = now_secs();
    long gen_log = 0;
    for (;;) {
        pthread_mutex_lock(&sv->mu);
        while (!sv->n_active && !sv->queue) pthread_cond_wait(&sv->cv, &sv->mu);
        drain_cancelled(sv);
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
            r->n_cached = e->seqs[r->seq].prefix_cached;
            if (e->prompt_checkpoints) e->seqs[r->seq].checkpoint_at = r->checkpoint_at;
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
            r->n_cached = s->prefix_cached;
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
                r->finish = r->cancelled ? "cancelled" : stopped || (s->done && s->n_gen < s->p.max_new) ? "stop" : "length";
                r->done = true;
                r->t_done = wall();
                engine_retire(e, r->seq); // keep the KV for prefix reuse (multi-turn agents) until memory is needed
                sv->served++;
                if (getenv("FLEET_TRACE_STATS"))
                    fprintf(stderr,"[fleet_stats] served=%ld rows=%ld shared=%ld preemptions=%ld checkpoints=%ld skipped=%ld state_moves=%ld fullslot_misses=%ld slot_preemptions=%ld endpoint_evictions=%ld checkpoint_evictions=%ld states=%d/%d\n",
                            sv->served,e->st.prefill_rows+e->st.decode_rows,e->st.shared_tokens,e->st.preemptions,e->st.checkpoints,
                            e->st.skipped_checkpoints,e->st.state_moves,e->st.fullslot_match_misses,e->st.slot_preemptions,
                            e->st.endpoint_evictions,e->st.checkpoint_evictions,e->kv.recurrent_used,e->kv.recurrent_limit);
            }
            pthread_cond_broadcast(&r->cv);
            pthread_mutex_unlock(&r->mu);
            if (!fin) sv->active[k++] = r;
        }
        sv->n_active = k;
        pthread_mutex_lock(&sv->mu); // publish engine-owned counters for telemetry (this thread owns the engine)
        snap_engine(sv);
        pthread_mutex_unlock(&sv->mu);
        double t = now_secs();
        if (t - t_log > 10) {
            fprintf(stderr, "[serve] %d active | %.1f tok/s generated | %ld served | %ld preemptions | %ld shared prompt tokens | swap %ld out %ld in (%.2fs, %.2f GB held)"
                            " | %ld tokens reused in %ld spans (%.2fs) | drafts %ld/%ld accepted in %ld verifies"
                            " | forward_rows %ld checkpoints %ld skipped %ld state_moves %ld fullslot_misses %ld slot_preemptions %ld\n",
                    sv->n_active, (sv->gen_tokens - gen_log) / (t - t_log), sv->served, e->st.preemptions, e->st.shared_tokens, e->st.swap_outs, e->st.swap_ins,
                    e->st.swap_secs, e->swap_used / 1e9, e->st.reuse_tokens, e->st.reuse_moves, e->st.reuse_secs, e->st.spec_accepted,
                    e->st.spec_drafted, e->st.spec_steps, e->st.prefill_rows+e->st.decode_rows, e->st.checkpoints,
                    e->st.skipped_checkpoints, e->st.state_moves, e->st.fullslot_match_misses, e->st.slot_preemptions);
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

// Text of a message's content: a string, or an array of parts. Text parts
// pass through; with media enabled (Glimmer render_content) image and video
// parts become their channel placeholders, anything else is ignored.
static void content_text(const json *j, const jnode *c, buf *out, bool media) {
    if (!c) return;
    if (c->type == J_STR) { bput(out, c->s, c->n); return; }
    for (const jnode *p = c->type == J_ARR ? json_first(j, c) : NULL; p; p = json_next(j, p)) {
        const char *ty = json_str(j, p, "type"), *tx = json_str(j, p, "text");
        if (!ty || !strcmp(ty, "text")) { if (tx) bstr(out, tx); }
        else if (media && !strcmp(ty, "image")) bstr(out, "<|patch|>");
        else if (media && !strcmp(ty, "video")) bstr(out, "<|video|>");
    }
}

// The fields of a finished chat reply's message (or delta): content,
// reasoning and, when tools were offered, the tool calls parsed from it.
// Returns the finish reason.
static pthread_once_t call_nonce_once = PTHREAD_ONCE_INIT;
static char call_nonce[80];
static void init_call_nonce(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    snprintf(call_nonce, sizeof call_nonce, "%lx_%lx_%lx", (unsigned long)getpid(),
             (unsigned long)ts.tv_sec, (unsigned long)ts.tv_nsec);
}

static const char *reply_fields(server *sv, req *r, buf *b) {
    static long next_call;
    const char *content = r->text[CH_CONTENT].s ? r->text[CH_CONTENT].s : "";
    size_t content_n = r->text[CH_CONTENT].n;
    const char *finish = r->finish;
    char *rest = NULL;
    // Glimmer calls come from the tool-recipient channel only (echoes in
    // reasoning or the final answer stay text); other families parse the
    // answer. A capped response must not execute even a complete earlier
    // call: later arguments may be truncated. Keep the original stop reason.
    const char *src = sv->fam == CHAT_GLIMMER ? (r->text[CH_TOOL].s ? r->text[CH_TOOL].s : "") : content;
    int capacity = 1;
    const char *marker = sv->fam == CHAT_GLIMMER ? "<atem:invoke" : "<tool_call>";
    for (const char *p = src; (p = strstr(p, marker)); p += strlen(marker)) capacity++;
    chat_call *calls = calloc((size_t)capacity, sizeof *calls);
    int nc = r->tools && !r->cancelled && !strcmp(finish, "stop")
        ? chat_parse_calls_tools(sv->fam, src, calls, capacity, &rest,
                                 (const char *const *)r->tool_names, r->n_tool_names) : 0;
    if (nc) {
        pthread_once(&call_nonce_once, init_call_nonce);
        if (sv->fam == CHAT_GLIMMER) {
            if (content_n) { bstr(b, "\"content\":"); bjson(b, content, content_n); }
            else bstr(b, "\"content\":null");
        } else {
            const char *p = rest;
            while (*p == ' ' || *p == '\n') p++;
            if (*p) { bstr(b, "\"content\":"); bjson(b, rest, strlen(rest)); }
            else bstr(b, "\"content\":null");
        }
        bstr(b, ",\"tool_calls\":[");
        for (int i = 0; i < nc; i++) {
            bfmt(b, "%s{\"index\":%d,\"id\":\"call_%s_%ld\",\"type\":\"function\",\"function\":{\"name\":", i ? "," : "", i, call_nonce, __sync_add_and_fetch(&next_call, 1));
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
        bjson(b, content, content_n);
    }
    free(rest);
    free(calls);
    if (r->text[CH_REASONING].n) { bstr(b, ",\"reasoning_content\":"); bjson(b, r->text[CH_REASONING].s, r->text[CH_REASONING].n); }
    return finish;
}

static void free_req(req *r) {
    free(r->prompt);
    for (int i = 0; i < 3; i++) { free(r->text[i].s); free(r->raw[i].s); }
    for (int i = 0; i < r->n_stop_str; i++) free(r->stop[i]);
    for (int i = 0; i < r->n_tool_names; i++) free(r->tool_names[i]);
    free(r->tool_names);
    pthread_mutex_destroy(&r->mu);
    pthread_cond_destroy(&r->cv);
    free(r);
}

// True when the client is gone: the socket read EOF, or failed for a reason
// other than "no data right now". A nonblocking peek, so it never disturbs a
// live keep-alive connection.
bool fleet_peer_gone(int fd) {
    char b;
    ssize_t k = recv(fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
    if (k == 0) return true;
    if (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return true;
    return false;
}

// Wait for r->done, polling for a vanished client in between: a cancelled
// request stops costing engine work (active) or is skipped before admission
// (queued, including buffered tool responses that only finish at the end).
static void wait_done(req *r, int fd) {
    pthread_mutex_lock(&r->mu);
    while (!r->done) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 50 * 1000000;
        if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
        pthread_cond_timedwait(&r->cv, &r->mu, &ts);
        if (!r->done && fleet_peer_gone(fd)) r->cancelled = true;
    }
    pthread_mutex_unlock(&r->mu);
}

// Structured telemetry for supervisor hardware admission and terminal
// display. capabilities states what this server supports (incremental tool
// streaming is false: tool calls buffer to completion and stream once, while
// engine activity stays visible through stats). stats reports live numbers
// only: server active/queued/served plus the engine thread's snapshot (KV
// free/total for the full and SWA pools, shared/generated/prefill counters).
// Queued requests are queued, never reported as decoding.
static void handle_fleet(server *sv, int fd, const char *path) {
    buf b = { 0 };
    if (!strcmp(path, "/v1/fleet/capabilities")) {
        const char *fam = sv->fam == CHAT_GLIMMER ? "glimmer" : sv->fam == CHAT_CHATML ? "chatml" : sv->fam == CHAT_LLAMA3 ? "llama3" : "plain";
        const char *tools = sv->fam == CHAT_GLIMMER ? "atem" : sv->fam == CHAT_CHATML ? "tool_call" : sv->fam == CHAT_LLAMA3 ? "llama_json" : "none";
        const char *reason = sv->fam == CHAT_GLIMMER ? "to_self" : sv->t_think >= 0 ? "think" : "none";
        engine *e = sv->e;
        bstr(&b, "{\"engine\":\"fleet\",\"context_window\":null,\"model\":");
        bjson(&b, sv->o->model_id, strlen(sv->o->model_id));
        bstr(&b, ",\"family\":");
        bjson(&b, fam, strlen(fam));
        bstr(&b, ",\"tools\":{\"format\":");
        bjson(&b, tools, strlen(tools));
        bfmt(&b, ",\"multi_call\":true,\"incremental_streaming\":false,\"auto_parse\":%s},\"reasoning\":{\"channel\":", sv->fam == CHAT_PLAIN ? "false" : "true");
        bjson(&b, reason, strlen(reason));
        bfmt(&b, ",\"direct_supported\":true},\"streaming\":true,\"usage\":true,\"prefix_sharing\":true,\"swa_coalescing\":%s,",
             e->recurrent_prefix_dedup && e->kv.window && !e->draft ? "true" : "false");
        bfmt(&b, "\"learned_drafter\":%s,\"ngram_drafts\":%s,\"prompt_checkpoints\":%s,\"admission\":{\"max_active\":%d,\"memory_aware\":%s}",
             e->draft ? "true" : "false", e->ngram ? "true" : "false", e->prompt_checkpoints ? "true" : "false",
             sv->o->max_active, sv->o->admit ? "true" : "false");
        bfmt(&b, ",\"max_active\":%d,\"features\":{\"native_tools\":%s,\"tool_streaming\":\"buffered\",\"shared_prefix_coalescing\":%s,\"ngram_spec\":%s,\"learned_drafter\":%s}}",
             sv->o->max_active, sv->fam != CHAT_PLAIN && !tok_qwen35(sv->tok) ? "true" : "false",
             e->recurrent_prefix_dedup && e->kv.window && !e->draft ? "true" : "false",
             e->ngram ? "true" : "false", e->draft ? "true" : "false");
    } else if (!strcmp(path, "/v1/fleet/stats")) {
        pthread_mutex_lock(&sv->mu);
        int queued = 0;
        for (req *r = sv->queue; r; r = r->next) queued++;
        int active = sv->snap.active;
        long served = sv->snap.served, ptok = sv->snap.prompt_tokens, gtok = sv->snap.generated_tokens;
        double uptime = now_secs() - sv->t_start;
        uint32_t ff = sv->snap.full_free, ft = sv->snap.full_total, sf = sv->snap.swa_free, st = sv->snap.swa_total;
        size_t total_bytes = sv->snap.kv_total_bytes, free_bytes = sv->snap.kv_free_bytes;
        int ru = sv->snap.recurrent_used, rl = sv->snap.recurrent_limit;
        long shared = sv->snap.shared_tokens, pre = sv->snap.prefill_rows, dec = sv->snap.decode_rows;
        long egen = sv->snap.gen_tokens, dedup = sv->snap.deduped_tokens;
        long preempt = sv->snap.preemptions, sout = sv->snap.swap_outs, sin = sv->snap.swap_ins;
        pthread_mutex_unlock(&sv->mu);
        bfmt(&b, "{\"active\":%d,\"queued\":%d,\"served\":%ld,\"prompt_tokens\":%ld,\"generated_tokens\":%ld,\"uptime_s\":%.1f", active, queued, served, ptok, gtok, uptime);
        bfmt(&b, ",\"engine\":{\"shared_tokens\":%ld,\"prefill_rows\":%ld,\"decode_rows\":%ld,\"gen_tokens\":%ld,\"deduped_tokens\":%ld,\"preemptions\":%ld,\"swap_outs\":%ld,\"swap_ins\":%ld}",
             shared, pre, dec, egen, dedup, preempt, sout, sin);
        bfmt(&b, ",\"tokens\":{\"prefill\":%ld,\"generated\":%ld,\"shared_prefix\":%ld},\"requests\":{\"completed\":%ld}", pre, gtok, shared, served);
        bfmt(&b, ",\"kv\":{\"total_bytes\":%zu,\"free_bytes\":%zu,\"full\":%s,\"full_free\":%u,\"full_total\":%u,\"swa_free\":%u,\"swa_total\":%u,\"recurrent_used\":%d,\"recurrent_limit\":%d},\"memory\":{\"rss_bytes\":null,\"physical_footprint_bytes\":null}}",
             total_bytes, free_bytes, (ft && !ff) || (st && !sf) ? "true" : "false",
             ff, ft, sf, st, ru, rl);
    } else {
        respond_error(fd, 404, "unknown endpoint");
        return;
    }
    respond(fd, 200, "application/json", b.s ? b.s : "{}", b.s ? b.n : 2);
    free(b.s);
}

static void handle_completion(conn *c, const char *body, size_t len, bool chat) {
    server *sv = c->sv;
    json *j = json_parse(body, len);
    if (!j) { respond_error(c->fd, 400, "invalid JSON"); return; }
    const jnode *root = json_root(j);
    req *r = calloc(1, sizeof *r);
    atomic_init(&r->cancelled, false);
    pthread_mutex_init(&r->mu, NULL);
    pthread_cond_init(&r->cv, NULL);
    // prompt
    char *text = NULL;
    bool direct = false; // "none" reasoning: the prompt ends inside the answer channel
    if (chat) {
        const jnode *msgs = json_get(j, root, "messages");
        if (!msgs || msgs->type != J_ARR || !msgs->n) { respond_error(c->fd, 400, "messages must be a non-empty array"); json_free(j); free_req(r); return; }
        if (tok_qwen35(sv->tok)) {
            bool unsupported=false;
            for(const jnode *m=json_first(j,msgs);m;m=json_next(j,m)) {
                const char *role=json_str(j,m,"role");
                const jnode *calls=json_get(j,m,"tool_calls"),*content=json_get(j,m,"content");
                unsupported |= (role && !strcmp(role,"tool")) || (calls && calls->type==J_ARR && calls->n);
                for(const jnode *part=content && content->type==J_ARR ? json_first(j,content) : NULL;part;part=json_next(j,part)) {
                    const char *type=json_str(j,part,"type");unsupported |= !type || strcmp(type,"text");
                }
            }
            if(unsupported) {respond_error(c->fd,400,"Qwen3.5 currently supports text messages and JSON actions; native tool calls and media are not implemented");json_free(j);free_req(r);return;}
        }
        chat_msg *cm = calloc(msgs->n, sizeof(chat_msg));
        buf *bodies = calloc(msgs->n, sizeof(buf));
        int k = 0, n_calls = 0;
        for (const jnode *m = json_first(j, msgs); m; m = json_next(j, m)) {
            const jnode *tc = json_get(j, m, "tool_calls");
            n_calls += tc && tc->type == J_ARR ? tc->n : 0;
        }
        chat_call *calls = calloc(n_calls ? n_calls : 1, sizeof(chat_call));
        char **argdump = calloc(n_calls ? n_calls : 1, sizeof(char *));
        bool glimmer_media = sv->fam == CHAT_GLIMMER;
        n_calls = 0;
        for (const jnode *m = json_first(j, msgs); m; m = json_next(j, m), k++) {
            content_text(j, json_get(j, m, "content"), &bodies[k], glimmer_media);
            const char *role = json_str(j, m, "role");
            cm[k].role = role ? role : "user";
            cm[k].content = bodies[k].s ? bodies[k].s : "";
            cm[k].reasoning_content = json_str(j, m, "reasoning_content");
            if (!cm[k].reasoning_content) cm[k].reasoning_content = json_str(j, m, "reasoning");
            cm[k].recipient = json_str(j, m, "recipient");
            const jnode *end_turn = json_get(j, m, "end_turn");
            if (end_turn && end_turn->type == J_BOOL) cm[k].end_turn = end_turn->num ? 1 : -1;
            if (!strcmp(cm[k].role, "tool")) {
                cm[k].tool_call_id = json_str(j, m, "tool_call_id");
                cm[k].name = json_str(j, m, "name");
            }
            const jnode *tc = json_get(j, m, "tool_calls"); // [{id, type, function: {name, arguments}}]
            cm[k].calls = calls + n_calls;
            for (const jnode *c = tc && tc->type == J_ARR ? json_first(j, tc) : NULL; c; c = json_next(j, c)) {
                const jnode *fn = json_get(j, c, "function");
                if (!json_str(j, fn, "name")) continue;
                const jnode *ag = json_get(j, fn, "arguments");
                const char *as = NULL;
                if (!ag) as = "{}";
                else if (ag->type == J_STR) as = ag->s;
                else if (ag->type == J_OBJ) as = argdump[n_calls] = json_dump(j, ag);
                calls[n_calls++] = (chat_call){ .name = json_str(j, fn, "name"), .arguments = as, .id = json_str(j, c, "id") };
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
        r->tool_names = calloc(n_tools ? n_tools : 1, sizeof *r->tool_names);
        for (const jnode *t = r->tools ? json_first(j, tl) : NULL; t; t = json_next(j, t)) {
            const jnode *fn = json_get(j, t, "function");
            const char *name = json_str(j, fn ? fn : t, "name");
            if (name) r->tool_names[r->n_tool_names++] = strdup(name);
        }
        time_t now = time(NULL);
        char date[16];
        strftime(date, sizeof date, "%Y-%m-%d", localtime(&now));
        const char *effort = json_str(j, root, "reasoning_effort");
        const char *reasoning = effort ? effort : sv->o->reasoning;
        direct = reasoning && !strcmp(reasoning, "none");
        chat_opts o = { direct, reasoning, date, (const char **)tools, n_tools };
        text = chat_format_opts(sv->tok, cm, k, &o);
        if (tok_qwen35(sv->tok) && !direct) r->channel = CH_REASONING;
        for (int i = 0; i < k; i++) free(bodies[i].s);
        for (int i = 0; i < n_tools; i++) free(tools[i]);
        for (int i = 0; i < n_calls; i++) free(argdump[i]);
        free(argdump);
        free(tools);
        free(calls);
        free(bodies);
        free(cm);
        if (tok_qwen35(sv->tok) && r->tools) {
            free(text);
            respond_error(c->fd, 400, "Qwen3.5 native tool-call format is not implemented; use text/JSON actions");
            json_free(j);
            free_req(r);
            return;
        }
    } else {
        const char *p = json_str(j, root, "prompt");
        if (!p) { respond_error(c->fd, 400, "prompt must be a string"); json_free(j); free_req(r); return; }
        text = strdup(p);
    }
    r->n_prompt = tok_encode(sv->tok, text, strlen(text), &r->prompt);
    if (chat && sv->e->prompt_checkpoints && tok_qwen35(sv->tok)) {
        const char *header = NULL, *next = text;
        while ((next = strstr(next, "<|im_start|>assistant\n"))) {
            header = next;
            next++;
        }
        if (header) {
            int32_t *ids;
            int n = tok_encode(sv->tok, text, header - text, &ids);
            if (n > 0 && n < r->n_prompt && !memcmp(ids, r->prompt, 4 * (size_t)n)) r->checkpoint_at = n;
            free(ids);
        }
    }
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
        // A direct answer opens no channel: the prompt already ends with
        // <|message|>, so the first tokens are answer text, not a header.
        r->pstate = direct ? PS_MESSAGE : PS_HEADER;
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
        wait_done(r, c->fd); // tool calls buffer to completion; a client that left cancels here
        if (r->cancelled) { free_req(r); return; }
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
            while (ok && !r->done && r->text[CH_CONTENT].n == sent[0] && r->text[CH_REASONING].n == sent[1]) {
                struct timespec tw;
                clock_gettime(CLOCK_REALTIME, &tw);
                tw.tv_nsec += 50 * 1000000;
                if (tw.tv_nsec >= 1000000000) { tw.tv_sec++; tw.tv_nsec -= 1000000000; }
                pthread_cond_timedwait(&r->cv, &r->mu, &tw);
                if (!r->done && fleet_peer_gone(c->fd)) r->cancelled = true;
            }
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
            if (r->done && r->text[CH_CONTENT].n == sent[0] && r->text[CH_REASONING].n == sent[1]) break;
        }
        while (!r->done) pthread_cond_wait(&r->cv, &r->mu);
        pthread_mutex_unlock(&r->mu);
        if (r->cancelled) { free(ev.s); free_req(r); return; }
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
        wait_done(r, c->fd);
        if (r->cancelled) { free_req(r); return; }
        buf b = { 0 };
        bfmt(&b, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", id, obj, created);
        bjson(&b, sv->o->model_id, strlen(sv->o->model_id));
        if (chat) {
            bstr(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",");
            const char *finish = reply_fields(sv, r, &b);
            bfmt(&b, "},\"finish_reason\":\"%s\"}]", finish);
        } else {
            bstr(&b, ",\"choices\":[{\"index\":0,\"text\":");
            bjson(&b, r->text[CH_CONTENT].s ? r->text[CH_CONTENT].s : "", r->text[CH_CONTENT].n);
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
        else if (!strcmp(method, "GET") && (!strcmp(path, "/v1/fleet/capabilities") || !strcmp(path, "/v1/fleet/stats"))) handle_fleet(c->sv, c->fd, path);
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
    snap_engine(sv); // telemetry reads live numbers even before the first step
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
