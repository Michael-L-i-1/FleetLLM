#include "json.h"
#include "unicode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *p, *end; json *j; } parser;

static void ws(parser *ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\n' || *ps->p == '\r' || *ps->p == '\t')) ps->p++;
}

static int new_node(parser *ps, jtype t) {
    json *j = ps->j;
    if (j->n_nodes == j->cap) {
        j->cap = j->cap ? j->cap * 2 : 1024;
        j->nodes = realloc(j->nodes, sizeof(jnode) * j->cap);
    }
    jnode *n = &j->nodes[j->n_nodes];
    memset(n, 0, sizeof *n);
    n->type = t;
    n->first = n->next = -1;
    return j->n_nodes++;
}

static unsigned hex4(const char *s) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v = v * 16 + (unsigned)(c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : 0);
    }
    return v;
}

// Unescape a string starting after the opening quote into the arena (which is
// sized to the input, so unescaped strings always fit).
static bool parse_str(parser *ps, const char **out, int *out_len) {
    json *j = ps->j;
    char *dst = j->strs + j->strs_len, *d = dst;
    while (ps->p < ps->end && *ps->p != '"') {
        char c = *ps->p++;
        if (c != '\\') { *d++ = c; continue; }
        if (ps->p >= ps->end) return false;
        char e = *ps->p++;
        switch (e) {
        case 'n': *d++ = '\n'; break;
        case 't': *d++ = '\t'; break;
        case 'r': *d++ = '\r'; break;
        case 'b': *d++ = '\b'; break;
        case 'f': *d++ = '\f'; break;
        case 'u': {
            if (ps->end - ps->p < 4) return false;
            uint32_t cp = hex4(ps->p);
            ps->p += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && ps->end - ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u') {
                uint32_t lo = hex4(ps->p + 2);
                if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); ps->p += 6; }
            }
            d += utf8_encode(cp, d);
            break;
        }
        default: *d++ = e; break; // \" \\ \/
        }
    }
    if (ps->p >= ps->end) return false;
    ps->p++; // closing quote
    *d++ = 0;
    *out = dst;
    *out_len = (int)(d - dst - 1);
    j->strs_len += (size_t)(d - dst);
    return true;
}

static int parse_value(parser *ps);

static int parse_container(parser *ps, bool obj) {
    int id = new_node(ps, obj ? J_OBJ : J_ARR);
    ps->p++;
    int prev = -1, count = 0;
    ws(ps);
    if (ps->p < ps->end && *ps->p == (obj ? '}' : ']')) { ps->p++; return id; }
    for (;;) {
        const char *key = NULL;
        int key_len = 0;
        if (obj) {
            ws(ps);
            if (ps->p >= ps->end || *ps->p != '"') return -1;
            ps->p++;
            if (!parse_str(ps, &key, &key_len)) return -1;
            ws(ps);
            if (ps->p >= ps->end || *ps->p != ':') return -1;
            ps->p++;
        }
        int c = parse_value(ps);
        if (c < 0) return -1;
        ps->j->nodes[c].key = key;
        ps->j->nodes[c].key_len = key_len;
        if (prev < 0) ps->j->nodes[id].first = c;
        else ps->j->nodes[prev].next = c;
        prev = c;
        count++;
        ws(ps);
        if (ps->p >= ps->end) return -1;
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == (obj ? '}' : ']')) { ps->p++; break; }
        return -1;
    }
    ps->j->nodes[id].n = count;
    return id;
}

static int parse_value(parser *ps) {
    ws(ps);
    if (ps->p >= ps->end) return -1;
    char c = *ps->p;
    if (c == '{' || c == '[') return parse_container(ps, c == '{');
    if (c == '"') {
        int id = new_node(ps, J_STR);
        ps->p++;
        const char *s;
        int n;
        if (!parse_str(ps, &s, &n)) return -1;
        ps->j->nodes[id].s = s;
        ps->j->nodes[id].n = n;
        return id;
    }
    if (!strncmp(ps->p, "true", 4) || !strncmp(ps->p, "false", 5)) {
        int id = new_node(ps, J_BOOL);
        bool t = c == 't';
        ps->j->nodes[id].num = t;
        ps->p += t ? 4 : 5;
        return id;
    }
    if (!strncmp(ps->p, "null", 4)) { ps->p += 4; return new_node(ps, J_NULL); }
    if (c == '-' || (c >= '0' && c <= '9') || c == 'N' || c == 'I') {
        int id = new_node(ps, J_NUM);
        char *e;
        ps->j->nodes[id].num = strtod(ps->p, &e);
        if (e == ps->p) return -1;
        ps->p = e;
        return id;
    }
    return -1;
}

json *json_parse(const char *text, size_t len) {
    json *j = calloc(1, sizeof *j);
    j->strs_cap = len + 1;
    j->strs = malloc(j->strs_cap);
    parser ps = { text, text + len, j };
    int root = parse_value(&ps);
    if (root != 0) {
        fprintf(stderr, "json: syntax error at byte %zu\n", (size_t)(ps.p - text));
        json_free(j);
        return NULL;
    }
    return j;
}

json *json_load_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    size_t got = fread(b, 1, (size_t)n, f);
    fclose(f);
    json *j = got == (size_t)n ? json_parse(b, (size_t)n) : NULL;
    free(b);
    return j;
}

void json_free(json *j) {
    if (!j) return;
    free(j->nodes);
    free(j->strs);
    free(j);
}

const jnode *json_get(const json *j, const jnode *obj, const char *key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    size_t kl = strlen(key);
    for (const jnode *c = json_first(j, obj); c; c = json_next(j, c))
        if ((size_t)c->key_len == kl && !memcmp(c->key, key, kl)) return c;
    return NULL;
}

const jnode *json_child(const json *j, const jnode *n, int i) {
    if (!n || (n->type != J_ARR && n->type != J_OBJ)) return NULL;
    const jnode *c = json_first(j, n);
    while (c && i-- > 0) c = json_next(j, c);
    return c;
}

double json_num(const json *j, const jnode *obj, const char *key, double def) {
    const jnode *n = json_get(j, obj, key);
    return n && (n->type == J_NUM || n->type == J_BOOL) ? n->num : def;
}

bool json_bool(const json *j, const jnode *obj, const char *key, bool def) {
    const jnode *n = json_get(j, obj, key);
    return n && n->type == J_BOOL ? n->num != 0 : def;
}

const char *json_str(const json *j, const jnode *obj, const char *key) {
    const jnode *n = json_get(j, obj, key);
    return n && n->type == J_STR ? n->s : NULL;
}
