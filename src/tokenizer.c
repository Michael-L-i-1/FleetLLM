// Byte-level BPE (GPT-2 style) with hand-written pre-tokenizers that follow
// the alternation order of the HF/llama.cpp regexes for each `tokenizer.ggml.pre`:
//   llama3: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   qwen2:  same with \p{N} (one digit)
//   llama4: [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//          |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//          |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
#include "tokenizer.h"
#include "unicode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER = 4, TT_UNUSED = 5, TT_BYTE = 6 };

typedef struct { const char *s; uint32_t len; } str;

struct tokenizer {
    int n_vocab;
    str *tokens;
    uint8_t *types;
    // vocab: string -> id (open addressing, -1 empty)
    int32_t *vmap;
    uint32_t vmask;
    // merges: (a,b) -> rank, result
    uint64_t *mkeys;
    uint32_t *mrank;
    int32_t *mres;
    uint32_t mmask;
    pretok_kind pre;
    bool ignore_merges;
    int32_t specials[4096];
    int n_specials; // sorted by length, longest first
    uint8_t special_first[256];
    uint32_t byte_to_cp[256];
    int16_t cp_to_byte[512];
    int32_t stops[16];
    int n_stops;
    int32_t bos;
};

static uint64_t fnv(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static uint32_t pow2_at_least(uint64_t n) { uint32_t m = 1; while (m < n) m <<= 1; return m; }

static int32_t vfind(const tokenizer *t, const char *s, size_t n) {
    uint32_t i = (uint32_t)fnv(s, n) & t->vmask;
    while (t->vmap[i] >= 0) {
        const str *e = &t->tokens[t->vmap[i]];
        if (e->len == n && memcmp(e->s, s, n) == 0) return t->vmap[i];
        i = (i + 1) & t->vmask;
    }
    return -1;
}

static uint64_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; return k; }

static bool mfind(const tokenizer *t, int32_t a, int32_t b, uint32_t *rank, int32_t *res) {
    uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    uint32_t i = (uint32_t)mix(key) & t->mmask;
    while (t->mres[i] >= 0) {
        if (t->mkeys[i] == key) { *rank = t->mrank[i]; *res = t->mres[i]; return true; }
        i = (i + 1) & t->mmask;
    }
    return false;
}

static int cmp_special_len(const void *a, const void *b, void *ctx) {
    const tokenizer *t = ctx;
    return (int)t->tokens[*(const int32_t *)b].len - (int)t->tokens[*(const int32_t *)a].len;
}

tokenizer *tok_load(const gguf_ctx *g) {
    size_t n;
    const char *model = gguf_str(g, "tokenizer.ggml.model", &n);
    if (!model || n != 4 || memcmp(model, "gpt2", 4) != 0) {
        fprintf(stderr, "tokenizer: only byte-level BPE ('gpt2') is supported\n");
        return NULL;
    }
    const gguf_kv *tk = gguf_find(g, "tokenizer.ggml.tokens");
    const gguf_kv *mk = gguf_find(g, "tokenizer.ggml.merges");
    const gguf_kv *ty = gguf_find(g, "tokenizer.ggml.token_type");
    if (!tk || !mk) { fprintf(stderr, "tokenizer: missing tokens/merges\n"); return NULL; }
    tokenizer *t = calloc(1, sizeof *t);
    t->n_vocab = (int)tk->n;
    t->tokens = malloc(sizeof(str) * tk->n);
    t->types = malloc(tk->n);
    const uint8_t *p = tk->data;
    for (uint64_t i = 0; i < tk->n; i++) {
        size_t len;
        const char *s = gguf_arr_next_str(&p, &len);
        t->tokens[i] = (str){ s, (uint32_t)len };
        t->types[i] = ty ? (uint8_t)gguf_arr_u64(ty, i) : TT_NORMAL;
    }
    uint32_t vs = pow2_at_least(tk->n * 2);
    t->vmask = vs - 1;
    t->vmap = malloc(sizeof(int32_t) * vs);
    memset(t->vmap, 0xff, sizeof(int32_t) * vs);
    for (int i = 0; i < t->n_vocab; i++) {
        uint32_t h = (uint32_t)fnv(t->tokens[i].s, t->tokens[i].len) & t->vmask;
        while (t->vmap[h] >= 0) {
            // keep the first id for duplicate strings
            if (t->tokens[t->vmap[h]].len == t->tokens[i].len && memcmp(t->tokens[t->vmap[h]].s, t->tokens[i].s, t->tokens[i].len) == 0) goto next;
            h = (h + 1) & t->vmask;
        }
        t->vmap[h] = i;
    next:;
    }
    uint32_t ms = pow2_at_least(mk->n * 2 + 2);
    t->mmask = ms - 1;
    t->mkeys = malloc(sizeof(uint64_t) * ms);
    t->mrank = malloc(sizeof(uint32_t) * ms);
    t->mres = malloc(sizeof(int32_t) * ms);
    memset(t->mres, 0xff, sizeof(int32_t) * ms);
    p = mk->data;
    char buf[1024];
    for (uint64_t r = 0; r < mk->n; r++) {
        size_t len;
        const char *s = gguf_arr_next_str(&p, &len);
        const char *sp = memchr(s, ' ', len);
        if (!sp) continue;
        size_t la = sp - s, lb = len - la - 1;
        if (la + lb >= sizeof buf) continue;
        int32_t a = vfind(t, s, la), b = vfind(t, sp + 1, lb);
        memcpy(buf, s, la);
        memcpy(buf + la, sp + 1, lb);
        int32_t res = vfind(t, buf, la + lb);
        if (a < 0 || b < 0 || res < 0) continue;
        uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
        uint32_t i = (uint32_t)mix(key) & t->mmask;
        while (t->mres[i] >= 0 && t->mkeys[i] != key) i = (i + 1) & t->mmask;
        if (t->mres[i] >= 0) continue; // keep the higher-priority (earlier) merge
        t->mkeys[i] = key;
        t->mrank[i] = (uint32_t)r;
        t->mres[i] = res;
    }
    const char *pre = gguf_str(g, "tokenizer.ggml.pre", &n);
    t->pre = PRE_LLAMA3;
    t->ignore_merges = true;
    if (pre && n == 6 && memcmp(pre, "llama4", 6) == 0) t->pre = PRE_LLAMA4;
    else if (pre && ((n == 5 && memcmp(pre, "qwen2", 5) == 0) || (n == 16 && memcmp(pre, "deepseek-r1-qwen", 16) == 0))) { t->pre = PRE_QWEN2; t->ignore_merges = false; }
    // special tokens: control and user-defined, matched verbatim, longest first
    for (int i = 0; i < t->n_vocab && t->n_specials < 4096; i++)
        if ((t->types[i] == TT_CONTROL || t->types[i] == TT_USER) && t->tokens[i].len > 0) t->specials[t->n_specials++] = i;
    // insertion sort by length, longest first (qsort_r's signature differs across libcs)
    for (int i = 1; i < t->n_specials; i++) {
        int32_t v = t->specials[i];
        int j = i - 1;
        while (j >= 0 && cmp_special_len(&t->specials[j], &v, t) > 0) { t->specials[j + 1] = t->specials[j]; j--; }
        t->specials[j + 1] = v;
    }
    for (int i = 0; i < t->n_specials; i++) t->special_first[(unsigned char)t->tokens[t->specials[i]].s[0]] = 1;
    // GPT-2 bytes_to_unicode
    int nb = 0;
    for (int b = 0; b < 256; b++) {
        bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        t->byte_to_cp[b] = printable ? (uint32_t)b : (uint32_t)(256 + nb++);
    }
    for (int i = 0; i < 512; i++) t->cp_to_byte[i] = -1;
    for (int b = 0; b < 256; b++) t->cp_to_byte[t->byte_to_cp[b]] = (int16_t)b;
    // stop tokens
    static const char *stop_names[] = { "<|eot|>", "<|eom|>", "<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "<end_of_turn>" };
    for (size_t i = 0; i < sizeof stop_names / sizeof *stop_names; i++) {
        int32_t id = vfind(t, stop_names[i], strlen(stop_names[i]));
        if (id >= 0) t->stops[t->n_stops++] = id;
    }
    const char *idk[] = { "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id" };
    for (int i = 0; i < 3; i++) {
        uint64_t v;
        if (gguf_u64(g, idk[i], &v) && t->n_stops < 16 && !tok_is_stop(t, (int32_t)v)) t->stops[t->n_stops++] = (int32_t)v;
    }
    uint64_t b;
    t->bos = gguf_u64(g, "tokenizer.ggml.bos_token_id", &b) ? (int32_t)b : -1;
    return t;
}

void tok_free(tokenizer *t) {
    if (!t) return;
    free(t->tokens); free(t->types); free(t->vmap); free(t->mkeys); free(t->mrank); free(t->mres);
    free(t);
}

bool tok_is_stop(const tokenizer *t, int32_t id) {
    for (int i = 0; i < t->n_stops; i++) if (t->stops[i] == id) return true;
    return false;
}
int tok_vocab_size(const tokenizer *t) { return t->n_vocab; }
int32_t tok_bos(const tokenizer *t) { return t->bos; }
int32_t tok_lookup(const tokenizer *t, const char *s) { return vfind(t, s, strlen(s)); }

// ---------------------------------------------------------------- output buffer
typedef struct { int32_t *v; int n, cap; } ivec;
static void push(ivec *o, int32_t x) {
    if (o->n == o->cap) { o->cap = o->cap ? o->cap * 2 : 64; o->v = realloc(o->v, sizeof(int32_t) * o->cap); }
    o->v[o->n++] = x;
}

// ---------------------------------------------------------------- BPE on one pre-token
static void bpe(const tokenizer *t, const char *s, size_t n, ivec *out) {
    // byte-level encode
    char enc[4096];
    size_t m = 0;
    int32_t sym[1024];
    int ns = 0;
    char *big = NULL;
    char *e = enc;
    if (n * 2 + 8 > sizeof enc) e = big = malloc(n * 2 + 8);
    for (size_t i = 0; i < n; i++) m += utf8_encode(t->byte_to_cp[(unsigned char)s[i]], e + m);
    if (t->ignore_merges) {
        int32_t id = vfind(t, e, m);
        if (id >= 0) { push(out, id); free(big); return; }
    }
    int32_t *syms = sym;
    if (n > 1024) syms = malloc(sizeof(int32_t) * n);
    for (size_t i = 0, off = 0; i < n; i++) {
        int len = utf8_encode(t->byte_to_cp[(unsigned char)s[i]], (char[4]){0});
        syms[ns++] = vfind(t, e + off, len);
        off += len;
    }
    for (;;) {
        uint32_t best = UINT32_MAX;
        int pos = -1;
        int32_t res = -1;
        for (int k = 0; k + 1 < ns; k++) {
            uint32_t r;
            int32_t rs;
            if (syms[k] >= 0 && syms[k + 1] >= 0 && mfind(t, syms[k], syms[k + 1], &r, &rs) && r < best) { best = r; pos = k; res = rs; }
        }
        if (pos < 0) break;
        syms[pos] = res;
        memmove(&syms[pos + 1], &syms[pos + 2], sizeof(int32_t) * (ns - pos - 2));
        ns--;
    }
    for (int k = 0; k < ns; k++) if (syms[k] >= 0) push(out, syms[k]);
    if (syms != sym) free(syms);
    free(big);
}

// ---------------------------------------------------------------- pre-tokenizers
static inline bool isL(uint32_t c) { return ucat(c) & U_L; }
static inline bool isN(uint32_t c) { return ucat(c) & U_N; }
static inline bool isS(uint32_t c) { return ucat(c) & U_SPACE; }
static inline bool isCRLF(uint32_t c) { return c == '\r' || c == '\n'; }
static inline bool isUp(uint32_t c) { return ucat(c) & (U_LU | U_LT | U_LM | U_LO | U_M); }
static inline bool isLo(uint32_t c) { return ucat(c) & (U_LL | U_LM | U_LO | U_M); }
static inline bool isPre(uint32_t c) { return !isCRLF(c) && !isL(c) && !isN(c); } // [^\r\n\p{L}\p{N}]
static inline bool isPunct(uint32_t c) { return !isS(c) && !isL(c) && !isN(c); } // [^\s\p{L}\p{N}]
static inline uint32_t lc(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

static int contraction(const uint32_t *c, int n, int i) {
    if (i >= n || c[i] != '\'' || i + 1 >= n) return 0;
    uint32_t a = lc(c[i + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (i + 2 < n) {
        uint32_t b = lc(c[i + 2]);
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    return 0;
}

static int run(const uint32_t *c, int n, int i, bool (*f)(uint32_t)) { int j = i; while (j < n && f(c[j])) j++; return j - i; }

// alternatives shared by all three patterns, in order: digits, punct, newlines, spaces
static int tail_alts(const uint32_t *c, int n, int i, int max_digits, bool slash) {
    if (isN(c[i])) { int r = run(c, n, i, isN); return r < max_digits ? r : max_digits; }
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
    int s = (c[i] == ' ' && i + 1 < n && isPunct(c[i + 1])) ? i + 1 : i;
    int r = run(c, n, s, isPunct);
    if (r > 0) {
        int j = s + r;
        while (j < n && (isCRLF(c[j]) || (slash && c[j] == '/'))) j++;
        return j - i;
    }
    if (isS(c[i])) {
        int j = i + run(c, n, i, isS);
        for (int k = j - 1; k >= i; k--) if (isCRLF(c[k])) return k + 1 - i; // \s*[\r\n]+
        if (j == n) return j - i;                                           // \s+(?!\S)
        if (j - 1 > i) return j - 1 - i;
        return j - i;                                                       // \s+
    }
    return 1;
}

static int match_llama3(const uint32_t *c, int n, int i, int max_digits) {
    int k = contraction(c, n, i);
    if (k) return k;
    if (isPre(c[i]) && i + 1 < n && isL(c[i + 1])) return 1 + run(c, n, i + 1, isL);
    if (isL(c[i])) return run(c, n, i, isL);
    return tail_alts(c, n, i, max_digits, false);
}

static int b1(const uint32_t *c, int n, int s) { // [U]*[Lo]+ starting at s, end or -1
    int amax = s + run(c, n, s, isUp);
    for (int a = amax; a >= s; a--) {
        if (a < n && isLo(c[a])) { int b = a + run(c, n, a, isLo); return b + contraction(c, n, b); }
    }
    return -1;
}
static int b2(const uint32_t *c, int n, int s) { // [U]+[Lo]*
    if (s >= n || !isUp(c[s])) return -1;
    int a = s + run(c, n, s, isUp);
    int b = a + run(c, n, a, isLo);
    return b + contraction(c, n, b);
}

static int match_llama4(const uint32_t *c, int n, int i) {
    int e;
    if (isPre(c[i]) && i + 1 < n && (e = b1(c, n, i + 1)) > 0) return e - i;
    if ((e = b1(c, n, i)) > 0) return e - i;
    if (isPre(c[i]) && i + 1 < n && (e = b2(c, n, i + 1)) > 0) return e - i;
    if ((e = b2(c, n, i)) > 0) return e - i;
    return tail_alts(c, n, i, 3, true);
}

static void pretokenize(const tokenizer *t, const char *s, size_t len, ivec *out) {
    if (!len) return;
    uint32_t *cp = malloc(sizeof(uint32_t) * (len + 1));
    size_t *off = malloc(sizeof(size_t) * (len + 1));
    int n = 0;
    for (size_t i = 0; i < len;) { off[n] = i; i += utf8_decode((const unsigned char *)s + i, len - i, &cp[n]); n++; }
    off[n] = len;
    for (int i = 0; i < n;) {
        int k = t->pre == PRE_LLAMA4 ? match_llama4(cp, n, i) : match_llama3(cp, n, i, t->pre == PRE_QWEN2 ? 1 : 3);
        if (k <= 0) k = 1;
        bpe(t, s + off[i], off[i + k] - off[i], out);
        i += k;
    }
    free(cp);
    free(off);
}

int tok_encode(const tokenizer *t, const char *text, size_t len, int32_t **out) {
    ivec o = {0};
    size_t start = 0;
    for (size_t i = 0; i < len; i++) {
        if (!t->special_first[(unsigned char)text[i]]) continue;
        for (int k = 0; k < t->n_specials; k++) {
            const str *sp = &t->tokens[t->specials[k]];
            if (sp->len <= len - i && memcmp(text + i, sp->s, sp->len) == 0) {
                pretokenize(t, text + start, i - start, &o);
                push(&o, t->specials[k]);
                i += sp->len - 1;
                start = i + 1;
                break;
            }
        }
    }
    pretokenize(t, text + start, len - start, &o);
    *out = o.v;
    return o.n;
}

size_t tok_piece(const tokenizer *t, int32_t id, char *buf, size_t cap) {
    if (id < 0 || id >= t->n_vocab) return 0;
    const str *s = &t->tokens[id];
    size_t w = 0;
    if (t->types[id] == TT_CONTROL || t->types[id] == TT_USER) {
        size_t n = s->len < cap ? s->len : cap;
        memcpy(buf, s->s, n);
        return n;
    }
    for (size_t i = 0; i < s->len && w < cap;) {
        uint32_t c;
        i += utf8_decode((const unsigned char *)s->s + i, s->len - i, &c);
        if (c < 512 && t->cp_to_byte[c] >= 0) buf[w++] = (char)t->cp_to_byte[c];
        else { char tmp[4]; int k = utf8_encode(c, tmp); for (int j = 0; j < k && w < cap; j++) buf[w++] = tmp[j]; }
    }
    return w;
}

char *tok_decode(const tokenizer *t, const int32_t *ids, int n) {
    size_t cap = 64 + (size_t)n * 16, w = 0;
    char *out = malloc(cap);
    for (int i = 0; i < n; i++) {
        if (w + 512 > cap) { cap *= 2; out = realloc(out, cap); }
        w += tok_piece(t, ids[i], out + w, cap - w - 1);
    }
    out[w] = 0;
    return out;
}
