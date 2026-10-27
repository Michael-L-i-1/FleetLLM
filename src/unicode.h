// Unicode category flags for the BPE pre-tokenizers, plus UTF-8 helpers.
#pragma once
#include <stddef.h>
#include <stdint.h>

enum {
    U_LU = 1, U_LL = 2, U_LT = 4, U_LM = 8, U_LO = 16, U_M = 32, U_N = 64, U_SPACE = 128,
    U_L = U_LU | U_LL | U_LT | U_LM | U_LO,
};

typedef struct { uint32_t lo, hi; uint8_t flags; } unicode_range;
extern const unicode_range unicode_ranges[];
extern const int unicode_n_ranges;

static inline uint8_t ucat(uint32_t cp) {
    if (cp < 0x80) { // fast path
        if (cp >= 'a' && cp <= 'z') return U_LL;
        if (cp >= 'A' && cp <= 'Z') return U_LU;
        if (cp >= '0' && cp <= '9') return U_N;
        if (cp == ' ' || (cp >= 9 && cp <= 13)) return U_SPACE;
        return 0;
    }
    int lo = 0, hi = unicode_n_ranges - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < unicode_ranges[mid].lo) hi = mid - 1;
        else if (cp > unicode_ranges[mid].hi) lo = mid + 1;
        else return unicode_ranges[mid].flags;
    }
    return 0;
}

// Decode one UTF-8 codepoint at s (len bytes left); returns bytes consumed (invalid -> 1, U+FFFD).
static inline int utf8_decode(const unsigned char *s, size_t len, uint32_t *cp) {
    unsigned c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    int n = (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (!n || (size_t)n > len) { *cp = 0xFFFD; return 1; }
    uint32_t v = c & (0x7F >> n);
    for (int i = 1; i < n; i++) {
        if ((s[i] & 0xC0) != 0x80) { *cp = 0xFFFD; return 1; }
        v = (v << 6) | (s[i] & 0x3F);
    }
    *cp = v;
    return n;
}

static inline int utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}
