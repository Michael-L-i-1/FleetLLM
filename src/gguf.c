#include "gguf.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { const uint8_t *p, *end; } rd;

static bool need(rd *r, size_t n) { return (size_t)(r->end - r->p) >= n; }
static uint32_t rd_u32(rd *r) { uint32_t v; memcpy(&v, r->p, 4); r->p += 4; return v; }
static uint64_t rd_u64(rd *r) { uint64_t v; memcpy(&v, r->p, 8); r->p += 8; return v; }

static size_t scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
    case GGUF_U16: case GGUF_I16: return 2;
    case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
    case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
    default: return 0;
    }
}

static void read_scalar(const uint8_t *p, uint32_t t, gguf_kv *kv) {
    switch (t) {
    case GGUF_U8: kv->num.u = *p; break;
    case GGUF_I8: kv->num.i = (int8_t)*p; kv->num.u = (uint64_t)kv->num.i; break;
    case GGUF_BOOL: kv->num.u = *p != 0; break;
    case GGUF_U16: { uint16_t v; memcpy(&v, p, 2); kv->num.u = v; } break;
    case GGUF_I16: { int16_t v; memcpy(&v, p, 2); kv->num.i = v; } break;
    case GGUF_U32: { uint32_t v; memcpy(&v, p, 4); kv->num.u = v; } break;
    case GGUF_I32: { int32_t v; memcpy(&v, p, 4); kv->num.i = v; } break;
    case GGUF_F32: { float v; memcpy(&v, p, 4); kv->num.f = v; } break;
    case GGUF_U64: { uint64_t v; memcpy(&v, p, 8); kv->num.u = v; } break;
    case GGUF_I64: { int64_t v; memcpy(&v, p, 8); kv->num.i = v; } break;
    case GGUF_F64: { double v; memcpy(&v, p, 8); kv->num.f = v; } break;
    }
}

// Skip one value of type t (used to step over array elements).
static bool skip_value(rd *r, uint32_t t) {
    if (t == GGUF_STR) {
        if (!need(r, 8)) return false;
        uint64_t n = rd_u64(r);
        if (!need(r, n)) return false;
        r->p += n;
        return true;
    }
    if (t == GGUF_ARR) {
        if (!need(r, 12)) return false;
        uint32_t et = rd_u32(r);
        uint64_t n = rd_u64(r);
        size_t es = scalar_size(et);
        if (es) { if (!need(r, n * es)) return false; r->p += n * es; return true; }
        for (uint64_t i = 0; i < n; i++) if (!skip_value(r, et)) return false;
        return true;
    }
    size_t s = scalar_size(t);
    if (!s || !need(r, s)) return false;
    r->p += s;
    return true;
}

static char *dup_str(rd *r) {
    uint64_t n = rd_u64(r);
    char *s = malloc(n + 1);
    memcpy(s, r->p, n);
    s[n] = 0;
    r->p += n;
    return s;
}

gguf_ctx *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return NULL; }
    struct stat st;
    fstat(fd, &st);
    const uint8_t *map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return NULL; }
    rd r = { map, map + st.st_size };
    if (!need(&r, 24) || memcmp(r.p, "GGUF", 4) != 0) { fprintf(stderr, "%s: not a GGUF file\n", path); goto fail; }
    r.p += 4;
    gguf_ctx *g = calloc(1, sizeof *g);
    g->map = map; g->size = st.st_size; g->fd = fd;
    g->version = rd_u32(&r);
    if (g->version < 2) { fprintf(stderr, "GGUF v%u unsupported\n", g->version); free(g); goto fail; }
    g->n_tensors = rd_u64(&r);
    g->n_kv = rd_u64(&r);
    g->kv = calloc(g->n_kv, sizeof *g->kv);
    for (uint64_t i = 0; i < g->n_kv; i++) {
        gguf_kv *kv = &g->kv[i];
        kv->key = dup_str(&r);
        kv->type = rd_u32(&r);
        if (kv->type == GGUF_STR) {
            kv->n = rd_u64(&r); kv->data = r.p; r.p += kv->n;
        } else if (kv->type == GGUF_ARR) {
            kv->arr_type = rd_u32(&r); kv->n = rd_u64(&r); kv->data = r.p;
            size_t es = scalar_size(kv->arr_type);
            if (es) r.p += kv->n * es;
            else for (uint64_t j = 0; j < kv->n; j++) skip_value(&r, kv->arr_type);
        } else {
            read_scalar(r.p, kv->type, kv);
            r.p += scalar_size(kv->type);
        }
    }
    uint64_t align = 32, a;
    if (gguf_u64(g, "general.alignment", &a)) align = a;
    g->tensors = calloc(g->n_tensors, sizeof *g->tensors);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        t->name = dup_str(&r);
        t->n_dims = rd_u32(&r);
        uint64_t ne = 1;
        for (uint32_t d = 0; d < 4; d++) t->dims[d] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) { t->dims[d] = rd_u64(&r); ne *= t->dims[d]; }
        t->type = rd_u32(&r);
        uint64_t off = rd_u64(&r);
        size_t be, bb;
        if (!ggml_block(t->type, &be, &bb)) { be = 1; bb = 0; } // unknown types: zero size, rejected on use
        t->nbytes = ne / be * bb;
        t->data = (const uint8_t *)(uintptr_t)off; // fixed up below
    }
    size_t data_start = ((size_t)(r.p - map) + align - 1) / align * align;
    for (uint64_t i = 0; i < g->n_tensors; i++) g->tensors[i].data = map + data_start + (size_t)(uintptr_t)g->tensors[i].data;
    return g;
fail:
    munmap((void *)map, st.st_size);
    close(fd);
    return NULL;
}

void gguf_close(gguf_ctx *g) {
    if (!g) return;
    for (uint64_t i = 0; i < g->n_kv; i++) free((void *)g->kv[i].key);
    for (uint64_t i = 0; i < g->n_tensors; i++) free((void *)g->tensors[i].name);
    free(g->kv); free(g->tensors);
    munmap((void *)g->map, g->size);
    close(g->fd);
    free(g);
}

const gguf_kv *gguf_find(const gguf_ctx *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++) if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    return NULL;
}

bool gguf_u64(const gguf_ctx *g, const char *key, uint64_t *out) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv || kv->type == GGUF_STR || kv->type == GGUF_ARR) return false;
    *out = (kv->type == GGUF_F32 || kv->type == GGUF_F64) ? (uint64_t)kv->num.f : kv->num.u;
    return true;
}

bool gguf_f32(const gguf_ctx *g, const char *key, float *out) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv || kv->type == GGUF_STR || kv->type == GGUF_ARR) return false;
    if (kv->type == GGUF_F32 || kv->type == GGUF_F64) *out = (float)kv->num.f;
    else if (kv->type == GGUF_I8 || kv->type == GGUF_I16 || kv->type == GGUF_I32 || kv->type == GGUF_I64) *out = (float)kv->num.i;
    else *out = (float)kv->num.u;
    return true;
}

const char *gguf_str(const gguf_ctx *g, const char *key, size_t *len) {
    const gguf_kv *kv = gguf_find(g, key);
    if (!kv || kv->type != GGUF_STR) return NULL;
    if (len) *len = kv->n;
    return (const char *)kv->data;
}

const gguf_tensor *gguf_tensor_find(const gguf_ctx *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

const char *gguf_arr_next_str(const uint8_t **pos, size_t *len) {
    uint64_t n;
    memcpy(&n, *pos, 8);
    const char *s = (const char *)(*pos + 8);
    *pos += 8 + n;
    *len = n;
    return s;
}

uint64_t gguf_arr_u64(const gguf_kv *kv, uint64_t i) {
    size_t es = scalar_size(kv->arr_type);
    gguf_kv tmp = {0};
    read_scalar(kv->data + i * es, kv->arr_type, &tmp);
    if (kv->arr_type == GGUF_F32 || kv->arr_type == GGUF_F64) return (uint64_t)tmp.num.f;
    return tmp.num.u;
}

bool ggml_block(uint32_t type, size_t *elems, size_t *bytes) {
    size_t e, b;
    switch (type) {
    case GGML_F32: e = 1; b = 4; break;
    case GGML_F16: case GGML_BF16: e = 1; b = 2; break;
    case GGML_Q4_0: e = 32; b = 18; break;
    case GGML_Q4_1: e = 32; b = 20; break;
    case GGML_Q5_0: e = 32; b = 22; break;
    case GGML_Q5_1: e = 32; b = 24; break;
    case GGML_Q8_0: e = 32; b = 34; break;
    case GGML_Q2_K: e = 256; b = 84; break;
    case GGML_Q3_K: e = 256; b = 110; break;
    case GGML_Q4_K: e = 256; b = 144; break;
    case GGML_Q5_K: e = 256; b = 176; break;
    case GGML_Q6_K: e = 256; b = 210; break;
    default: return false;
    }
    *elems = e; *bytes = b;
    return true;
}

const char *ggml_type_name(uint32_t type) {
    switch (type) {
    case GGML_F32: return "F32"; case GGML_F16: return "F16"; case GGML_BF16: return "BF16";
    case GGML_Q4_0: return "Q4_0"; case GGML_Q4_1: return "Q4_1"; case GGML_Q5_0: return "Q5_0";
    case GGML_Q5_1: return "Q5_1"; case GGML_Q8_0: return "Q8_0"; case GGML_Q2_K: return "Q2_K";
    case GGML_Q3_K: return "Q3_K"; case GGML_Q4_K: return "Q4_K"; case GGML_Q5_K: return "Q5_K";
    case GGML_Q6_K: return "Q6_K";
    default: return "?";
    }
}
