#include "safetensors.h"
#include "json.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int dtype_of(const char *s) {
    static const char *names[] = { "F32", "F16", "BF16", "U32", "I32", "U8", "I8", "I64" };
    for (int i = 0; i < ST_OTHER; i++) if (!strcmp(s, names[i])) return i;
    return ST_OTHER;
}

const char *st_dtype_name(int dtype) {
    static const char *names[] = { "F32", "F16", "BF16", "U32", "I32", "U8", "I8", "I64", "?" };
    return names[dtype >= 0 && dtype <= ST_OTHER ? dtype : ST_OTHER];
}

static bool open_shard(st_ctx *s, const char *path) {
    if (s->n_files == 64) { fprintf(stderr, "safetensors: too many shards\n"); return false; }
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return false; }
    struct stat sb;
    fstat(fd, &sb);
    const uint8_t *map = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return false; }
    s->files[s->n_files++] = (st_file){ map, (size_t)sb.st_size, fd };
    uint64_t hlen;
    memcpy(&hlen, map, 8);
    if (8 + hlen > (uint64_t)sb.st_size) { fprintf(stderr, "%s: bad header\n", path); return false; }
    json *j = json_parse((const char *)map + 8, hlen);
    if (!j) return false;
    const uint8_t *base = map + 8 + hlen;
    for (const jnode *e = json_first(j, json_root(j)); e; e = json_next(j, e)) {
        if (e->key_len == 12 && !memcmp(e->key, "__metadata__", 12)) continue;
        const char *dt = json_str(j, e, "dtype");
        const jnode *shape = json_get(j, e, "shape"), *off = json_get(j, e, "data_offsets");
        if (!dt || !shape || !off || off->n != 2) continue;
        if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 1024; s->t = realloc(s->t, sizeof(st_tensor) * s->cap); }
        st_tensor *t = &s->t[s->n++];
        memset(t, 0, sizeof *t);
        t->name = strndup(e->key, e->key_len);
        t->dtype = dtype_of(dt);
        t->n_dims = shape->n < 4 ? shape->n : 4;
        for (int i = 0; i < t->n_dims; i++) t->shape[i] = (int64_t)json_child(j, shape, i)->num;
        size_t a = (size_t)json_child(j, off, 0)->num, b = (size_t)json_child(j, off, 1)->num;
        t->data = base + a;
        t->nbytes = b - a;
    }
    json_free(j);
    return true;
}

st_ctx *st_open_dir(const char *dir) {
    st_ctx *s = calloc(1, sizeof *s);
    char path[4096];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    json *idx = json_load_file(path);
    if (idx) {
        // distinct shard names, in order of first appearance
        const jnode *wm = json_get(idx, json_root(idx), "weight_map");
        char *seen[64];
        int n_seen = 0;
        for (const jnode *e = json_first(idx, wm); e; e = json_next(idx, e)) {
            bool dup = false;
            for (int i = 0; i < n_seen; i++) if (!strcmp(seen[i], e->s)) dup = true;
            if (dup || n_seen == 64) continue;
            seen[n_seen++] = (char *)e->s;
            snprintf(path, sizeof path, "%s/%s", dir, e->s);
            if (!open_shard(s, path)) { json_free(idx); st_close(s); return NULL; }
        }
        json_free(idx);
    } else {
        snprintf(path, sizeof path, "%s/model.safetensors", dir);
        if (!open_shard(s, path)) { st_close(s); return NULL; }
    }
    return s;
}

void st_close(st_ctx *s) {
    if (!s) return;
    for (int i = 0; i < s->n_files; i++) { munmap((void *)s->files[i].map, s->files[i].size); close(s->files[i].fd); }
    for (int i = 0; i < s->n; i++) free(s->t[i].name);
    free(s->t);
    free(s);
}

const st_tensor *st_find(const st_ctx *s, const char *name) {
    for (int i = 0; i < s->n; i++) if (!strcmp(s->t[i].name, name)) return &s->t[i];
    return NULL;
}
