// safetensors reader for (possibly sharded) Hugging Face / MLX checkpoints.
// Every shard is memory-mapped; tensor data is never copied.
#pragma once
#include <stddef.h>
#include <stdint.h>

enum st_dtype { ST_F32, ST_F16, ST_BF16, ST_U32, ST_I32, ST_U8, ST_I8, ST_I64, ST_OTHER };

typedef struct {
    char *name;
    int dtype, n_dims;
    int64_t shape[4]; // shape[n_dims-1] is the contiguous dimension
    const uint8_t *data;
    size_t nbytes;
} st_tensor;

typedef struct { const uint8_t *map; size_t size; int fd; } st_file;

typedef struct {
    st_file files[64];
    int n_files;
    st_tensor *t;
    int n, cap;
} st_ctx;

// Opens dir/model.safetensors or every shard listed in dir/model.safetensors.index.json.
st_ctx *st_open_dir(const char *dir);
void st_close(st_ctx *s);
const st_tensor *st_find(const st_ctx *s, const char *name);
const char *st_dtype_name(int dtype);
