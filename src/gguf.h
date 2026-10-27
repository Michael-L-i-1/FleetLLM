// GGUF v2/v3 reader. The file is memory-mapped; tensor data is never copied.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ggml tensor types we know about (values match ggml's enum)
enum ggml_type {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7,
    GGML_Q8_0 = 8, GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13,
    GGML_Q6_K = 14, GGML_BF16 = 30,
};

// GGUF metadata value types
enum gguf_vtype {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32, GGUF_BOOL,
    GGUF_STR, GGUF_ARR, GGUF_U64, GGUF_I64, GGUF_F64,
};

typedef struct {
    const char *key; // NUL-terminated copy
    uint32_t type;
    union { uint64_t u; int64_t i; double f; } num; // scalars (bool/ints/floats)
    const uint8_t *data; // STR: bytes; ARR: first element
    uint64_t n;          // STR: length; ARR: element count
    uint32_t arr_type;   // ARR element type
} gguf_kv;

typedef struct {
    const char *name;
    uint32_t n_dims;
    uint64_t dims[4]; // dims[0] is the contiguous (row) dimension
    uint32_t type;
    const uint8_t *data;
    size_t nbytes;
} gguf_tensor;

typedef struct {
    const uint8_t *map;
    size_t size;
    int fd;
    uint32_t version;
    gguf_kv *kv;
    uint64_t n_kv;
    gguf_tensor *tensors;
    uint64_t n_tensors;
} gguf_ctx;

gguf_ctx *gguf_open(const char *path);
void gguf_close(gguf_ctx *g);

const gguf_kv *gguf_find(const gguf_ctx *g, const char *key);
bool gguf_u64(const gguf_ctx *g, const char *key, uint64_t *out);
bool gguf_f32(const gguf_ctx *g, const char *key, float *out);
// Returns a pointer into the mapping and its length (not NUL-terminated).
const char *gguf_str(const gguf_ctx *g, const char *key, size_t *len);
const gguf_tensor *gguf_tensor_find(const gguf_ctx *g, const char *name);

// Iterate a string array: call with *pos = kv->data initially.
const char *gguf_arr_next_str(const uint8_t **pos, size_t *len);
// Numeric array element i as u64 / f64.
uint64_t gguf_arr_u64(const gguf_kv *kv, uint64_t i);

// (elements per block, bytes per block) for a ggml type; returns false if unknown.
bool ggml_block(uint32_t type, size_t *elems, size_t *bytes);
const char *ggml_type_name(uint32_t type);
