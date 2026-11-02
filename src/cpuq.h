// Quantized weights and matmul for the CPU backend.
//
// At load, every weight matrix is repacked into one of two block formats with
// 32 values per block, so a single pair of kernels serves every source format:
//   CQ4: 4-bit codes, value = scale * code + bias (f16 scale and bias per block).
//        Exact for MLX affine 4-bit, GGUF Q4_0, Q4_1 and Q4_K (up to f16
//        rounding of Q4_K's derived block scales).
//   CQ8: int8 codes, value = scale * code (f16 scale per block). Exact for Q8_0;
//        any other format (Q5/Q6_K, F16, ...) is requantized to it (8-bit, so
//        near-lossless).
// Activations are quantized per row to int8 in the same 32-value blocks, and
// block dot products run as int8 dot-product instructions where the CPU has
// them (ARMv8.2 dotprod: Pi 5, Apple), with a portable scalar fallback.
#pragma once
#include "model.h"
#include <stdint.h>

#define CQ_BLOCK 32

enum { CQ4 = 4, CQ8 = 8 };

typedef struct {
    int kind, rows, cols; // cols % CQ_BLOCK == 0
    // Blocks are stored by groups of 4 rows (rows padded to a multiple of 4):
    // [row/4][block][row%4], so one load gets a block's 4 scales.
    uint8_t *codes;       // CQ4: 16 bytes per block (byte j = code[j] | code[j+16] << 4); CQ8: 32 int8 per block
    uint16_t *scale, *bias; // f16 per block; bias CQ4 only
} cq_mat;

// Repack w (any supported qmat format); NULL if cols is not a multiple of CQ_BLOCK.
cq_mat *cq_from_qmat(const qmat *w);
void cq_free(cq_mat *m);
size_t cq_bytes(const cq_mat *m);

// int8 activations: n rows of cols values, per-block scale and scale * sum(codes)
typedef struct { int n, cols; int8_t *q; float *s, *sq; size_t cap; const float *src; } cq_act;
void cq_quantize(cq_act *a, const float *x, int n, int cols);

// y[t * ldy + r] = W[r] . x[t] for rows [r0, r1) (r0 a multiple of 4) and all n activation rows.
void cq_matmul_rows(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1);

// Instruction set the kernels were built for ("neon+dotprod" or "scalar").
const char *cq_isa(void);
