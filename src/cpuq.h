// Quantized weights and matmul for the CPU backend.
//
// At load, every weight matrix is repacked into a block format with 32 values
// per block that keeps the source's bits (so decode reads no more bytes than
// the file holds):
//   CQ4   4-bit codes, value = scale * code + bias (f16 per block). MLX affine
//         4-bit, GGUF Q4_1.                                             5.0 bits
//   CQ4S  4-bit codes, value = scale * (code - 8) (f16 per block). Q4_0. 4.5 bits
//   CQ4K  4-bit codes, value = d * sc * code - dmin * mn: 6-bit sc, mn per
//         block, f16 d, dmin per 256 values. Q4_K.                    4.6 bits
//   CQ6K  6-bit codes - 32, value = d * sc * code: int8 sc per 16 values, f16 d
//         per 256. Q6_K.                                              6.6 bits
//   CQ8   int8 codes, value = scale * code (f16 per block). Q8_0 exactly; any
//         other format (Q5, Q2/3/5_K, F16, ...) is requantized to it.  8.5 bits
// Activations are quantized per row to int8 in the same 32-value blocks, and
// block dot products run as int8 dot-product instructions where the CPU has
// them (ARMv8.2 dotprod: Pi 5, Apple), with a portable scalar fallback.
#pragma once
#include "model.h"
#include <stdint.h>

#define CQ_BLOCK 32

enum { CQ4, CQ4S, CQ4K, CQ6K, CQ8 };

// Blocks are stored by groups of 4 rows (rows padded to a multiple of 4), the
// 4 rows of a block adjacent, so one load gets a block's 4 scales:
// [row / 4][block][row % 4].
typedef struct {
    int kind, rows, cols; // cols % CQ_BLOCK == 0 (% 256 for CQ4K, CQ6K)
    uint8_t *codes;       // 4-bit kinds: 16 bytes per block (byte j = code[j] | code[j + 16] << 4)
                          // CQ6K: those 16 bytes of low 4 bits, then 8 bytes of high 2 bits
                          //       (byte j: bits of codes j, j + 8, j + 16, j + 24 from the bottom)
                          // CQ8: 32 int8
    uint16_t *scale, *bias; // CQ4, CQ4S, CQ8: f16 scale per block; CQ4: f16 bias per block
    uint8_t *sub;         // CQ4K: [sc x 4 rows, mn x 4 rows] per block group;
                          // CQ6K: [int8 sc of values 0-15 x 4, of 16-31 x 4]
    uint16_t *super;      // CQ4K, CQ6K: [d x 4 rows, dmin x 4 rows] f16 per 256 values
} cq_mat;

// Repack w (any qmat format); NULL if cols is not a multiple of CQ_BLOCK.
cq_mat *cq_from_qmat(const qmat *w);
void cq_free(cq_mat *m);
size_t cq_bytes(const cq_mat *m);
const char *cq_kind_name(int kind);
// Largest |value - reference| over the first rows of w, relative to the largest
// |reference| (0 for an exact repack; ~1/254 for requantized CQ8).
float cq_check(const qmat *w, const cq_mat *m, int rows);

// int8 activations: n rows of cols values, per-block scale and scale * sum(codes)
typedef struct { int n, cols; int8_t *q; float *s, *sq; size_t cap; const float *src; } cq_act;
void cq_quantize(cq_act *a, const float *x, int n, int cols);

// y[t * ldy + r] = W[r] . x[t] for rows [r0, r1) (r0 a multiple of 4) and all n activation rows.
void cq_matmul_rows(const cq_mat *w, const cq_act *a, float *y, int ldy, int r0, int r1);

// Instruction set the kernels were built for ("neon+dotprod" or "scalar").
const char *cq_isa(void);
