// Metal backend (Apple GPU). Each host region holding weights (the mmap'd GGUF,
// or every safetensors shard plus the converted norm vectors) is wrapped as a
// zero-copy MTLBuffer and tensors are addressed by offset; KV pages are f16 in
// shared buffers so the engine's copy-on-write bookkeeping stays on the CPU.
// One command buffer per forward step.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "backend.h"
#include "engine.h"
#include "kernels_metal.h"
#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MV_MAX_ROWS 8            // fused mat-vec up to here
#define FUSED_MAX_ROWS 64        // fused tile-dequant GEMM up to here; above: dequantize once + dense GEMM
#define SCRATCH_BYTES (256u << 20) // f16 dequant scratch; bigger matrices go in row chunks
#define ATTN_SPLIT 256           // max positions per flash-decoding split
// Positions per split for a row attending to ctx positions: ~48 splits (enough
// parallelism for a single decoding row), a multiple of the 32-position tile in
// 32..ATTN_SPLIT. It depends only on the row itself, so a row's result does not
// depend on what else is in the batch.
static uint32_t attn_split(uint32_t ctx) {
    uint32_t sz = (ctx + 47) / 48;
    sz = (sz + 31) / 32 * 32;
    return sz < 32 ? 32 : sz > ATTN_SPLIT ? ATTN_SPLIT : sz;
}

typedef struct { uint32_t K, M, N, ldy, row_bytes, x_stride; } MvArgs;
typedef struct { uint32_t K, rows, row_bytes, r0; } DqArgs;
typedef struct { uint32_t K, rows, N, ldy, r0, x_stride; } GemmArgs;
typedef struct { uint32_t d; float eps; uint32_t has_w, add, gather, half_out; } NormArgs;
typedef struct { uint32_t nh, nkv, hd; float eps; uint32_t has_qn, has_kn, rope, neox, qs, ks; } QkArgs;
typedef struct { uint32_t nkv, hd, page, ks; uint64_t layer_off; uint32_t vs, layer; } KvwArgs;
typedef struct { uint32_t nh, nkv, hd; float scale; uint64_t page_elems, layer_off; uint32_t page, gate, qs, gs, half_out, layer; uint64_t scale_off; } AttnArgs;
typedef struct { uint32_t ctx_end, start, tab, split, part; } RowInfo; // part: partials slot, NAX_ROW for attn_nax rows
#define NAX_ROW 0xffffffffu

// Group consecutive rows of one sequence (same split size and origin) for the
// attention kernel: up to 32 / G rows (8 at most) share every K/V tile read.
// Rows with skip[i] set belong to other kernels. Writes {first row, count}
// pairs; returns the group count, the largest group through *rmax, and raises
// *max_sp to the largest split count.
// Grouped kernel: the matrix-unit flash kernel by default; CINFER_ATTN_SCALAR=1
// selects the ALU kernel.
// Rows that are not input (decode, verification) run attn_vec, one row per
// threadgroup; CINFER_ATTN_VEC=0 groups them into the matrix kernel instead.
static bool attn_vec_on(void) { static int v = -1; if (v < 0) v = !getenv("CINFER_ATTN_VEC") || atoi(getenv("CINFER_ATTN_VEC")); return v; }

// Input rows (prompt / appended text) run attention on the neural accelerators
// (attn_nax), decode and verify rows the split kernels; CINFER_ATTN_NAX=0 keeps
// every row on the split kernels. attn_nax splits contexts into fixed spans of
// NAX_SPAN positions (CINFER_NAX_SPAN), merged by attn_reduce.
#define NAX_SPAN 8192
static bool attn_nax_on(void) { static int v = -1; if (v < 0) v = !getenv("CINFER_ATTN_NAX") || atoi(getenv("CINFER_ATTN_NAX")); return v; }
static uint32_t nax_span(void) {
    static int v = -1;
    if (v < 0) { v = getenv("CINFER_NAX_SPAN") ? atoi(getenv("CINFER_NAX_SPAN")) / 32 * 32 : NAX_SPAN; if (v < 32) v = 32; }
    return (uint32_t)v;
}

static bool attn_mma_on(void) { static int v = -1; if (v < 0) v = getenv("CINFER_ATTN_SCALAR") == NULL; return v; }

static int attn_groups(const RowInfo *ri, const int *seq, const int *pos, const uint8_t *skip, int n, int G, uint32_t *grp, int *rmax, uint32_t *max_sp) {
    int R = 32 / G < 8 ? 32 / G : 8, ng = 0;
    if (R < 1) R = 1;
    *rmax = 1;
    for (int i = 0; i < n; i++) {
        if (skip && skip[i]) continue;
        uint32_t S = ri[i].split, o = ri[i].start / S * S, ns = (ri[i].ctx_end + 1 - o + S - 1) / S;
        if (ns > *max_sp) *max_sp = ns;
        bool join = ng && i > 0 && !(skip && skip[i - 1]) && seq[i] == seq[i - 1] && pos[i] == pos[i - 1] + 1 && grp[2 * (ng - 1) + 1] < (uint32_t)R && S == ri[i - 1].split &&
                    o == ri[i - 1].start / ri[i - 1].split * ri[i - 1].split;
        if (join) {
            if ((int)++grp[2 * (ng - 1) + 1] > *rmax) *rmax = (int)grp[2 * (ng - 1) + 1];
        } else {
            grp[2 * ng] = (uint32_t)i;
            grp[2 * ng + 1] = 1;
            ng++;
        }
    }
    return ng;
}

typedef struct { float scale, cap; } CapArgs;
typedef struct { uint32_t d; float eps_post, eps; uint32_t post, half_out; } AddNormArgs;

@interface CIMetal : NSObject {
@public
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pso;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *bufs;
    id<MTLBuffer> wbuf[66];
    const uint8_t *wbase[66];
    size_t wlen[66];
    int n_w;
    bool nax; // affine GEMM on the neural accelerators (Metal 4)
    // per layer: fused [q|k|v|gate] and [ffn gate|up] tiled matrices (type 0 when not fused)
    qmat *f_qkvg, *f_gu;
    qmat *d_kv, *d_gu; // drafter: fused [k|v] and [gate|up] per layer
    bool h16; // every matmul is tiled: matmul inputs are produced directly as f16
    id<MTLBuffer> invf, d_invf;
}
@end
@implementation CIMetal
@end

static id<MTLComputePipelineState> P(CIMetal *mt, NSString *name) {
    id<MTLComputePipelineState> p = mt->pso[name];
    if (!p) { fprintf(stderr, "metal: no kernel %s\n", name.UTF8String); exit(1); }
    return p;
}

// Shared scratch buffer that grows to at least `bytes`.
static id<MTLBuffer> B(CIMetal *mt, NSString *name, size_t bytes) {
    id<MTLBuffer> b = mt->bufs[name];
    if (!b || b.length < bytes) {
        size_t n = 256;
        while (n < bytes) n <<= 1;
        b = [mt->dev newBufferWithLength:n options:MTLResourceStorageModeShared];
        if (!b) { fprintf(stderr, "metal: allocation of %zu bytes failed\n", n); exit(1); }
        mt->bufs[name] = b;
    }
    return b;
}

// Bind host pointer p (inside one of the wrapped weight regions) at index idx.
static void set_w(id<MTLComputeCommandEncoder> e, CIMetal *mt, const void *p, int idx) {
    const uint8_t *q = p;
    for (int i = 0; i < mt->n_w; i++)
        if (q >= mt->wbase[i] && q < mt->wbase[i] + mt->wlen[i]) { [e setBuffer:mt->wbuf[i] offset:(NSUInteger)(q - mt->wbase[i]) atIndex:idx]; return; }
    if (!p) { [e setBuffer:mt->wbuf[0] offset:0 atIndex:idx]; return; } // unused optional argument
    fprintf(stderr, "metal: pointer %p is outside the weight regions\n", p);
    exit(1);
}

static const char *sname(const qmat *w) { return w->scales_f16 ? "f16" : "bf"; }

static const char *qname(uint32_t type) {
    switch (type) {
    case GGML_Q2_K: return "q2k"; case GGML_Q3_K: return "q3k"; case GGML_Q4_K: return "q4k";
    case GGML_Q5_K: return "q5k"; case GGML_Q6_K: return "q6k"; case GGML_Q8_0: return "q8_0";
    case GGML_Q4_0: return "q4_0";
    default: fprintf(stderr, "metal: weight type %s not supported\n", ggml_type_name(type)); exit(1);
    }
}

static void set_bytes(id<MTLComputeCommandEncoder> e, const void *p, size_t n, int idx) { [e setBytes:p length:n atIndex:idx]; }

// Largest batch that uses the fused mat-vec; CINFER_MV_MAX overrides it (e.g. a
// huge value makes every batch size compute rows identically, for exactness checks).
static int env_int(const char *name, int def) { const char *s = getenv(name); return s ? atoi(s) : def; }
static int mv_max(void) { static int v = -1; if (v < 0) v = env_int("CINFER_MV_MAX", MV_MAX_ROWS); return v; }
static int fused_max(void) { static int v = -1; if (v < 0) v = env_int("CINFER_FUSED_MAX", FUSED_MAX_ROWS); return v; }

typedef struct { uint32_t K, M, N, ldy, x_stride; } AmvArgs;
typedef struct { int32_t K, N, M, ldy, split, part_stride; } AtArgs;

// ---- tiled affine layout (see kernels_nax.metal: TiledBlock) ----
// 4-bit affine matrices are repacked at load into 64-row x 64-K blocks with
// their scales/biases inline, so each NAX GEMM K step is one contiguous read.
#define TILE 64
typedef struct { void *buf; int split, split_mv; } tiled_t; // buf: retained id<MTLBuffer>

static int tiled_gpt(const qmat *w) { return w->group >= TILE ? 1 : TILE / w->group; }
static size_t tiled_block_bytes(const qmat *w) { return (size_t)TILE * TILE * w->bits / 8 + (size_t)TILE * tiled_gpt(w) * 2 * 2; }
static bool tiled_ok(const qmat *w) {
    return w->type == QT_AFFINE && w->bits == 4 && w->cols % TILE == 0 && (w->group == 32 || w->group == 64 || w->group == 128);
}

typedef struct { const qmat *w; uint8_t *dst; } repack_job;
static void repack_rows(void *ctx, int rt0, int rt1) {
    const repack_job *j = ctx;
    const qmat *w = j->w;
    int n_kt = w->cols / TILE, gpt = tiled_gpt(w), ng = w->cols / w->group, cb = TILE * TILE * w->bits / 8, rbt = TILE * w->bits / 8;
    size_t bb = tiled_block_bytes(w);
    for (int rt = rt0; rt < rt1; rt++)
        for (int kt = 0; kt < n_kt; kt++) {
            uint8_t *blk = j->dst + ((size_t)rt * n_kt + kt) * bb;
            uint16_t *par = (uint16_t *)(blk + cb);
            for (int r = 0; r < TILE; r++) {
                int row = rt * TILE + r;
                if (row >= w->rows) {
                    memset(blk + r * rbt, 0, rbt);
                    memset(par + r * gpt * 2, 0, gpt * 4);
                    continue;
                }
                memcpy(blk + r * rbt, w->data + (size_t)row * w->row_bytes + (size_t)kt * rbt, rbt);
                for (int g = 0; g < gpt; g++) {
                    int gi = (kt * TILE + g * (w->group < TILE ? w->group : TILE)) / w->group;
                    memcpy(par + (r * gpt + g) * 2, w->scales + (size_t)row * ng + gi, 2);
                    memcpy(par + (r * gpt + g) * 2 + 1, w->biases + (size_t)row * ng + gi, 2);
                }
            }
        }
}

// K splits (measured on M5 Pro): GEMM so that ~2048 single-simdgroup
// threadgroups (2 per row tile) run with >= 4 K tiles per split; mat-vec so
// that ~8192 simdgroups run with >= 8 K tiles each. Both depend only on the
// weight shape.
static int tiled_split(const qmat *w, int units_per_tile, int target, int min_kt) {
    int units = (w->rows + TILE - 1) / TILE * units_per_tile, n_kt = w->cols / TILE, split = 1;
    while (units * split * 2 <= target && n_kt / (split * 2) >= min_kt) split *= 2;
    return split;
}
#define TMV_MAX 4 // tokens handled by the tiled mat-vec (3-4 via its f16-dot variant); more go to the NAX GEMM
static int tmv_nt(int n) { return n <= 2 ? n : 4; }
// CINFER_TMV_MAX overrides TMV_MAX (0: the NAX GEMM for every batch size, so a
// row's result is bit-identical whatever the batch size; used by bench --check)
static int tmv_max(void) { static int v = -1; if (v < 0) v = env_int("CINFER_TMV_MAX", TMV_MAX); return v; }

// untiled affine mat-vec over tokens [t0, t0+n), n <= 8
static void matmul_affine_mv(CIMetal *mt, id<MTLComputeCommandEncoder> e, const qmat *w, id<MTLBuffer> x, id<MTLBuffer> y, int ldy, int t0, int n) {
    int nt = n == 1 ? 1 : n == 2 ? 2 : n <= 4 ? 4 : 8;
    AmvArgs a = { (uint32_t)w->cols, (uint32_t)w->rows, (uint32_t)n, (uint32_t)ldy, (uint32_t)w->cols };
    [e setComputePipelineState:P(mt, [NSString stringWithFormat:@"amv_%s_%d_%d_%d", sname(w), w->group, w->bits, nt])];
    set_w(e, mt, w->data, 0);
    set_w(e, mt, w->scales, 1);
    set_w(e, mt, w->biases, 2);
    [e setBuffer:x offset:(NSUInteger)t0 * w->cols * 4 atIndex:3];
    [e setBuffer:y offset:(NSUInteger)t0 * ldy * 4 atIndex:4];
    set_bytes(e, &a, sizeof a, 5);
    [e dispatchThreadgroups:MTLSizeMake((w->rows + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

// Bytes for a matmul input of n rows: f32 rows, or f16 rows padded to a whole
// number of 64-row tiles (the NAX GEMM reads, but never stores, the pad rows).
static size_t act_bytes(int n, int cols) {
    size_t f32 = 4 * (size_t)n * cols, f16 = 2 * (size_t)((n + 63) / 64 * 64) * cols;
    return f32 > f16 ? f32 : f16;
}

// x_half: x already holds f16 rows padded to a multiple of 64 (see act_bytes).
// x_off / y_off: byte offsets into x / y (tiled path only).
static void matmul_affine(CIMetal *mt, id<MTLComputeCommandEncoder> e, const qmat *w, id<MTLBuffer> x, NSUInteger x_off, bool x_half, id<MTLBuffer> y,
                          NSUInteger y_off, int ldy, int n) {
    if (w->gpu) {
        const tiled_t *tw = w->gpu;
        int bm = n <= 16 ? 16 : 32, padded = (n + 63) / 64 * 64;
        id<MTLBuffer> xh = x;
        if (!x_half) { // activations as f16, padded to a whole number of 64-row tiles (pad rows are never stored)
            xh = B(mt, @"x16", 2 * (size_t)padded * w->cols);
            uint32_t cnt = (uint32_t)((size_t)n * w->cols);
            [e setComputePipelineState:P(mt, @"to_half")];
            [e setBuffer:x offset:x_off atIndex:0];
            [e setBuffer:xh offset:0 atIndex:1];
            set_bytes(e, &cnt, 4, 2);
            [e dispatchThreads:MTLSizeMake(cnt / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            x_off = 0;
        }
        int tiles = (w->rows + TILE - 1) / TILE, mv = n <= tmv_max(), split = mv ? tw->split_mv : tw->split;
        id<MTLBuffer> out = split > 1 ? B(mt, @"kpart", 4 * (size_t)split * n * w->rows) : y;
        AtArgs a = { w->cols, w->rows, n, split > 1 ? w->rows : ldy, split, n * w->rows };
        [e setBuffer:(__bridge id<MTLBuffer>)tw->buf offset:0 atIndex:0];
        [e setBuffer:xh offset:x_off atIndex:3];
        [e setBuffer:out offset:split > 1 ? 0 : y_off atIndex:4];
        if (mv) {
            [e setComputePipelineState:P(mt, [NSString stringWithFormat:@"tmv_%s_%d_%d", sname(w), w->group, tmv_nt(n)])];
            set_bytes(e, &a, sizeof a, 5);
            [e dispatchThreadgroups:MTLSizeMake(tiles * 4, 1, split) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            [e setComputePipelineState:P(mt, [NSString stringWithFormat:@"aqms_%s_%d_%d", sname(w), w->group, bm])];
            set_bytes(e, &a, sizeof a, 5);
            [e dispatchThreadgroups:MTLSizeMake(tiles * 2, (n + bm - 1) / bm, split) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        if (split > 1) {
            int32_t a4[4] = { w->rows, n, split, ldy };
            [e setComputePipelineState:P(mt, @"split_sum")];
            [e setBuffer:out offset:0 atIndex:0];
            [e setBuffer:y offset:y_off atIndex:1];
            set_bytes(e, a4, sizeof a4, 2);
            [e dispatchThreads:MTLSizeMake(w->rows, n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        return;
    }
    for (int t0 = 0; t0 < n; t0 += 8) // no NAX / untiled layout: mat-vec in groups of 8 tokens
        matmul_affine_mv(mt, e, w, x, y, ldy, t0, n - t0 < 8 ? n - t0 : 8);
}

// y[t*ldy + r] = W[r] . x[t] for n tokens; x is f16 when mt->h16 (all matmuls tiled)
static void matmul(CIMetal *mt, id<MTLComputeCommandEncoder> e, const qmat *w, id<MTLBuffer> x, id<MTLBuffer> y, int ldy, int n) {
    if (w->type == QT_AFFINE) { matmul_affine(mt, e, w, x, 0, mt->h16, y, 0, ldy, n); return; }
    const char *ty = qname(w->type);
    if (n <= mv_max()) {
        int nt = n == 1 ? 1 : n == 2 ? 2 : n <= 4 ? 4 : 8;
        MvArgs a = { (uint32_t)w->cols, (uint32_t)w->rows, (uint32_t)n, (uint32_t)ldy, (uint32_t)w->row_bytes, (uint32_t)w->cols };
        [e setComputePipelineState:P(mt, [NSString stringWithFormat:@"mv_%s_%d", ty, nt])];
        set_w(e, mt, w->data, 0);
        [e setBuffer:x offset:0 atIndex:1];
        [e setBuffer:y offset:0 atIndex:2];
        set_bytes(e, &a, sizeof a, 3);
        [e dispatchThreadgroups:MTLSizeMake((w->rows + 7) / 8, (n + nt - 1) / nt, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    if (n <= fused_max()) { // decode weight tiles into threadgroup memory, MMA on the tile
        int tn = n <= 16 ? 16 : 32;
        MvArgs a = { (uint32_t)w->cols, (uint32_t)w->rows, (uint32_t)n, (uint32_t)ldy, (uint32_t)w->row_bytes, (uint32_t)w->cols };
        [e setComputePipelineState:P(mt, [NSString stringWithFormat:@"mmf_%s_%d", ty, tn])];
        set_w(e, mt, w->data, 0);
        [e setBuffer:x offset:0 atIndex:1];
        [e setBuffer:y offset:0 atIndex:2];
        set_bytes(e, &a, sizeof a, 3);
        [e dispatchThreadgroups:MTLSizeMake((n + tn - 1) / tn, (w->rows + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    // large batches (prefill): dequantize once into f16 scratch (in row chunks), then dense GEMM
    id<MTLBuffer> scratch = B(mt, @"dq", SCRATCH_BYTES);
    int chunk = (int)(SCRATCH_BYTES / ((size_t)w->cols * 2)) / 64 * 64;
    if (chunk < 64) chunk = 64;
    for (int r0 = 0; r0 < w->rows; r0 += chunk) {
        int rows = w->rows - r0 < chunk ? w->rows - r0 : chunk;
        DqArgs d = { (uint32_t)w->cols, (uint32_t)rows, (uint32_t)w->row_bytes, (uint32_t)r0 };
        [e setComputePipelineState:P(mt, [NSString stringWithFormat:@"dq_%s", ty])];
        set_w(e, mt, w->data, 0);
        [e setBuffer:scratch offset:0 atIndex:1];
        set_bytes(e, &d, sizeof d, 2);
        [e dispatchThreads:MTLSizeMake(w->cols / 16, rows, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        GemmArgs g = { (uint32_t)w->cols, (uint32_t)rows, (uint32_t)n, (uint32_t)ldy, (uint32_t)r0, (uint32_t)w->cols };
        [e setComputePipelineState:P(mt, @"gemm_f16")];
        [e setBuffer:scratch offset:0 atIndex:0];
        [e setBuffer:x offset:0 atIndex:1];
        [e setBuffer:y offset:0 atIndex:2];
        set_bytes(e, &g, sizeof g, 3);
        [e dispatchThreadgroups:MTLSizeMake((n + 31) / 32, (rows + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
}

// add=false writes a matmul input (f16 when mt->h16); add=true accumulates into the f32 residual
static void rms(CIMetal *mt, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x, id<MTLBuffer> y, const float *w, int d, float eps, bool add, int rows, id<MTLBuffer> gather) {
    NormArgs a = { (uint32_t)d, eps, w != NULL, add, gather != nil, !add && mt->h16 };
    [e setComputePipelineState:P(mt, @"rms_norm")];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:y offset:0 atIndex:1];
    set_w(e, mt, w, 2);
    set_bytes(e, &a, sizeof a, 3);
    [e setBuffer:gather ? gather : x offset:0 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// x += wp ? rms(o) * wp : o;  h = rms(x) * wn  (one dispatch for the residual
// step plus the next block's pre-norm)
static bool add_norm_ok(int d) { return d % 4 == 0 && d <= 512 * 4 * 8; }
static void add_norm(CIMetal *mt, id<MTLComputeCommandEncoder> e, id<MTLBuffer> o, id<MTLBuffer> x, id<MTLBuffer> h, const float *wp, float eps_post,
                     const float *wn, float eps, int d, int rows) {
    AddNormArgs a = { (uint32_t)d, eps_post, eps, wp != NULL, mt->h16 };
    [e setComputePipelineState:P(mt, @"add_rms_norm")];
    [e setBuffer:o offset:0 atIndex:0];
    [e setBuffer:x offset:0 atIndex:1];
    [e setBuffer:h offset:0 atIndex:2];
    set_w(e, mt, wp, 3);
    set_w(e, mt, wn, 4);
    set_bytes(e, &a, sizeof a, 5);
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
}

// y[i] += b[i % w]; b is a weight-region pointer, or NULL to add buffer bb
static void add_bcast(CIMetal *mt, id<MTLComputeCommandEncoder> e, id<MTLBuffer> y, const float *b, id<MTLBuffer> bb, uint32_t w, size_t n) {
    [e setComputePipelineState:P(mt, @"add_bcast")];
    [e setBuffer:y offset:0 atIndex:0];
    if (b) set_w(e, mt, b, 1);
    else [e setBuffer:bb offset:0 atIndex:1];
    set_bytes(e, &w, sizeof w, 2);
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void argmax_dispatch(CIMetal *mt, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x, id<MTLBuffer> out, int vocab, int rows) {
    uint32_t v = (uint32_t)vocab;
    [e setComputePipelineState:P(mt, @"argmax_rows")];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:out offset:0 atIndex:1];
    set_bytes(e, &v, 4, 2);
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

static void *metal_alloc(backend *b, size_t bytes, void **handle) {
    CIMetal *mt = (__bridge CIMetal *)b->impl;
    id<MTLBuffer> buf = [mt->dev newBufferWithLength:(bytes ? bytes : 16) options:MTLResourceStorageModeShared];
    if (!buf) { fprintf(stderr, "metal: KV allocation of %zu bytes failed\n", bytes); exit(1); }
    *handle = (__bridge_retained void *)buf;
    return buf.contents;
}

static void write_buf(id<MTLBuffer> b, const void *src, size_t n) { memcpy(b.contents, src, n); }

static int metal_forward(backend *be, const model *m, kv_cache *kv, const seq_pages *const *pages, const row_t *rows, int n, float *logits) {
    @autoreleasepool {
        CIMetal *mt = (__bridge CIMetal *)be->impl;
        const model_cfg *c = &m->c;
        int d = c->d, hd = c->hd, qd = c->n_head * hd, kvd = c->n_kv * hd;
        size_t f = 4;
        // matmul inputs (h, att, act, hsel) are sized by act_bytes
        id<MTLBuffer> xb = B(mt, @"x", f * n * d), hb = B(mt, @"h", act_bytes(n, d)), qb = B(mt, @"q", f * n * qd),
                      kb = B(mt, @"k", f * n * kvd), vb = B(mt, @"v", f * n * kvd), gb = B(mt, @"g", f * n * qd),
                      ab = B(mt, @"att", act_bytes(n, qd)), ob = B(mt, @"o", f * n * d), fgb = B(mt, @"fg", f * n * c->ff),
                      fub = B(mt, @"fu", f * n * c->ff), actb = B(mt, @"act", act_bytes(n, c->ff));
        int qkvg_w = qd + 2 * kvd + (m->layers[0].wg.data ? qd : 0);
        id<MTLBuffer> qkvgb = mt->f_qkvg[0].type ? B(mt, @"qkvg", f * n * qkvg_w) : nil, gub = mt->f_gu[0].type ? B(mt, @"gu", f * n * 2 * c->ff) : nil;
        // host-side inputs, written into shared memory before commit
        int32_t *toks = malloc(sizeof(int32_t) * n);
        uint32_t *posv = malloc(sizeof(uint32_t) * n);
        for (int i = 0; i < n; i++) { toks[i] = rows[i].token; posv[i] = (uint32_t)rows[i].pos; }
        model_embed(m, toks, n, (float *)xb.contents);
        write_buf(B(mt, @"pos", 4 * n), posv, 4 * n);
        // page tables: one per distinct sequence in the step
        int nseq = 0;
        for (int i = 0; i < n; i++) if (rows[i].seq + 1 > nseq) nseq = rows[i].seq + 1;
        size_t tab_n = 1;
        for (int s = 0; s < nseq; s++) tab_n += 2 * (size_t)pages[s]->n_pages;
        uint32_t *tab = calloc(tab_n, 4), *off_full = malloc(4 * nseq), *off_swa = malloc(4 * nseq);
        size_t t = 0;
        for (int s = 0; s < nseq; s++) {
            off_full[s] = (uint32_t)t;
            memcpy(tab + t, pages[s]->full, 4 * pages[s]->n_pages);
            t += pages[s]->n_pages;
            off_swa[s] = (uint32_t)t;
            if (kv->window) memcpy(tab + t, pages[s]->swa, 4 * pages[s]->n_pages);
            t += pages[s]->n_pages;
        }
        write_buf(B(mt, @"tab", 4 * tab_n), tab, 4 * tab_n);
        RowInfo *ri_full = malloc(sizeof(RowInfo) * n), *ri_swa = malloc(sizeof(RowInfo) * n);
        uint64_t *slot_full = malloc(8 * n), *slot_swa = malloc(8 * n), *sslot_full = malloc(8 * n), *sslot_swa = malloc(8 * n);
        bool q8 = kv->full.elem == 1; // 8-bit KV
        uint32_t max_sp_full = 1, max_sp_swa = 1; // max attention splits over rows
        int *rseq = malloc(sizeof(int) * n), *rpos = malloc(sizeof(int) * n), rmax_full = 1, rmax_swa = 1;
        uint32_t *grp_full = malloc(8 * n), *grp_swa = malloc(8 * n);
        for (int i = 0; i < n; i++) {
            const row_t *r = &rows[i];
            const seq_pages *pg = pages[r->seq];
            uint32_t sw = kv->window && r->ctx_end + 1 > kv->window ? (uint32_t)(r->ctx_end + 1 - kv->window) : 0;
            uint32_t nf = (uint32_t)r->ctx_end + 1, nw = nf - sw;
            ri_full[i] = (RowInfo){ (uint32_t)r->ctx_end, 0, off_full[r->seq], attn_split(nf), 0 };
            ri_swa[i] = (RowInfo){ (uint32_t)r->ctx_end, sw, off_swa[r->seq], attn_split(nw), 0 };
            rseq[i] = r->seq;
            rpos[i] = r->pos;
            slot_full[i] = (uint64_t)pg->full[r->pos / KV_PAGE] * kv->full.page_stride + (uint64_t)(r->pos % KV_PAGE) * hd;
            slot_swa[i] = kv->window ? (uint64_t)pg->swa[r->pos / KV_PAGE] * kv->swa.page_stride + (uint64_t)(r->pos % KV_PAGE) * hd : 0;
            sslot_full[i] = (uint64_t)pg->full[r->pos / KV_PAGE] * kv->full.page_stride + kv->full.scale_off + (uint64_t)(r->pos % KV_PAGE) * 2;
            sslot_swa[i] = kv->window ? (uint64_t)pg->swa[r->pos / KV_PAGE] * kv->swa.page_stride + kv->swa.scale_off + (uint64_t)(r->pos % KV_PAGE) * 2 : 0;
        }
        // input rows go to attn_nax in runs of consecutive positions of one sequence
        // (64 rows per group); the others get compacted partials slots
        uint32_t *ngrp = malloc(8 * n), *pslot = malloc(4 * n), *nslot = malloc(4 * n), *vrows = malloc(4 * n), n_part = 0, n_nrow = 0, nax_sp = 1;
        uint8_t *skip = calloc(n, 1); // rows not in the matrix kernel's groups
        int n_nax = 0, n_vec = 0;
        bool nax_attn = mt->nax && attn_nax_on() && (hd == 64 || hd == 128), vec_attn = attn_vec_on();
        for (int i = 0; i < n; i++) {
            if (vec_attn && !rows[i].input) {
                vrows[n_vec++] = (uint32_t)i;
                skip[i] = 1;
                const RowInfo *rs[2] = { &ri_full[i], &ri_swa[i] };
                uint32_t *mx[2] = { &max_sp_full, &max_sp_swa };
                for (int k = 0; k < 2; k++) {
                    uint32_t S = rs[k]->split, o = rs[k]->start / S * S, ns = (rs[k]->ctx_end + 1 - o + S - 1) / S;
                    if (ns > *mx[k]) *mx[k] = ns;
                }
            }
            if (nax_attn && rows[i].input) skip[i] = 1;
        }
        if (n_vec) write_buf(B(mt, @"vrows", 4 * n), vrows, 4 * n_vec);
        free(vrows);
        for (int i = 0; i < n;) {
            bool use = nax_attn && rows[i].input;
            int j = i + 1;
            while (j < n && rows[j].seq == rows[i].seq && rows[j].pos == rows[j - 1].pos + 1 && (nax_attn && rows[j].input) == use) j++;
            for (int k = i; k < j; k++) {
                ri_full[k].part = ri_swa[k].part = pslot[k] = use ? NAX_ROW : n_part++;
                nslot[k] = use ? n_nrow++ : NAX_ROW;
            }
            for (int k = i; use && k < j; k += 64, n_nax++) {
                int last = (k + 64 < j ? k + 64 : j) - 1;
                uint32_t ns = (uint32_t)rows[last].ctx_end / nax_span() + 1;
                if (ns > nax_sp) nax_sp = ns;
                ngrp[2 * n_nax] = (uint32_t)k;
                ngrp[2 * n_nax + 1] = (uint32_t)(last + 1 - k);
            }
            i = j;
        }
        if (n_nax) write_buf(B(mt, @"ngrp", 8 * n), ngrp, 8 * n_nax);
        write_buf(B(mt, @"pslot", 4 * n), pslot, 4 * n);
        if (n_nax && nax_sp > 1) write_buf(B(mt, @"nslot", 4 * n), nslot, 4 * n);
        free(ngrp); free(pslot); free(nslot);
        int G = c->n_head / c->n_kv;
        int ng_full = attn_groups(ri_full, rseq, rpos, skip, n, G, grp_full, &rmax_full, &max_sp_full);
        int ng_swa = attn_groups(ri_swa, rseq, rpos, skip, n, G, grp_swa, &rmax_swa, &max_sp_swa);
        free(skip);
        write_buf(B(mt, @"grp_full", 8 * n), grp_full, 8 * ng_full);
        write_buf(B(mt, @"grp_swa", 8 * n), grp_swa, 8 * ng_swa);
        free(rseq); free(rpos); free(grp_full); free(grp_swa);
        write_buf(B(mt, @"ri_full", sizeof(RowInfo) * n), ri_full, sizeof(RowInfo) * n);
        write_buf(B(mt, @"ri_swa", sizeof(RowInfo) * n), ri_swa, sizeof(RowInfo) * n);
        write_buf(B(mt, @"slot_full", 8 * n), slot_full, 8 * n);
        write_buf(B(mt, @"slot_swa", 8 * n), slot_swa, 8 * n);
        write_buf(B(mt, @"sslot_full", 8 * n), sslot_full, 8 * n);
        write_buf(B(mt, @"sslot_swa", 8 * n), sslot_swa, 8 * n);
        free(sslot_full); free(sslot_swa);
        int ns = 0;
        uint32_t *sel = malloc(4 * n);
        for (int i = 0; i < n; i++) if (rows[i].logits) sel[ns++] = (uint32_t)i;
        if (ns) write_buf(B(mt, @"sel", 4 * ns), sel, 4 * ns);
        int n_split_max = (int)(max_sp_full > max_sp_swa ? max_sp_full : max_sp_swa);
        id<MTLBuffer> part = B(mt, @"part", (size_t)(n_part ? n_part : 1) * c->n_head * n_split_max * (hd + 2) * 4);
        id<MTLBuffer> npart = n_nax && nax_sp > 1 ? B(mt, @"npart", (size_t)n_nrow * c->n_head * nax_sp * (hd + 2) * 4) : part;
        id<MTLBuffer> full_pool = (__bridge id<MTLBuffer>)kv->full.handle;
        id<MTLBuffer> swa_pool = kv->window ? (__bridge id<MTLBuffer>)kv->swa.handle : full_pool;
        id<MTLBuffer> hsel = ns ? B(mt, @"hsel", act_bytes(ns, d)) : nil, lg = ns ? B(mt, @"logits", f * (size_t)ns * c->vocab) : nil;

        id<MTLCommandBuffer> cb = [mt->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        bool fd = c->n_head / c->n_kv <= 16 && (hd == 64 || hd == 128);
        if (!fd) { fprintf(stderr, "metal: head_dim %d / GQA group %d not supported by attention kernel\n", hd, c->n_head / c->n_kv); exit(1); }
        for (int li = 0; li < c->n_layer; li++) {
            const layer *l = &m->layers[li];
            bool fuse_norms = add_norm_ok(d);
            if (li == 0 || !fuse_norms) rms(mt, e, xb, hb, l->attn_norm, d, c->eps, false, n, nil); // else done by the previous layer
            // q/k/v/gate: one fused matmul into [n][q|k|v|g] rows, or four separate ones
            id<MTLBuffer> qB = qb, kB = kb, vB = vb, gB = gb;
            NSUInteger qO = 0, kO = 0, vO = 0, gO = 0;
            uint32_t qs = qd, ks = kvd, vs = kvd, gs = qd;
            if (mt->f_qkvg[li].type) {
                matmul(mt, e, &mt->f_qkvg[li], hb, qkvgb, qkvg_w, n);
                qB = kB = vB = gB = qkvgb;
                kO = 4 * qd; vO = 4 * (qd + kvd); gO = 4 * (qd + 2 * kvd);
                qs = ks = vs = gs = qkvg_w;
            } else {
                matmul(mt, e, &l->wq, hb, qb, qd, n);
                matmul(mt, e, &l->wk, hb, kb, kvd, n);
                matmul(mt, e, &l->wv, hb, vb, kvd, n);
                if (l->wg.data) matmul(mt, e, &l->wg, hb, gb, qd, n);
            }
            if (l->bq) add_bcast(mt, e, qb, l->bq, nil, qd, (size_t)n * qd);
            if (l->bk) add_bcast(mt, e, kb, l->bk, nil, kvd, (size_t)n * kvd);
            if (l->bv) add_bcast(mt, e, vb, l->bv, nil, kvd, (size_t)n * kvd);
            if (l->q_norm || l->k_norm || l->rope) {
                QkArgs a = { (uint32_t)c->n_head, (uint32_t)c->n_kv, (uint32_t)hd, c->eps, l->q_norm != NULL, l->k_norm != NULL, l->rope, c->rope_neox, qs, ks };
                [e setComputePipelineState:P(mt, @"qk_norm_rope")];
                [e setBuffer:qB offset:qO atIndex:0];
                [e setBuffer:kB offset:kO atIndex:1];
                set_w(e, mt, l->q_norm, 2);
                set_w(e, mt, l->k_norm, 3);
                [e setBuffer:B(mt, @"pos", 4 * n) offset:0 atIndex:4];
                [e setBuffer:mt->invf offset:0 atIndex:5];
                set_bytes(e, &a, sizeof a, 6);
                [e dispatchThreadgroups:MTLSizeMake(n, c->n_head + c->n_kv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            const kv_pool *pl = l->swa ? &kv->swa : &kv->full;
            id<MTLBuffer> pool = l->swa ? swa_pool : full_pool;
            uint64_t layer_off = (uint64_t)l->pool_layer * 2 * c->n_kv * KV_PAGE * hd;
            {
                KvwArgs a = { (uint32_t)c->n_kv, (uint32_t)hd, KV_PAGE, ks, layer_off, vs, (uint32_t)l->pool_layer };
                [e setComputePipelineState:P(mt, q8 ? @"kv_write8" : @"kv_write")];
                [e setBuffer:kB offset:kO atIndex:0];
                [e setBuffer:vB offset:vO atIndex:1];
                [e setBuffer:pool offset:0 atIndex:2];
                [e setBuffer:B(mt, l->swa ? @"slot_swa" : @"slot_full", 8 * n) offset:0 atIndex:3];
                set_bytes(e, &a, sizeof a, 4);
                if (q8) {
                    [e setBuffer:B(mt, l->swa ? @"sslot_swa" : @"sslot_full", 8 * n) offset:0 atIndex:5];
                    [e dispatchThreadgroups:MTLSizeMake(n, c->n_kv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
                } else [e dispatchThreads:MTLSizeMake(kvd, n, 1) threadsPerThreadgroup:MTLSizeMake(kvd < 256 ? kvd : 256, 1, 1)];
            }
            {
                AttnArgs a = { (uint32_t)c->n_head, (uint32_t)c->n_kv, (uint32_t)hd, c->attn_scale, pl->page_stride, layer_off, KV_PAGE, l->wg.data != NULL, qs, gs, mt->h16,
                               (uint32_t)l->pool_layer, pl->scale_off };
                uint32_t sp[2] = { 0, l->swa ? max_sp_swa : max_sp_full }; // .x unused (per-row split sizes)
                bool mma = attn_mma_on();
                id<MTLBuffer> rib = B(mt, l->swa ? @"ri_swa" : @"ri_full", sizeof(RowInfo) * n);
                int ngr = l->swa ? ng_swa : ng_full, rmx = l->swa ? rmax_swa : rmax_full;
                if (n_vec) {
                    [e setComputePipelineState:P(mt, [NSString stringWithFormat:q8 ? @"attn_vec8_%d_%d" : @"attn_vec_%d_%d", hd, G <= 4 ? 4 : G <= 8 ? 8 : 16])];
                    [e setBuffer:qB offset:qO atIndex:0];
                    [e setBuffer:pool offset:0 atIndex:1];
                    [e setBuffer:rib offset:0 atIndex:2];
                    [e setBuffer:B(mt, @"tab", 4 * tab_n) offset:0 atIndex:3];
                    [e setBuffer:part offset:0 atIndex:4];
                    set_bytes(e, &a, sizeof a, 5);
                    set_bytes(e, sp, sizeof sp, 6);
                    [e setBuffer:B(mt, @"vrows", 4 * n) offset:0 atIndex:7];
                    [e dispatchThreadgroups:MTLSizeMake(n_vec, c->n_kv, sp[1]) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                if (ngr) {
                    [e setComputePipelineState:P(mt, [NSString stringWithFormat:mma ? (q8 ? @"attn_mma8_%d" : @"attn_mma_%d") : (q8 ? @"attn_fd8_%d" : @"attn_fd_%d"), hd])];
                    [e setBuffer:qB offset:qO atIndex:0];
                    [e setBuffer:pool offset:0 atIndex:1];
                    [e setBuffer:rib offset:0 atIndex:2];
                    [e setBuffer:B(mt, @"tab", 4 * tab_n) offset:0 atIndex:3];
                    [e setBuffer:part offset:0 atIndex:4];
                    set_bytes(e, &a, sizeof a, 5);
                    set_bytes(e, sp, sizeof sp, 6);
                    [e setBuffer:B(mt, l->swa ? @"grp_swa" : @"grp_full", 8 * n) offset:0 atIndex:7];
                    [e dispatchThreadgroups:MTLSizeMake(ngr, c->n_kv, sp[1]) threadsPerThreadgroup:MTLSizeMake(mma ? 128 : 32 * G * rmx, 1, 1)];
                }
                if (ngr || n_vec) {
                    [e setComputePipelineState:P(mt, @"attn_reduce")];
                    [e setBuffer:part offset:0 atIndex:0];
                    [e setBuffer:gB offset:gO atIndex:1];
                    [e setBuffer:ab offset:0 atIndex:2];
                    set_bytes(e, &a, sizeof a, 3);
                    set_bytes(e, sp, sizeof sp, 4);
                    [e setBuffer:B(mt, @"pslot", 4 * n) offset:0 atIndex:5];
                    [e dispatchThreadgroups:MTLSizeMake(n, c->n_head, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
                if (n_nax) {
                    [e setComputePipelineState:P(mt, [NSString stringWithFormat:q8 ? @"attn_nax8_%d" : @"attn_nax_%d", hd])];
                    [e setBuffer:qB offset:qO atIndex:0];
                    [e setBuffer:pool offset:0 atIndex:1];
                    [e setBuffer:rib offset:0 atIndex:2];
                    [e setBuffer:B(mt, @"tab", 4 * tab_n) offset:0 atIndex:3];
                    [e setBuffer:ab offset:0 atIndex:4];
                    set_bytes(e, &a, sizeof a, 5);
                    [e setBuffer:B(mt, @"ngrp", 8 * n) offset:0 atIndex:6];
                    [e setBuffer:gB offset:gO atIndex:7];
                    [e setBuffer:npart offset:0 atIndex:8];
                    [e setBuffer:nax_sp > 1 ? B(mt, @"nslot", 4 * n) : B(mt, @"pslot", 4 * n) offset:0 atIndex:9];
                    uint32_t nsp[2] = { nax_span(), nax_sp };
                    set_bytes(e, nsp, sizeof nsp, 10);
                    [e dispatchThreadgroups:MTLSizeMake(n_nax, c->n_head, nax_sp) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    if (nax_sp > 1) {
                        [e setComputePipelineState:P(mt, @"attn_reduce")];
                        [e setBuffer:npart offset:0 atIndex:0];
                        [e setBuffer:gB offset:gO atIndex:1];
                        [e setBuffer:ab offset:0 atIndex:2];
                        set_bytes(e, &a, sizeof a, 3);
                        uint32_t rsp[2] = { 0, nax_sp };
                        set_bytes(e, rsp, sizeof rsp, 4);
                        [e setBuffer:B(mt, @"nslot", 4 * n) offset:0 atIndex:5];
                        [e dispatchThreadgroups:MTLSizeMake(n, c->n_head, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    }
                }
            }
            matmul(mt, e, &l->wo, ab, ob, d, n);
            if (fuse_norms) add_norm(mt, e, ob, xb, hb, l->post_attn_norm, c->post_eps, l->ffn_norm, c->eps, d, n);
            else {
                if (l->post_attn_norm) rms(mt, e, ob, xb, l->post_attn_norm, d, c->post_eps, true, n, nil);
                else add_bcast(mt, e, xb, NULL, ob, (uint32_t)(n * d), (size_t)n * d);
                rms(mt, e, xb, hb, l->ffn_norm, d, c->eps, false, n, nil);
            }
            uint32_t sa[3] = { (uint32_t)c->ff, (uint32_t)c->ff, mt->h16 };
            if (mt->f_gu[li].type) {
                matmul(mt, e, &mt->f_gu[li], hb, gub, 2 * c->ff, n);
                sa[1] = 2 * c->ff;
                [e setComputePipelineState:P(mt, @"silu_mul")];
                [e setBuffer:gub offset:0 atIndex:0];
                [e setBuffer:gub offset:4 * c->ff atIndex:1];
            } else {
                matmul(mt, e, &l->w_gate, hb, fgb, c->ff, n);
                matmul(mt, e, &l->w_up, hb, fub, c->ff, n);
                [e setComputePipelineState:P(mt, @"silu_mul")];
                [e setBuffer:fgb offset:0 atIndex:0];
                [e setBuffer:fub offset:0 atIndex:1];
            }
            [e setBuffer:actb offset:0 atIndex:2];
            set_bytes(e, sa, sizeof sa, 3);
            [e dispatchThreads:MTLSizeMake((size_t)n * c->ff, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            matmul(mt, e, &l->w_down, actb, ob, d, n);
            if (fuse_norms && li + 1 < c->n_layer) add_norm(mt, e, ob, xb, hb, l->post_ffn_norm, c->post_eps, m->layers[li + 1].attn_norm, c->eps, d, n);
            else if (l->post_ffn_norm) rms(mt, e, ob, xb, l->post_ffn_norm, d, c->post_eps, true, n, nil);
            else add_bcast(mt, e, xb, NULL, ob, (uint32_t)(n * d), (size_t)n * d);
            for (int ci = 0; ci < be->n_cap; ci++) // residual stream after this layer, for the drafter
                if (be->cap_layers[ci] == li) {
                    uint32_t ca[3] = { (uint32_t)d, (uint32_t)(be->n_cap * d), (uint32_t)(ci * d) };
                    [e setComputePipelineState:P(mt, @"cap_write")];
                    [e setBuffer:xb offset:0 atIndex:0];
                    [e setBuffer:B(mt, @"cap", 2 * (size_t)n * be->n_cap * d) offset:0 atIndex:1];
                    set_bytes(e, ca, sizeof ca, 2);
                    [e dispatchThreads:MTLSizeMake(d, n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
        }
        if (ns) {
            rms(mt, e, xb, hsel, m->out_norm, d, c->eps, false, ns, B(mt, @"sel", 4 * ns));
            matmul(mt, e, &m->output, hsel, lg, c->vocab, ns);
            if (c->softcap > 0 || c->logit_scale != 1.f) {
                CapArgs a = { c->logit_scale, c->softcap };
                [e setComputePipelineState:P(mt, @"softcap")];
                [e setBuffer:lg offset:0 atIndex:0];
                set_bytes(e, &a, sizeof a, 1);
                [e dispatchThreads:MTLSizeMake((size_t)ns * c->vocab, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            if (be->argmax) argmax_dispatch(mt, e, lg, B(mt, @"amax", 4 * ns), c->vocab, ns);
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        static int prof = -1;
        if (prof < 0) prof = getenv("CINFER_PROFILE") != NULL;
        if (prof) { // per-step GPU time, averaged per batch size
            static double acc[1025];
            static int cnt[1025];
            int k = n > 1024 ? 1024 : n;
            acc[k] += cb.GPUEndTime - cb.GPUStartTime;
            if (++cnt[k] % 16 == 0 || getenv("CINFER_PROFILE_EACH"))
                fprintf(stderr, "[profile] n=%d ctx<=%d gpu %.2f ms/step\n", n, rows[n - 1].ctx_end + 1, getenv("CINFER_PROFILE_EACH") ? (cb.GPUEndTime - cb.GPUStartTime) * 1e3 : acc[k] / cnt[k] * 1e3);
        }
        free(toks); free(posv); free(tab); free(off_full); free(off_swa); free(ri_full); free(ri_swa);
        free(slot_full); free(slot_swa); free(sel);
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "metal: command buffer failed (status %ld): %s\n", (long)cb.status, cb.error.localizedDescription.UTF8String);
            return 1;
        }
        if (ns && logits) memcpy(logits, lg.contents, sizeof(float) * (size_t)ns * c->vocab);
        if (ns && be->argmax) memcpy(be->argmax, B(mt, @"amax", 4 * ns).contents, 4 * (size_t)ns);
        return 0;
    }
}

static void compile_into(CIMetal *mt, NSString *src, MTLCompileOptions *opt) {
    NSError *err = nil;
    id<MTLLibrary> lib = [mt->dev newLibraryWithSource:src options:opt ? opt : [MTLCompileOptions new] error:&err];
    if (!lib) { fprintf(stderr, "metal: kernel compile failed: %s\n", err.localizedDescription.UTF8String); exit(1); }
    for (NSString *name in lib.functionNames) {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        id<MTLComputePipelineState> p = [mt->dev newComputePipelineStateWithFunction:fn error:&err];
        if (!p) { fprintf(stderr, "metal: pipeline %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
        mt->pso[name] = p;
    }
}

// ---------------------------------------------------------------- DFlash drafter

static size_t repack_one(CIMetal *mt, qmat *w);
static size_t fuse_tiled(CIMetal *mt, qmat **ws, int n, qmat *out);

static int metal_attach(backend *b, model *dm) {
    @autoreleasepool {
        CIMetal *mt = (__bridge CIMetal *)b->impl;
        if (!mt->nax) { fprintf(stderr, "metal: the drafter needs the NAX (M5) path\n"); return 1; }
        for (int i = 0; i < dm->n_regions && mt->n_w < 66; i++) { // norm vectors etc.
            int k = mt->n_w++;
            mt->wbase[k] = dm->regions[i].base;
            mt->wlen[k] = (dm->regions[i].len + 16383) / 16384 * 16384;
            mt->wbuf[k] = [mt->dev newBufferWithBytesNoCopy:(void *)mt->wbase[k] length:mt->wlen[k] options:MTLResourceStorageModeShared deallocator:nil];
        }
        double t0 = now_secs();
        size_t bytes = 0;
        mt->d_kv = calloc(dm->c.n_layer, sizeof(qmat));
        mt->d_gu = calloc(dm->c.n_layer, sizeof(qmat));
        for (int i = 0; i < dm->c.n_layer; i++) {
            layer *l = &dm->layers[i];
            qmat *kv[] = { &l->wk, &l->wv }, *gu[] = { &l->w_gate, &l->w_up };
            bytes += fuse_tiled(mt, kv, 2, &mt->d_kv[i]) + fuse_tiled(mt, gu, 2, &mt->d_gu[i]);
            qmat *rest[] = { &l->wq, &l->wo, &l->w_down };
            for (int k = 0; k < 3; k++) if (!rest[k]->gpu) bytes += repack_one(mt, rest[k]);
            if (!mt->d_kv[i].gpu || !mt->d_gu[i].gpu) { fprintf(stderr, "metal: drafter layer %d cannot be tiled\n", i); return 1; }
        }
        bytes += repack_one(mt, &dm->fc);
        model_drop_host_copies(dm);
        mt->d_invf = [mt->dev newBufferWithBytes:dm->inv_freq length:sizeof(float) * dm->c.hd / 2 options:MTLResourceStorageModeShared];
        fprintf(stderr, "[metal] drafter: %.2f GB of 4-bit tiles in %.1fs\n", bytes / 1e9, now_secs() - t0);
        return 0;
    }
}

// One drafter step over context rows (KV from target hidden states) and block
// rows (anchor + masks, bidirectional within the block); writes the argmax of
// block rows 1..B-1 of every block (blocks are consecutive runs of B = blk_len rows).
static int metal_draft(backend *be, const model *dm, const model *tm, kv_cache *dkv, const seq_pages *const *pages, const draft_row *ctx, int n_ctx,
                       const draft_row *blk, int n_blk, int blk_len, int32_t *out) {
    @autoreleasepool {
        CIMetal *mt = (__bridge CIMetal *)be->impl;
        const model_cfg *c = &dm->c;
        int d = c->d, hd = c->hd, qd = c->n_head * hd, kvd = c->n_kv * hd, R = n_ctx + n_blk, B_ = blk_len;
        int n_sel = n_blk / B_ * (B_ - 1);
        size_t f = 4;
        // host-side inputs
        int nseq = 0;
        for (int i = 0; i < n_ctx; i++) if (ctx[i].seq + 1 > nseq) nseq = ctx[i].seq + 1;
        for (int i = 0; i < n_blk; i++) if (blk[i].seq + 1 > nseq) nseq = blk[i].seq + 1;
        size_t tab_n = 1;
        for (int s = 0; s < nseq; s++) tab_n += pages[s] ? (size_t)pages[s]->n_pages : 0;
        uint32_t *tab = calloc(tab_n, 4), *off = calloc(nseq ? nseq : 1, 4), *posv = malloc(4 * (R ? R : 1)), *gidx = malloc(4 * (n_ctx ? n_ctx : 1));
        uint32_t *sel = malloc(4 * (n_sel ? n_sel : 1));
        uint64_t *slot = malloc(8 * (R ? R : 1)), *sslot = malloc(8 * (R ? R : 1));
        bool q8 = dkv->full.elem == 1;
        RowInfo *ri = malloc(sizeof(RowInfo) * (n_blk ? n_blk : 1));
        size_t t = 0;
        for (int s = 0; s < nseq; s++) {
            off[s] = (uint32_t)t;
            if (pages[s]) { memcpy(tab + t, pages[s]->full, 4 * pages[s]->n_pages); t += pages[s]->n_pages; }
        }
        uint32_t max_sp = 1;
        for (int i = 0; i < R; i++) {
            const draft_row *r = i < n_ctx ? &ctx[i] : &blk[i - n_ctx];
            posv[i] = (uint32_t)r->pos;
            slot[i] = (uint64_t)pages[r->seq]->full[r->pos / KV_PAGE] * dkv->full.page_stride + (uint64_t)(r->pos % KV_PAGE) * hd;
            sslot[i] = (uint64_t)pages[r->seq]->full[r->pos / KV_PAGE] * dkv->full.page_stride + dkv->full.scale_off + (uint64_t)(r->pos % KV_PAGE) * 2;
            if (i < n_ctx) gidx[i] = (uint32_t)r->src;
            else {
                uint32_t end = (uint32_t)r->ctx_end, start = c->window && end + 1 > (uint32_t)c->window ? end + 1 - c->window : 0;
                ri[i - n_ctx] = (RowInfo){ end, start, off[r->seq], attn_split(end + 1 - start), (uint32_t)(i - n_ctx) };
            }
        }
        for (int b = 0, k = 0; b < n_blk / B_; b++) for (int j = 1; j < B_; j++) sel[k++] = (uint32_t)(b * B_ + j);
        int *bseq = malloc(sizeof(int) * (n_blk ? n_blk : 1)), *bpos = malloc(sizeof(int) * (n_blk ? n_blk : 1)), d_rmax = 1, dG = c->n_head / c->n_kv;
        for (int i = 0; i < n_blk; i++) { bseq[i] = blk[i].seq; bpos[i] = blk[i].pos; }
        uint32_t *dgrp = malloc(8 * (n_blk ? n_blk : 1));
        int d_ng = n_blk ? attn_groups(ri, bseq, bpos, NULL, n_blk, dG, dgrp, &d_rmax, &max_sp) : 0;
        if (d_ng) write_buf(B(mt, @"d_grp", 8 * n_blk), dgrp, 8 * d_ng);
        free(bseq); free(bpos); free(dgrp);
        id<MTLBuffer> hb = B(mt, @"d_h", f * (n_blk ? n_blk : 1) * d), ctxn = B(mt, @"d_ctxn", act_bytes(n_ctx ? n_ctx : 1, d)),
                      hnb = B(mt, @"d_hn", act_bytes(n_blk, d)), kvb = B(mt, @"d_kv", f * (size_t)R * 2 * kvd), qb = B(mt, @"d_q", f * n_blk * qd),
                      ab = B(mt, @"d_att", act_bytes(n_blk, qd)), ob = B(mt, @"d_o", f * n_blk * d), gub = B(mt, @"d_gu", f * (size_t)n_blk * 2 * c->ff),
                      actb = B(mt, @"d_act", act_bytes(n_blk, c->ff)), hsel = B(mt, @"d_hsel", act_bytes(n_sel, d)),
                      lg = B(mt, @"d_logits", f * (size_t)n_sel * tm->c.vocab), amax = B(mt, @"d_amax", 4 * n_sel);
        int32_t *toks = malloc(4 * (n_blk ? n_blk : 1));
        for (int i = 0; i < n_blk; i++) toks[i] = blk[i].src;
        model_embed_raw(tm, toks, n_blk, (float *)hb.contents); // target embeddings, no embed norm
        free(toks);
        write_buf(B(mt, @"d_pos", 4 * R), posv, 4 * R);
        write_buf(B(mt, @"d_slot", 8 * R), slot, 8 * R);
        write_buf(B(mt, @"d_sslot", 8 * R), sslot, 8 * R);
        free(sslot);
        write_buf(B(mt, @"d_tab", 4 * tab_n), tab, 4 * tab_n);
        if (n_blk) {
            write_buf(B(mt, @"d_ri", sizeof(RowInfo) * n_blk), ri, sizeof(RowInfo) * n_blk);
            uint32_t *ps = malloc(4 * n_blk);
            for (int i = 0; i < n_blk; i++) ps[i] = (uint32_t)i;
            write_buf(B(mt, @"d_pslot", 4 * n_blk), ps, 4 * n_blk);
            free(ps);
        }
        if (n_ctx) write_buf(B(mt, @"d_gidx", 4 * n_ctx), gidx, 4 * n_ctx);
        if (n_sel) write_buf(B(mt, @"d_sel", 4 * n_sel), sel, 4 * n_sel);
        id<MTLBuffer> part = B(mt, @"part", (size_t)(n_blk ? n_blk : 1) * c->n_head * max_sp * (hd + 2) * 4);
        id<MTLBuffer> pool = (__bridge id<MTLBuffer>)dkv->full.handle;

        id<MTLCommandBuffer> cb = [mt->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        int fcw = d * dm->n_target;
        if (n_ctx) { // context: rms(fc(concat target hidden states)) -> ctxn
            id<MTLBuffer> cin = B(mt, @"d_cin", act_bytes(n_ctx, fcw)), cf = B(mt, @"d_cf", f * (size_t)n_ctx * d);
            uint32_t w = (uint32_t)fcw;
            [e setComputePipelineState:P(mt, @"gather_half")];
            [e setBuffer:B(mt, @"cap", 2) offset:0 atIndex:0];
            [e setBuffer:cin offset:0 atIndex:1];
            [e setBuffer:B(mt, @"d_gidx", 4 * n_ctx) offset:0 atIndex:2];
            set_bytes(e, &w, 4, 3);
            [e dispatchThreads:MTLSizeMake(fcw, n_ctx, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            matmul_affine(mt, e, &dm->fc, cin, 0, true, cf, 0, d, n_ctx); // cin is always f16
            rms(mt, e, cf, ctxn, dm->enc_norm, d, c->eps, false, n_ctx, nil);
        }
        for (int li = 0; li < c->n_layer; li++) {
            const layer *l = &dm->layers[li];
            uint64_t layer_off = (uint64_t)l->pool_layer * 2 * c->n_kv * KV_PAGE * hd;
            if (n_blk) rms(mt, e, hb, hnb, l->attn_norm, d, c->eps, false, n_blk, nil);
            // k|v for context rows then block rows; q for block rows
            if (n_ctx) matmul_affine(mt, e, &mt->d_kv[li], ctxn, 0, mt->h16, kvb, 0, 2 * kvd, n_ctx);
            if (n_blk) {
                matmul_affine(mt, e, &mt->d_kv[li], hnb, 0, mt->h16, kvb, (NSUInteger)n_ctx * 2 * kvd * f, 2 * kvd, n_blk);
                matmul_affine(mt, e, &l->wq, hnb, 0, mt->h16, qb, 0, qd, n_blk);
            }
            // QK norm + RoPE: K over all rows, Q over block rows
            for (int pass = 0; pass < 2; pass++) {
                int rows = pass ? n_blk : R;
                if (!rows) continue;
                QkArgs a = { pass ? (uint32_t)c->n_head : 0, pass ? 0 : (uint32_t)c->n_kv, (uint32_t)hd, c->eps, 1, 1, 1, 1, (uint32_t)qd, (uint32_t)(2 * kvd) };
                [e setComputePipelineState:P(mt, @"qk_norm_rope")];
                [e setBuffer:qb offset:0 atIndex:0];
                [e setBuffer:kvb offset:0 atIndex:1];
                set_w(e, mt, l->q_norm, 2);
                set_w(e, mt, l->k_norm, 3);
                [e setBuffer:B(mt, @"d_pos", 4 * R) offset:pass ? 4 * (NSUInteger)n_ctx : 0 atIndex:4];
                [e setBuffer:mt->d_invf offset:0 atIndex:5];
                set_bytes(e, &a, sizeof a, 6);
                [e dispatchThreadgroups:MTLSizeMake(rows, pass ? c->n_head : c->n_kv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            {
                KvwArgs a = { (uint32_t)c->n_kv, (uint32_t)hd, KV_PAGE, (uint32_t)(2 * kvd), layer_off, (uint32_t)(2 * kvd), (uint32_t)l->pool_layer };
                [e setComputePipelineState:P(mt, q8 ? @"kv_write8" : @"kv_write")];
                [e setBuffer:kvb offset:0 atIndex:0];
                [e setBuffer:kvb offset:4 * (NSUInteger)kvd atIndex:1];
                [e setBuffer:pool offset:0 atIndex:2];
                [e setBuffer:B(mt, @"d_slot", 8 * R) offset:0 atIndex:3];
                set_bytes(e, &a, sizeof a, 4);
                if (q8) {
                    [e setBuffer:B(mt, @"d_sslot", 8 * R) offset:0 atIndex:5];
                    [e dispatchThreadgroups:MTLSizeMake(R, c->n_kv, 1) threadsPerThreadgroup:MTLSizeMake(hd, 1, 1)];
                } else [e dispatchThreads:MTLSizeMake(kvd, R, 1) threadsPerThreadgroup:MTLSizeMake(kvd < 256 ? kvd : 256, 1, 1)];
            }
            if (!n_blk) continue;
            {
                AttnArgs a = { (uint32_t)c->n_head, (uint32_t)c->n_kv, (uint32_t)hd, c->attn_scale, dkv->full.page_stride, layer_off, KV_PAGE, 0, (uint32_t)qd, 0, mt->h16,
                               (uint32_t)l->pool_layer, dkv->full.scale_off };
                uint32_t sp[2] = { 0, max_sp };
                bool mma = attn_mma_on();
                [e setComputePipelineState:P(mt, [NSString stringWithFormat:mma ? (q8 ? @"attn_mma8_%d" : @"attn_mma_%d") : (q8 ? @"attn_fd8_%d" : @"attn_fd_%d"), hd])];
                [e setBuffer:qb offset:0 atIndex:0];
                [e setBuffer:pool offset:0 atIndex:1];
                [e setBuffer:B(mt, @"d_ri", sizeof(RowInfo) * n_blk) offset:0 atIndex:2];
                [e setBuffer:B(mt, @"d_tab", 4 * tab_n) offset:0 atIndex:3];
                [e setBuffer:part offset:0 atIndex:4];
                set_bytes(e, &a, sizeof a, 5);
                set_bytes(e, sp, sizeof sp, 6);
                [e setBuffer:B(mt, @"d_grp", 8 * n_blk) offset:0 atIndex:7];
                [e dispatchThreadgroups:MTLSizeMake(d_ng, c->n_kv, max_sp) threadsPerThreadgroup:MTLSizeMake(mma ? 128 : 32 * dG * d_rmax, 1, 1)];
                [e setComputePipelineState:P(mt, @"attn_reduce")];
                [e setBuffer:part offset:0 atIndex:0];
                [e setBuffer:qb offset:0 atIndex:1]; // gate unused
                [e setBuffer:ab offset:0 atIndex:2];
                set_bytes(e, &a, sizeof a, 3);
                set_bytes(e, sp, sizeof sp, 4);
                [e setBuffer:B(mt, @"d_pslot", 4 * n_blk) offset:0 atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake(n_blk, c->n_head, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            matmul_affine(mt, e, &l->wo, ab, 0, mt->h16, ob, 0, d, n_blk);
            add_bcast(mt, e, hb, NULL, ob, (uint32_t)(n_blk * d), (size_t)n_blk * d);
            rms(mt, e, hb, hnb, l->ffn_norm, d, c->eps, false, n_blk, nil);
            matmul_affine(mt, e, &mt->d_gu[li], hnb, 0, mt->h16, gub, 0, 2 * c->ff, n_blk);
            uint32_t sa[3] = { (uint32_t)c->ff, (uint32_t)(2 * c->ff), mt->h16 };
            [e setComputePipelineState:P(mt, @"silu_mul")];
            [e setBuffer:gub offset:0 atIndex:0];
            [e setBuffer:gub offset:4 * (NSUInteger)c->ff atIndex:1];
            [e setBuffer:actb offset:0 atIndex:2];
            set_bytes(e, sa, sizeof sa, 3);
            [e dispatchThreads:MTLSizeMake((size_t)n_blk * c->ff, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            matmul_affine(mt, e, &l->w_down, actb, 0, mt->h16, ob, 0, d, n_blk);
            add_bcast(mt, e, hb, NULL, ob, (uint32_t)(n_blk * d), (size_t)n_blk * d);
        }
        if (n_sel) { // final norm on rows 1.. of each block, target lm_head, argmax
            rms(mt, e, hb, hsel, dm->out_norm, d, c->eps, false, n_sel, B(mt, @"d_sel", 4 * n_sel));
            matmul(mt, e, &tm->output, hsel, lg, tm->c.vocab, n_sel);
            argmax_dispatch(mt, e, lg, amax, tm->c.vocab, n_sel);
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        free(tab); free(off); free(posv); free(gidx); free(sel); free(slot); free(ri);
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "metal: drafter command buffer failed: %s\n", cb.error.localizedDescription.UTF8String);
            return 1;
        }
        if (n_sel) memcpy(out, amax.contents, 4 * (size_t)n_sel);
        return 0;
    }
}

static void metal_destroy(backend *b) {
    CFBridgingRelease(b->impl);
    free(b);
}

// Repack one 4-bit affine matrix into the tiled layout (w->gpu); returns bytes.
static size_t repack_one(CIMetal *mt, qmat *w) {
    int tiles = (w->rows + TILE - 1) / TILE;
    size_t bytes = (size_t)tiles * (w->cols / TILE) * tiled_block_bytes(w);
    id<MTLBuffer> buf = [mt->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buf) { fprintf(stderr, "metal: allocation of %zu bytes for repacked weights failed\n", bytes); exit(1); }
    repack_job j = { w, buf.contents };
    parallel_for(tiles, 4, repack_rows, &j);
    tiled_t *tw = calloc(1, sizeof *tw);
    tw->buf = (__bridge_retained void *)buf;
    tw->split = tiled_split(w, 2, 2048, 4);
    tw->split_mv = tiled_split(w, 8, 8192, 8);
    w->gpu = tw;
    return bytes;
}

// Repack several matrices sharing in-features into one tiled matrix (rows
// stacked in order) so they run as one GEMM; returns bytes, or 0 when the
// matrices cannot be fused (then out->type stays 0).
static size_t fuse_tiled(CIMetal *mt, qmat **ws, int n, qmat *out) {
    int rows = 0;
    for (int i = 0; i < n; i++) {
        const qmat *w = ws[i];
        if (!w->data || !tiled_ok(w) || w->gpu || w->rows % TILE || w->cols != ws[0]->cols || w->group != ws[0]->group ||
            w->scales_f16 != ws[0]->scales_f16)
            return 0;
        rows += w->rows;
    }
    *out = *ws[0];
    out->rows = rows;
    size_t bb = tiled_block_bytes(out), per_tile = (size_t)(out->cols / TILE) * bb, bytes = (size_t)(rows / TILE) * per_tile;
    id<MTLBuffer> buf = [mt->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buf) { fprintf(stderr, "metal: allocation of %zu bytes for repacked weights failed\n", bytes); exit(1); }
    tiled_t *tw = calloc(1, sizeof *tw);
    tw->buf = (__bridge_retained void *)buf;
    tw->split = tiled_split(out, 2, 2048, 4);
    tw->split_mv = tiled_split(out, 8, 8192, 8);
    uint8_t *dst = buf.contents;
    for (int i = 0; i < n; i++) {
        repack_job j = { ws[i], dst };
        parallel_for(ws[i]->rows / TILE, 4, repack_rows, &j);
        dst += (size_t)(ws[i]->rows / TILE) * per_tile;
        ws[i]->gpu = tw; // the parts now live in the fused copy
    }
    out->gpu = tw;
    return bytes;
}

backend *metal_backend_new(model *m) {
    @autoreleasepool {
        CIMetal *mt = [CIMetal new];
        mt->dev = MTLCreateSystemDefaultDevice();
        if (!mt->dev) { fprintf(stderr, "metal: no device\n"); exit(1); }
        mt->queue = [mt->dev newCommandQueue];
        // affine kernels are templates: instantiate the (scale type, group, bits) combos in use
        NSMutableString *inst = [NSMutableString new], *inst_nax = [NSMutableString new];
        NSMutableSet *seen = [NSMutableSet new];
        mt->nax = [mt->dev supportsFamily:MTLGPUFamilyMetal4] && [mt->dev supportsFamily:MTLGPUFamilyApple10] && !getenv("CINFER_NO_NAX");
        const qmat *mats[8 * 1024];
        int nm = 0;
        mats[nm++] = &m->output;
        for (int i = 0; i < m->c.n_layer && nm + 8 <= 8 * 1024; i++) {
            const layer *l = &m->layers[i];
            const qmat *ls[] = { &l->wq, &l->wk, &l->wv, &l->wg, &l->wo, &l->w_gate, &l->w_up, &l->w_down };
            for (int k = 0; k < 8; k++) mats[nm++] = ls[k];
        }
        // a drafter attached later is quantized to 4-bit / group 64 / bf16 scales
        static const qmat drafter_fmt = { .type = QT_AFFINE, .rows = 64, .cols = 64, .bits = 4, .group = 64, .data = (const uint8_t *)1 };
        for (int i = 0; i <= nm; i++) {
            const qmat *w = i < nm ? mats[i] : mt->nax ? &drafter_fmt : NULL;
            if (!w || !w->data || w->type != QT_AFFINE) continue;
            if (w->bits != 2 && w->bits != 4 && w->bits != 8) { fprintf(stderr, "metal: %d-bit affine weights not supported\n", w->bits); exit(1); }
            if (w->cols % 32) { fprintf(stderr, "metal: affine weights need in-features %% 32 == 0\n"); exit(1); }
            NSString *key = [NSString stringWithFormat:@"%s_%d_%d", sname(w), w->group, w->bits];
            if ([seen containsObject:key]) continue;
            [seen addObject:key];
            const char *st = w->scales_f16 ? "half" : "bfloat";
            for (int nt = 1; nt <= 8; nt *= 2)
                [inst appendFormat:@"template [[host_name(\"amv_%@_%d\")]] kernel decltype(amv<%s, %d, %d, %d>) amv<%s, %d, %d, %d>;\n", key, nt, st, w->group, w->bits, nt, st, w->group, w->bits, nt];
            if (tiled_ok(w)) {
                NSString *kg = [key substringToIndex:[key rangeOfString:@"_" options:NSBackwardsSearch].location];
                for (int nt = 1; nt <= TMV_MAX; nt = nt == 2 ? 4 : nt + 1)
                    [inst_nax appendFormat:@"template [[host_name(\"tmv_%@_%d\")]] [[kernel]] decltype(tmv<%s, %d, %d>) tmv<%s, %d, %d>;\n", kg, nt, st, w->group, nt, st, w->group, nt];
            }
            if (tiled_ok(w))
                for (int bm = 16; bm <= 32; bm *= 2)
                    [inst_nax appendFormat:@"template [[host_name(\"aqms_%@_%d\")]] [[kernel]] decltype(aqms<%s, half, %d, %d, %d>) aqms<%s, half, %d, %d, %d>;\n", [key substringToIndex:[key rangeOfString:@"_" options:NSBackwardsSearch].location], bm,
                                           st, w->group, w->bits, bm, st, w->group, w->bits, bm];
        }
        mt->pso = [NSMutableDictionary new];
        NSString *src = [[NSString alloc] initWithBytes:kernels_metal length:kernels_metal_len encoding:NSUTF8StringEncoding];
        compile_into(mt, [src stringByAppendingString:inst], nil);
        if (mt->nax && inst_nax.length) {
            NSString *nsrc = [[NSString alloc] initWithBytes:kernels_nax_metal length:kernels_nax_metal_len encoding:NSUTF8StringEncoding];
            MTLCompileOptions *o = [MTLCompileOptions new];
            o.languageVersion = MTLLanguageVersion4_0;
            compile_into(mt, [nsrc stringByAppendingString:inst_nax], o);
        }
        mt->bufs = [NSMutableDictionary new];
        // repack 4-bit affine matrices into the tiled layout (NAX path); others keep
        // the original layout, which then has to be 16-byte aligned
        double t0 = now_secs();
        size_t repacked = 0;
        bool need_align = false;
        tp_init(0);
        mt->f_qkvg = calloc(m->c.n_layer, sizeof(qmat));
        mt->f_gu = calloc(m->c.n_layer, sizeof(qmat));
        if (mt->nax && !getenv("CINFER_NO_FUSE"))
            for (int i = 0; i < m->c.n_layer; i++) {
                layer *l = &m->layers[i];
                qmat *qkvg[] = { &l->wq, &l->wk, &l->wv, &l->wg }, *gu[] = { &l->w_gate, &l->w_up };
                if (!l->bq && !l->bk && !l->bv) repacked += fuse_tiled(mt, qkvg, l->wg.data ? 4 : 3, &mt->f_qkvg[i]);
                repacked += fuse_tiled(mt, gu, 2, &mt->f_gu[i]);
            }
        for (int i = 0; i < nm; i++) {
            qmat *w = (qmat *)mats[i];
            if (!w->data || w->type != QT_AFFINE || w->gpu) continue;
            if (!mt->nax || !tiled_ok(w)) { need_align = true; continue; }
            bool dup = false; // tied embeddings: output may alias another matrix
            for (int k = 0; k < i; k++) if (mats[k]->data == w->data && mats[k]->gpu) { w->gpu = mats[k]->gpu; dup = true; }
            if (dup) continue;
            repacked += repack_one(mt, w);
        }
        if (need_align) model_align_mats(m);
        mt->h16 = m->output.gpu != NULL;
        for (int i = 0; i < nm; i++) if (mats[i]->data && !mats[i]->gpu) mt->h16 = false;
        if (repacked) fprintf(stderr, "[metal] repacked %.2f GB of 4-bit weights into NAX tiles in %.1fs\n", repacked / 1e9, now_secs() - t0);
        // every weight region as a no-copy buffer (page-aligned base, length rounded to 16 KB)
        for (int i = 0; i < m->n_regions; i++) {
            mt->wbase[i] = m->regions[i].base;
            mt->wlen[i] = (m->regions[i].len + 16383) / 16384 * 16384;
            mt->wbuf[i] = [mt->dev newBufferWithBytesNoCopy:(void *)mt->wbase[i] length:mt->wlen[i] options:MTLResourceStorageModeShared deallocator:nil];
            if (!mt->wbuf[i]) { fprintf(stderr, "metal: could not wrap weight region %d (%zu bytes)\n", i, mt->wlen[i]); exit(1); }
        }
        mt->n_w = m->n_regions;
        mt->invf = [mt->dev newBufferWithBytes:m->inv_freq length:sizeof(float) * m->c.hd / 2 options:MTLResourceStorageModeShared];
        backend *b = calloc(1, sizeof *b);
        b->name = "metal";
        b->kv_elem = 2;
        b->alloc_kv = metal_alloc;
        b->forward = metal_forward;
        b->attach = metal_attach;
        b->argmax_ok = true;
        b->draft = metal_draft;
        b->destroy = metal_destroy;
        b->impl = (__bridge_retained void *)mt;
        return b;
    }
}
