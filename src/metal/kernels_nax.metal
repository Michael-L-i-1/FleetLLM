// Affine-quantized GEMM on the M5 neural accelerators (Metal 4 tensor ops),
// compiled only on GPUs that support them.
//
// The tile loop follows MLX's qmm_t_nax_tgp_impl (quantized_nax.h; MIT,
// Copyright © 2023-2025 Apple Inc.) over the vendored steel/gemm/nax.h tiles.
// Differences: weights are pre-tiled at load (contiguous 64x64 blocks with
// inline scales/biases instead of MLX's row-major codes + separate scale
// arrays), 4-bit codes decode with the fp16 magic-number trick into 16-byte
// threadgroup stores, the next block is prefetched into registers during the
// NAX math, K can be split across threadgroups, activations are f16 padded to
// whole tiles, and outputs are f32 with a caller-given row stride.
//
// Instantiations are appended at runtime for the (scale type, group, bits)
// combinations the loaded model uses: see metal.m.
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

using namespace metal;
using namespace mlx::steel;

// K in-features, N out-features (weight rows), M tokens; grid.z splits K and
// split > 1 writes partials to y + z * part_stride (row stride ldy)
struct AtArgs { int K, N, M, ldy, split, part_stride; };

// ---------------------------------------------------------------------------
// Tiled affine GEMM. Weights are repacked at load into blocks of 64 output
// rows x 64 K-values, one block per (row tile, K tile), contiguous in K order:
//   [row tile][k tile] { codes: 64 rows x (64*bits/8) bytes, row-major
//                        params: 64 rows x GPT x (scale, bias) as S }
// (GPT = groups per K tile = max(1, 64 / group); a group wider than the tile
// repeats its params in each tile). A threadgroup streams one row tile's
// blocks front to back, so every K step is a single contiguous read.
// ---------------------------------------------------------------------------
template <typename S, int group_size, int bits>
struct TiledBlock {
    STEEL_CONST int TR = 64, TK = 64;
    STEEL_CONST int GPT = group_size >= TK ? 1 : TK / group_size;
    STEEL_CONST int code_bytes = TR * TK * bits / 8;
    STEEL_CONST int bytes = code_bytes + TR * GPT * 2 * sizeof(S);
};

// y[t*ldy + n] = sum_z part[z][t][n] (split-K reduction; part rows have stride N)
kernel void split_sum(device const float* part [[buffer(0)]], device float* y [[buffer(1)]], constant int4& a [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const int N = a.x, M = a.y, split = a.z, ldy = a.w;
    if (int(gid.x) >= N || int(gid.y) >= M) return;
    float s = 0.0f;
    for (int z = 0; z < split; z++) s += part[((size_t)z * M + gid.y) * N + gid.x];
    y[(size_t)gid.y * ldy + gid.x] = s;
}

// ---------------------------------------------------------------------------
// Mat-vec over the tiled layout for 1-4 tokens (no NAX, no threadgroup memory).
// Each simdgroup owns 8 rows of a row tile: lane l takes row l/4 and 16 of the
// tile's 64 K values ((l%4)*16), so a K step reads 256 contiguous bytes; the
// 4 lanes of a row are reduced with shuffles at the end. 2 simdgroups per
// threadgroup; grid.x = row tiles * 4, grid.y = token groups of NT, grid.z
// splits K (as in aqmt).
// ---------------------------------------------------------------------------
// HALF_DOT: the 16-value partial dot products run in f16 (needed to stay
// bandwidth-bound at 3-4 tokens); otherwise f32.
template <typename S, const int group_size, const int NT, const bool HALF_DOT = (NT > 2)>
[[kernel]] void tmv(const device uint8_t* w [[buffer(0)]], const device half* x [[buffer(3)]], device float* y [[buffer(4)]],
                    const constant AtArgs& a [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    using TB = TiledBlock<S, group_size, 4>;
    const int K = a.K, n_kt_all = K / 64;
    const int kt_begin = tg.z * n_kt_all / a.split, n_kt = (tg.z + 1) * n_kt_all / a.split - kt_begin;
    const int rt = tg.x / 4, r_in = ((tg.x % 4) * 2 + sg) * 8 + lane / 4, kq = (lane % 4) * 16;
    const int t0 = tg.y * NT;
    const device uint8_t* blk = w + ((size_t)rt * n_kt_all + kt_begin) * TB::bytes;
    const device uint2* cw = (const device uint2*)(blk + r_in * 32 + kq / 2);
    const device S* pp = (const device S*)(blk + TB::code_bytes) + (r_in * TB::GPT + (TB::GPT > 1 ? kq / group_size : 0)) * 2;
    constexpr int bstride_w = TB::bytes / 8, bstride_p = TB::bytes / sizeof(S);
    const device half* xp[NT];
    for (int t = 0; t < NT; t++) xp[t] = x + (size_t)min(t0 + t, a.M - 1) * K + kt_begin * 64 + kq;
    float acc[NT];
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    // HALF_DOT: codes decode to q - 8 (1032 = 1024 + 8) and the bias absorbs 8 * scale
    const half2 m2 = half2(HALF_DOT ? 1032.0h : 1024.0h);
    // UK K tiles per iteration with all loads issued first (memory-level parallelism)
    constexpr int UK = NT == 1 ? 4 : NT == 2 ? 2 : 1;
    for (int kt0 = 0; kt0 < n_kt; kt0 += UK) {
        uint2 v[UK];
        float s[UK], b[UK];
        for (int u = 0; u < UK; u++) {
            const int kt = min(kt0 + u, n_kt - 1);
            v[u] = cw[kt * bstride_w];
            s[u] = float(pp[kt * bstride_p]);
            b[u] = float(pp[kt * bstride_p + 1]) + (HALF_DOT ? 8.0f * s[u] : 0.0f);
        }
        for (int u = 0; u < UK && kt0 + u < n_kt; u++) {
            const int kt = kt0 + u;
            // codes: word .x holds values 0..7 (nibble i = value i), .y values 8..15
            half4 q[4];
            {
                const uint vx = v[u].x, vy = v[u].y;
                half2 a0 = as_type<half2>((vx & 0x000F000Fu) | 0x64006400u) - m2, a1 = as_type<half2>(((vx >> 4) & 0x000F000Fu) | 0x64006400u) - m2;
                half2 a2 = as_type<half2>(((vx >> 8) & 0x000F000Fu) | 0x64006400u) - m2, a3 = as_type<half2>(((vx >> 12) & 0x000F000Fu) | 0x64006400u) - m2;
                q[0] = half4(a0.x, a1.x, a2.x, a3.x);
                q[1] = half4(a0.y, a1.y, a2.y, a3.y);
                half2 b0 = as_type<half2>((vy & 0x000F000Fu) | 0x64006400u) - m2, b1 = as_type<half2>(((vy >> 4) & 0x000F000Fu) | 0x64006400u) - m2;
                half2 b2 = as_type<half2>(((vy >> 8) & 0x000F000Fu) | 0x64006400u) - m2, b3 = as_type<half2>(((vy >> 12) & 0x000F000Fu) | 0x64006400u) - m2;
                q[2] = half4(b0.x, b1.x, b2.x, b3.x);
                q[3] = half4(b0.y, b1.y, b2.y, b3.y);
            }
            for (int t = 0; t < NT; t++) {
                const device half4* xv = (const device half4*)(xp[t] + kt * 64);
                float d, sx;
                if (HALF_DOT) { // codes are centered (q - 8), so the f16 partial sums stay small
                    half4 dh = half4(0.0h), sh = half4(0.0h);
                    for (int i = 0; i < 4; i++) { half4 xh = xv[i]; dh = fma(xh, q[i], dh); sh += xh; }
                    d = float(dh.x) + float(dh.y) + float(dh.z) + float(dh.w);
                    sx = float(sh.x) + float(sh.y) + float(sh.z) + float(sh.w);
                } else {
                    d = 0.0f;
                    sx = 0.0f;
                    for (int i = 0; i < 4; i++) {
                        float4 xf = float4(xv[i]);
                        d += dot(xf, float4(q[i]));
                        sx += xf.x + xf.y + xf.z + xf.w;
                    }
                }
                acc[t] += s[u] * d + b[u] * sx;
            }
        }
    }
    for (int t = 0; t < NT; t++) {
        float r = acc[t];
        r += simd_shuffle_xor(r, 1);
        r += simd_shuffle_xor(r, 2);
        const int row = rt * 64 + r_in;
        if (lane % 4 == 0 && row < a.N && t0 + t < a.M) y[(a.split > 1 ? (size_t)tg.z * a.part_stride : 0) + (size_t)(t0 + t) * a.ldy + row] = r;
    }
}

// ---------------------------------------------------------------------------
// Single-simdgroup variant: a threadgroup is one simdgroup owning 32 rows of a
// row tile (half a block) and BM tokens, so there is no cross-simdgroup
// barrier; lane = weight row, 64 values decoded per lane per K step.
// grid.x = row tiles * 2, grid.y = token tiles of BM, grid.z = K split.
// ---------------------------------------------------------------------------
template <typename S, typename XT, const int group_size, const int bits, const int BM = 16>
[[kernel]] void aqms(const device uint8_t* w [[buffer(0)]], const device XT* x [[buffer(3)]], device float* y [[buffer(4)]],
                     const constant AtArgs& a [[buffer(5)]], uint3 tid [[threadgroup_position_in_grid]],
                     uint simd_lid [[thread_index_in_simdgroup]]) {
    using TB = TiledBlock<S, group_size, bits>;
    static_assert(bits == 4, "tiled path: 4-bit only for now");
    constexpr int BK = 64, BKP = BK + 8, NG = BK > group_size ? BK / group_size : 1;
    threadgroup half Ws[32 * BKP];
    const int K = a.K, N = a.N, M = a.M;
    const int n_kt_all = K / BK;
    const int kt0 = tid.z * n_kt_all / a.split, n_kt = (tid.z + 1) * n_kt_all / a.split - kt0;
    const int rt = tid.x / 2, half_ = tid.x % 2;
    const int y_col = rt * 64 + half_ * 32, y_row = tid.y * BM;
    const short wr = half_ * 32 + simd_lid; // row within the 64-row block
    const device uint8_t* blk = w + ((size_t)rt * n_kt_all + kt0) * TB::bytes;
    threadgroup half* dst = Ws + simd_lid * BKP;

    x += y_row * static_cast<int64_t>(K) + kt0 * BK;
    y += (a.split > 1 ? (size_t)tid.z * a.part_stride : 0) + y_row * static_cast<int64_t>(a.ldy) + y_col;
    const short sgp_sm = max(0, min(BM, M - y_row)), sgp_sn = min(32, N - y_col);

    NAXTile<float, BM / 16, 2> Dtile;
    Dtile.clear();
    uint4 raw[2];
    S ps[NG], pb[NG];
    auto fetch = [&](const device uint8_t* b) {
        const device uint4* cw = (const device uint4*)(b + wr * 32);
        raw[0] = cw[0];
        raw[1] = cw[1];
        const device S* pp = (const device S*)(b + TB::code_bytes) + wr * TB::GPT * 2;
        for (short g = 0; g < NG; g++) { ps[g] = pp[2 * g]; pb[g] = pp[2 * g + 1]; }
    };
    fetch(blk);
    const half2 m2 = half2(1024.0h);
    for (int kt = 0; kt < n_kt; kt++) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 8; i++) {
            const short g = NG > 1 ? i * 8 / group_size : 0;
            const half2 s2 = half2(half(ps[g])), b2 = half2(half(pb[g]));
            const uint v = raw[i / 4][i % 4];
            half2 p0 = fma(as_type<half2>((v & 0x000F000Fu) | 0x64006400u) - m2, s2, b2);
            half2 p1 = fma(as_type<half2>(((v >> 4) & 0x000F000Fu) | 0x64006400u) - m2, s2, b2);
            half2 p2 = fma(as_type<half2>(((v >> 8) & 0x000F000Fu) | 0x64006400u) - m2, s2, b2);
            half2 p3 = fma(as_type<half2>(((v >> 12) & 0x000F000Fu) | 0x64006400u) - m2, s2, b2);
            half4 lo = half4(p0.x, p1.x, p2.x, p3.x), hi = half4(p0.y, p1.y, p2.y, p3.y);
            *((threadgroup uint4*)(dst + 8 * i)) = uint4(as_type<uint2>(lo), as_type<uint2>(hi));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        blk += TB::bytes;
        if (kt + 1 < n_kt) fetch(blk);
        STEEL_PRAGMA_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += 32) {
            NAXTile<half, BM / 16, 2> Atile;
            NAXTile<half, 2, 2> Btile;
            Atile.load(x + kk1, K);
            Btile.template load<half, BKP, 1>(Ws + kk1);
            tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
        }
        x += BK;
    }
    if (sgp_sm == BM && sgp_sn == 32) Dtile.store(y, a.ldy);
    else if (sgp_sm > 0 && sgp_sn > 0) Dtile.store_safe(y, a.ldy, short2(sgp_sn, sgp_sm));
}
