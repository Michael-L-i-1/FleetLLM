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

// ---------------------------------------------------------------------------
// Prefill attention on the neural accelerators. Rows that come in runs of
// consecutive positions of one sequence (prompt chunks, tool results) are
// processed 64 at a time per query head: 4 simdgroups x 16 rows, K/V read from
// the pages straight into registers (no threadgroup staging), S = Q K^T and
// O += P V as MPP 16x32x16 matmuls, online softmax in the exp2 domain. Long
// contexts split into fixed spans of nsp.x positions (grid z; the span grid
// depends only on positions, so a row's result does not depend on its batch)
// whose partials attn_reduce merges; with one span the normalized (optionally
// gated) output is written directly, bit-identical to the merge. The lane layout, matmul wrapper, row reductions and tile loop
// follow vllm-metal's paged_attention_nax (pagedattention_nax.metal,
// Apache-2.0), itself adapted from MLX steel_attention_nax (MIT, Copyright ©
// 2025 Apple Inc.). Differences: our page layout (16-token pages, per-head
// position-contiguous K then V), int8 K/V with per-token f16 scales applied as
// the fragments load, f32 queries, and per-row visible ranges [start, ctx_end]
// (causal and sliding-window masks alike) from RowInfo.
struct AttnArgs { uint nh, nkv, hd; float scale; ulong page_elems, layer_off; uint page, gate, qs, gs, half_out, layer; ulong scale_off; };
struct RowInfo { uint ctx_end, start, tab, split, part; };
#define NAX_NEG (-3.402823466e+38f) // finite mask sentinel

template <typename T>
using frag8 = metal::vec<T, 8>;

// C[16x32] (two 16x16 fragments) += A[16x16] B; TB: B fragments are rows 0-15 /
// 16-31 of a 32 x 16 tile (K^T), else columns 0-15 / 16-31 of a 16 x 32 tile (V)
template <typename CT, typename AT_, typename BT, bool TB>
METAL_FUNC void nax_mma(thread frag8<CT>& c0, thread frag8<CT>& c1, const thread frag8<AT_>& a, const thread frag8<BT>& b0, const thread frag8<BT>& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, TB, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ta = op.template get_left_input_cooperative_tensor<AT_, BT, CT>();
    auto tb = op.template get_right_input_cooperative_tensor<AT_, BT, CT>();
    auto tc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ta)>, metal::remove_addrspace_t<decltype(tb)>, CT>();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 8; i++) { ta[i] = a[i]; tb[i] = b0[i]; tb[8 + i] = b1[i]; tc[i] = c0[i]; tc[8 + i] = c1[i]; }
    op.run(ta, tb, tc);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 8; i++) { c0[i] = tc[i]; c1[i] = tc[8 + i]; }
}

// fold each of the lane's 2 rows across the 4 lanes holding it (lane bits 0 and 3)
template <bool MAX>
METAL_FUNC void nax_rows(const thread frag8<float>& f, thread float* v) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        float r = MAX ? max(max(f[i * 4], f[i * 4 + 1]), max(f[i * 4 + 2], f[i * 4 + 3])) : (f[i * 4] + f[i * 4 + 1]) + (f[i * 4 + 2] + f[i * 4 + 3]);
        float o = simd_shuffle_xor(r, ushort(1));
        r = MAX ? max(r, o) : r + o;
        o = simd_shuffle_xor(r, ushort(8));
        r = MAX ? max(r, o) : r + o;
        v[i] = MAX ? max(v[i], r) : v[i] + r;
    }
}

// 4 consecutive K or V values of one token: f16, or int8 times the token's scale.
// int8 -> f16 with the magic-number trick (byte + 128 in the mantissa of 1024.0h,
// as for the 4-bit weights): a hardware char -> half convert per value cost more
// than the matmuls.
template <bool Q8>
METAL_FUNC half4 nax_kv4(device const half* pool, ulong at, half s) {
    if (Q8) {
        const uint w = *(device const uint*)((device const char*)pool + at) ^ 0x80808080u;
        const half2 lo = as_type<half2>((w & 0x00FF00FFu) | 0x64006400u), hi = as_type<half2>(((w >> 8) & 0x00FF00FFu) | 0x64006400u);
        const half2 e = (lo - 1152.0h) * s, o = (hi - 1152.0h) * s; // bytes 0, 2 and 1, 3 (the subtraction is exact)
        return half4(e.x, o.x, e.y, o.y);
    }
    return *(device const half4*)(pool + at);
}

template <int HD, bool Q8>
[[kernel, max_total_threads_per_threadgroup(128)]] void attn_nax(
    device const float* q [[buffer(0)]], device const half* pool [[buffer(1)]], device const RowInfo* ri [[buffer(2)]],
    device const uint* tab [[buffer(3)]], device float* out [[buffer(4)]], constant AttnArgs& a [[buffer(5)]],
    device const uint2* groups [[buffer(6)]], device const float* g [[buffer(7)]], device float* part [[buffer(8)]],
    device const uint* slot [[buffer(9)]], constant uint2& nsp [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int TD = HD / 16, TK = 32, KS = HD + 8;
    threadgroup half kvt[2 * TK * KS]; // the tile (int8: dequantized) once for all 4 simdgroups
    const uint2 grp = groups[tg.x]; // {first row, rows <= 64}
    const uint h = tg.y, kvh = h / (a.nh / a.nkv);
    const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
    const int nrow = (int)grp.y, r0 = sg * 16;
    const RowInfo first = ri[grp.x], last = ri[grp.x + nrow - 1];
    // this lane's two query rows and their visible ranges (padding rows see nothing)
    bool ok[2];
    uint row[2], vs[2], ve[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const int rl = r0 + fm + i * 8;
        ok[i] = rl < nrow;
        row[i] = grp.x + (ok[i] ? rl : 0);
        const RowInfo r = ri[row[i]];
        vs[i] = ok[i] ? r.start : 1;
        ve[i] = ok[i] ? r.ctx_end : 0;
    }
    // tiles every row of this simdgroup sees in full need no mask
    const bool sg_full = r0 + 16 <= nrow;
    const uint sg_lo = sg_full ? ri[grp.x + r0 + 15].start : 0xffffffffu, sg_hi = sg_full ? ri[grp.x + r0].ctx_end : 0;
    frag8<half> Q[TD];
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < TD; d++)
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            float4 v = ok[i] ? *(device const float4*)(q + (ulong)row[i] * a.qs + h * HD + d * 16 + fn) : float4(0.0f);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) Q[d][i * 4 + j] = half(v[j]);
        }
    float mx[2] = { NAX_NEG, NAX_NEG }, sum[2] = { 0.0f, 0.0f };
    frag8<float> O[TD];
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < TD; d++) O[d] = frag8<float>(0.0f);
    const float sl2 = a.scale * M_LOG2E_F;
    const ulong vofs = (ulong)a.nkv * a.page * HD, head_off = a.layer_off + (ulong)kvh * a.page * HD;
    const ulong sc_off = a.scale_off + (((ulong)a.layer * 2) * a.nkv + kvh) * a.page * 2, sc_v = (ulong)a.nkv * a.page * 2;
    const uint s0 = tg.z * nsp.x, lo = max(s0, first.start / TK * TK), end = min(last.ctx_end, s0 + nsp.x - 1);
    for (uint t0 = lo; t0 <= end; t0 += TK) {
        { // 4 threads per token: the tile's K and V rows (int8: dequantized) into threadgroup memory
            constexpr int CH = HD / 16; // char4 chunks per thread
            threadgroup_barrier(mem_flags::mem_threadgroup); // the previous tile's readers are done
            const uint tk = tid / 4, d0 = (tid % 4) * (HD / 4), p = t0 + tk;
            threadgroup half4* kr = (threadgroup half4*)(kvt + tk * KS + d0);
            threadgroup half4* vr = (threadgroup half4*)(kvt + (TK + tk) * KS + d0);
            if (p <= end && p >= first.start) { // (earlier ones may sit on released sliding-window pages)
                const ulong page = tab[last.tab + p / a.page];
                const ulong base = page * a.page_elems + head_off + (ulong)(p % a.page) * HD + d0;
                device const char* pb = (device const char*)pool + page * a.page_elems + sc_off + (p % a.page) * 2;
                const half sk = Q8 ? *(device const half*)pb : 1.0h, sv = Q8 ? *(device const half*)(pb + sc_v) : 1.0h;
                STEEL_PRAGMA_UNROLL
                for (int c = 0; c < CH; c++) { kr[c] = nax_kv4<Q8>(pool, base + c * 4, sk); vr[c] = nax_kv4<Q8>(pool, base + vofs + c * 4, sv); }
            } else {
                STEEL_PRAGMA_UNROLL
                for (int c = 0; c < CH; c++) kr[c] = vr[c] = half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        frag8<float> S0 = frag8<float>(0.0f), S1 = frag8<float>(0.0f);
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < TD; d++) {
            frag8<half> K0, K1;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const half4 k0 = *(threadgroup const half4*)(kvt + (i * 8 + fm) * KS + d * 16 + fn);
                const half4 k1 = *(threadgroup const half4*)(kvt + ((2 + i) * 8 + fm) * KS + d * 16 + fn);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) { K0[i * 4 + j] = k0[j]; K1[i * 4 + j] = k1[j]; }
            }
            nax_mma<float, half, half, true>(S0, S1, Q[d], K0, K1);
        }
        if (sg_full && t0 >= sg_lo && t0 + TK - 1 <= sg_hi) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; e++) { S0[e] *= sl2; S1[e] *= sl2; }
        } else {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++)
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    const uint p0 = t0 + fn + j, p1 = p0 + 16;
                    S0[i * 4 + j] = p0 >= vs[i] && p0 <= ve[i] ? S0[i * 4 + j] * sl2 : NAX_NEG;
                    S1[i * 4 + j] = p1 >= vs[i] && p1 <= ve[i] ? S1[i * 4 + j] * sl2 : NAX_NEG;
                }
        }
        float nm[2] = { mx[0], mx[1] };
        nax_rows<true>(S0, nm);
        nax_rows<true>(S1, nm);
        float fac[2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) { fac[i] = fast::exp2(mx[i] - nm[i]); mx[i] = nm[i]; }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++)
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                const float s0 = S0[i * 4 + j], s1 = S1[i * 4 + j];
                S0[i * 4 + j] = s0 == NAX_NEG ? 0.0f : fast::exp2(s0 - nm[i]);
                S1[i * 4 + j] = s1 == NAX_NEG ? 0.0f : fast::exp2(s1 - nm[i]);
            }
        float rs[2] = { 0.0f, 0.0f };
        nax_rows<false>(S0, rs);
        nax_rows<false>(S1, rs);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) sum[i] = sum[i] * fac[i] + rs[i];
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < TD; d++)
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; e++) O[d][e] *= fac[e / 4];
        simdgroup_barrier(mem_flags::mem_none);
        // O += P V: V fragment rows are tokens (quarters 2f, 2f+1), columns 16 dims
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < TD; d += 2) {
            if (TD >= 8 && d == TD / 2) threadgroup_barrier(mem_flags::mem_none); // scheduling point, as in MLX
            STEEL_PRAGMA_UNROLL
            for (short f = 0; f < 2; f++) {
                frag8<half> V0, V1;
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const short qq = 2 * f + i;
                    threadgroup const half* vt = kvt + (TK + qq * 8 + fm) * KS + fn;
                    const half4 v0 = *(threadgroup const half4*)(vt + d * 16), v1 = *(threadgroup const half4*)(vt + (d + 1) * 16);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) { V0[i * 4 + j] = v0[j]; V1[i * 4 + j] = v1[j]; }
                }
                nax_mma<float, float, half, false>(O[d], O[d + 1], f == 0 ? S0 : S1, V0, V1);
            }
        }
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        if (!ok[i]) continue;
        if (nsp.y > 1) { // partial for attn_reduce: natural-log max, sum, unnormalized O
            device float* dst = part + (((ulong)slot[row[i]] * a.nh + h) * nsp.y + tg.z) * (HD + 2);
            if (fn == 0) { dst[0] = mx[i] * M_LN2_F; dst[1] = sum[i]; }
            STEEL_PRAGMA_UNROLL
            for (short d = 0; d < TD; d++)
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) dst[2 + d * 16 + fn + j] = O[d][i * 4 + j];
            continue;
        }
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < TD; d++)
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                const uint col = h * HD + d * 16 + fn + j;
                float v = O[d][i * 4 + j] / sum[i]; // as attn_reduce: o / L
                if (a.gate) v *= 1.0f / (1.0f + exp(-g[(ulong)row[i] * a.gs + col]));
                const ulong idx = (ulong)row[i] * a.nh * HD + col;
                if (a.half_out) ((device half*)out)[idx] = half(v);
                else out[idx] = v;
            }
    }
}

// Decode / verify attention over fixed spans of sp.x positions, shared across
// sequences. A threadgroup takes one span for up to 4 row groups (a decode row,
// its drafts and the G query heads of one KV head per group, 16 query slots per
// simdgroup) whose sequences hold the same pages there: a common system prompt
// and tool definitions are read and dequantized once for every agent instead of
// once per agent. Partials are per (row, head, span) for attn_reduce; span
// boundaries depend only on positions and masked tiles add exactly nothing, so a
// row's result does not depend on what it was grouped with. Tile loop as attn_nax.
struct SpanItem { uint span, n, g[4]; };
template <int HD, bool Q8>
[[kernel, max_total_threads_per_threadgroup(128)]] void attn_naxs(
    device const float* q [[buffer(0)]], device const half* pool [[buffer(1)]], device const RowInfo* ri [[buffer(2)]],
    device const uint* tab [[buffer(3)]], device float* part [[buffer(4)]], constant AttnArgs& a [[buffer(5)]],
    constant uint2& sp [[buffer(6)]], device const uint2* groups [[buffer(7)]], device const SpanItem* items [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr int TD = HD / 16, TK = 32, KS = HD + 8;
    threadgroup half kvt[2 * TK * KS]; // the tile (int8: dequantized) once for all 4 simdgroups
    const SpanItem it = items[tg.x];
    const uint kvh = tg.y, G = a.nh / a.nkv;
    const short qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
    const bool act = sg < it.n;
    const uint2 grp = groups[it.g[act ? sg : 0]]; // {first row, rows <= 16 / G} of one sequence
    // the item's groups hold the same pages over the span: tiles come from the first
    // group's page table, up to the furthest position any of them attends to
    uint end = 0, vis = 0xffffffffu; // furthest position attended, first visible one
    for (uint k = 0; k < it.n; k++) {
        const uint2 gk = groups[it.g[k]];
        end = max(end, ri[gk.x + gk.y - 1].ctx_end);
        vis = min(vis, ri[gk.x].start);
    }
    const uint tb = ri[groups[it.g[0]].x].tab, s0 = max(it.span * sp.x, vis / TK * TK);
    end = min(end, it.span * sp.x + sp.x - 1);
    // this lane's two (row, head) query slots and their visible ranges (padding sees nothing)
    bool ok[2];
    uint row[2], hq[2], vs[2], ve[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const uint slot = fm + i * 8, r = slot / G;
        ok[i] = act && r < grp.y;
        row[i] = grp.x + (ok[i] ? r : 0);
        hq[i] = kvh * G + slot % G;
        const RowInfo rr = ri[row[i]];
        vs[i] = ok[i] ? rr.start : 1;
        ve[i] = ok[i] ? rr.ctx_end : 0;
    }
    frag8<half> Q[TD];
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < TD; d++)
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            float4 v = ok[i] ? *(device const float4*)(q + (ulong)row[i] * a.qs + hq[i] * HD + d * 16 + fn) : float4(0.0f);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) Q[d][i * 4 + j] = half(v[j]);
        }
    float mx[2] = { NAX_NEG, NAX_NEG }, sum[2] = { 0.0f, 0.0f };
    frag8<float> O[TD];
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < TD; d++) O[d] = frag8<float>(0.0f);
    const float sl2 = a.scale * M_LOG2E_F;
    const ulong vofs = (ulong)a.nkv * a.page * HD, head_off = a.layer_off + (ulong)kvh * a.page * HD;
    const ulong sc_off = a.scale_off + (((ulong)a.layer * 2) * a.nkv + kvh) * a.page * 2, sc_v = (ulong)a.nkv * a.page * 2;
    for (uint t0 = s0; t0 <= end; t0 += TK) {
        const uint vis_lo = vis; // positions before it may sit on released sliding-window pages
        { // 4 threads per token: the tile's K and V rows (int8: dequantized) into threadgroup memory
            constexpr int CH = HD / 16; // char4 chunks per thread
            threadgroup_barrier(mem_flags::mem_threadgroup); // the previous tile's readers are done
            const uint tk = tid / 4, d0 = (tid % 4) * (HD / 4), p = t0 + tk;
            threadgroup half4* kr = (threadgroup half4*)(kvt + tk * KS + d0);
            threadgroup half4* vr = (threadgroup half4*)(kvt + (TK + tk) * KS + d0);
            if (p <= end && p >= vis_lo) {
                const ulong page = tab[tb + p / a.page];
                const ulong base = page * a.page_elems + head_off + (ulong)(p % a.page) * HD + d0;
                device const char* pb = (device const char*)pool + page * a.page_elems + sc_off + (p % a.page) * 2;
                const half sk = Q8 ? *(device const half*)pb : 1.0h, sv = Q8 ? *(device const half*)(pb + sc_v) : 1.0h;
                STEEL_PRAGMA_UNROLL
                for (int c = 0; c < CH; c++) { kr[c] = nax_kv4<Q8>(pool, base + c * 4, sk); vr[c] = nax_kv4<Q8>(pool, base + vofs + c * 4, sv); }
            } else {
                STEEL_PRAGMA_UNROLL
                for (int c = 0; c < CH; c++) kr[c] = vr[c] = half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        frag8<float> S0 = frag8<float>(0.0f), S1 = frag8<float>(0.0f);
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < TD; d++) {
            frag8<half> K0, K1;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const half4 k0 = *(threadgroup const half4*)(kvt + (i * 8 + fm) * KS + d * 16 + fn);
                const half4 k1 = *(threadgroup const half4*)(kvt + ((2 + i) * 8 + fm) * KS + d * 16 + fn);
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) { K0[i * 4 + j] = k0[j]; K1[i * 4 + j] = k1[j]; }
            }
            nax_mma<float, half, half, true>(S0, S1, Q[d], K0, K1);
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++)
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                const uint p0 = t0 + fn + j, p1 = p0 + 16;
                S0[i * 4 + j] = p0 >= vs[i] && p0 <= ve[i] ? S0[i * 4 + j] * sl2 : NAX_NEG;
                S1[i * 4 + j] = p1 >= vs[i] && p1 <= ve[i] ? S1[i * 4 + j] * sl2 : NAX_NEG;
            }
        float nm[2] = { mx[0], mx[1] };
        nax_rows<true>(S0, nm);
        nax_rows<true>(S1, nm);
        float fac[2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) { fac[i] = fast::exp2(mx[i] - nm[i]); mx[i] = nm[i]; }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++)
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                const float s0 = S0[i * 4 + j], s1 = S1[i * 4 + j];
                S0[i * 4 + j] = s0 == NAX_NEG ? 0.0f : fast::exp2(s0 - nm[i]);
                S1[i * 4 + j] = s1 == NAX_NEG ? 0.0f : fast::exp2(s1 - nm[i]);
            }
        float rs[2] = { 0.0f, 0.0f };
        nax_rows<false>(S0, rs);
        nax_rows<false>(S1, rs);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) sum[i] = sum[i] * fac[i] + rs[i];
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < TD; d++)
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; e++) O[d][e] *= fac[e / 4];
        simdgroup_barrier(mem_flags::mem_none);
        // O += P V: V fragment rows are tokens (quarters 2f, 2f+1), columns 16 dims
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < TD; d += 2) {
            if (TD >= 8 && d == TD / 2) threadgroup_barrier(mem_flags::mem_none); // scheduling point, as in MLX
            STEEL_PRAGMA_UNROLL
            for (short f = 0; f < 2; f++) {
                frag8<half> V0, V1;
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const short qq = 2 * f + i;
                    threadgroup const half* vt = kvt + (TK + qq * 8 + fm) * KS + fn;
                    const half4 v0 = *(threadgroup const half4*)(vt + d * 16), v1 = *(threadgroup const half4*)(vt + (d + 1) * 16);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) { V0[i * 4 + j] = v0[j]; V1[i * 4 + j] = v1[j]; }
                }
                nax_mma<float, float, half, false>(O[d], O[d + 1], f == 0 ? S0 : S1, V0, V1);
            }
        }
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) { // partial for attn_reduce: natural-log max, sum, unnormalized O
        if (!ok[i]) continue;
        device float* dst = part + (((ulong)ri[row[i]].part * a.nh + hq[i]) * sp.y + it.span) * (HD + 2);
        if (fn == 0) { dst[0] = mx[i] * M_LN2_F; dst[1] = sum[i]; }
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < TD; d++)
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) dst[2 + d * 16 + fn + j] = O[d][i * 4 + j];
    }
}

template [[host_name("attn_naxs_64")]] [[kernel]] decltype(attn_naxs<64, false>) attn_naxs<64, false>;
template [[host_name("attn_naxs_128")]] [[kernel]] decltype(attn_naxs<128, false>) attn_naxs<128, false>;
template [[host_name("attn_naxs8_64")]] [[kernel]] decltype(attn_naxs<64, true>) attn_naxs<64, true>;
template [[host_name("attn_naxs8_128")]] [[kernel]] decltype(attn_naxs<128, true>) attn_naxs<128, true>;
template [[host_name("attn_nax_64")]] [[kernel]] decltype(attn_nax<64, false>) attn_nax<64, false>;
template [[host_name("attn_nax_128")]] [[kernel]] decltype(attn_nax<128, false>) attn_nax<128, false>;
template [[host_name("attn_nax8_64")]] [[kernel]] decltype(attn_nax<64, true>) attn_nax<64, true>;
template [[host_name("attn_nax8_128")]] [[kernel]] decltype(attn_nax<128, true>) attn_nax<128, true>;
