// cinfer Metal kernels (compiled at runtime from this embedded source).
//
// Weights are read in place from the mmap'd GGUF / safetensors (zero-copy
// buffers); K-quant block layouts match ggml, affine layouts match MLX. Activations are f32; KV cache is f16 in pages of
// PAGE tokens laid out [page][layer][k|v][kv_head][PAGE][head_dim].
//
// Matmul strategy (as in vllm-metal): batches <= 8 rows use a fused mat-vec on
// the packed blocks; larger batches dequantize the weight once into an f16
// scratch matrix and run a dense simdgroup-matrix GEMM.
#include <metal_stdlib>
using namespace metal;

#define QK 256

static inline float hf(device const uchar* p) { return float(*((device const half*)p)); }
static inline uint ld4(device const uchar* p) { return *((device const uint*)p); }
static inline uint ld2(device const uchar* p) {
    device const ushort* s = (device const ushort*)p;
    return uint(s[0]) | (uint(s[1]) << 16);
}
static inline float4 u8x4(uint v) { return float4(as_type<uchar4>(v)); }

static inline uchar2 smk4(int j, device const uchar* q) {
    if (j < 4) return uchar2(q[j] & 63, q[j + 4] & 63);
    return uchar2((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

// Sub-block j (16 values) of a 256-value group starting at p, as raw codes q
// (4 x float4) with value = dl * q - ml. Uses 32-bit loads and byte-lane masks.
// Q2_K/Q4_K/Q5_K blocks are 4-byte aligned; Q3_K/Q6_K/Q8_0/Q4_0 only 2-byte.
// T: 2 Q2_K, 3 Q3_K, 4 Q4_K, 5 Q5_K, 6 Q6_K, 8 Q8_0 (8 blocks), 40 Q4_0 (8 blocks)
template <int T>
static inline void deqq(device const uchar* p, int j, thread float4* q, thread float& dl, thread float& ml) {
    if (T == 2) {
        uchar sc = p[j];
        dl = hf(p + 80) * float(sc & 0xF);
        ml = hf(p + 82) * float(sc >> 4);
        int jj = j & 7, shift = 2 * (jj >> 1);
        device const uchar* b = p + 16 + (j >> 3) * 32 + (jj & 1) * 16;
        for (int k = 0; k < 4; k++) q[k] = u8x4((ld4(b + 4 * k) >> shift) & 0x03030303u);
    } else if (T == 3) {
        device const uchar* s = p + 96;
        int lo = j < 8 ? (s[j] & 0xF) : (s[j - 8] >> 4);
        int hi = (s[8 + (j & 3)] >> (2 * (j >> 2))) & 3;
        dl = hf(p + 108) * float((lo | (hi << 4)) - 32);
        ml = 4.0f * dl; // value = dl*(q - 4), q includes the high bit as +4
        int h = j >> 3, jj = j & 7, shift = 2 * (jj >> 1), l0 = (jj & 1) * 16, hbit = h * 4 + (jj >> 1);
        device const uchar* qs = p + 32 + h * 32 + l0;
        device const uchar* hm = p + l0;
        for (int k = 0; k < 4; k++)
            q[k] = u8x4(((ld2(qs + 4 * k) >> shift) & 0x03030303u) | (((ld2(hm + 4 * k) >> hbit) & 0x01010101u) << 2));
    } else if (T == 4 || T == 5) {
        uchar2 sm = smk4(j >> 1, p + 4);
        dl = hf(p) * float(sm.x);
        ml = hf(p + 2) * float(sm.y);
        int c = j >> 2, r = j & 3, l0 = (r & 1) * 16;
        device const uchar* b = p + (T == 4 ? 16 : 48) + c * 32 + l0;
        int nsh = (r >> 1) * 4;
        for (int k = 0; k < 4; k++) {
            uint v = (ld4(b + 4 * k) >> nsh) & 0x0F0F0F0Fu;
            if (T == 5) v |= ((ld4(p + 16 + l0 + 4 * k) >> (2 * c + (r >> 1))) & 0x01010101u) << 4;
            q[k] = u8x4(v);
        }
    } else if (T == 6) {
        dl = hf(p + 208) * float(as_type<char>(p[192 + j]));
        ml = 32.0f * dl;
        int n = j >> 3, jj = j & 7, quad = jj >> 1, l0 = (jj & 1) * 16;
        device const uchar* ql = p + 64 * n + (quad & 1) * 32 + l0;
        device const uchar* qh = p + 128 + 32 * n + l0;
        int lsh = (quad >> 1) * 4, hsh = 2 * quad;
        for (int k = 0; k < 4; k++)
            q[k] = u8x4(((ld2(ql + 4 * k) >> lsh) & 0x0F0F0F0Fu) | (((ld2(qh + 4 * k) >> hsh) & 0x03030303u) << 4));
    } else if (T == 8) {
        device const uchar* b = p + (j >> 1) * 34;
        dl = hf(b);
        ml = 0.0f;
        device const uchar* qq = b + 2 + (j & 1) * 16;
        for (int k = 0; k < 4; k++) q[k] = float4(as_type<char4>(ld2(qq + 4 * k)));
    } else { // Q4_0: 18-byte blocks of 32; sub-block j&1 selects low/high nibbles
        device const uchar* b = p + (j >> 1) * 18;
        dl = hf(b);
        ml = 8.0f * dl;
        int nsh = (j & 1) * 4;
        for (int k = 0; k < 4; k++) q[k] = u8x4((ld2(b + 2 + 4 * k) >> nsh) & 0x0F0F0F0Fu);
    }
}

// ---------------------------------------------------------------------------
// Fused mat-vec for small batches: Y[t][r] = W[r] . X[t] for NT tokens.
// 4 simdgroups x 2 rows each per threadgroup; lanes 0-15 / 16-31 take
// alternating blocks, lane&15 the sub-block. Codes are decoded once into
// registers and reused for all NT tokens; the min/offset term uses sum(x).
// ---------------------------------------------------------------------------
struct MvArgs { uint K, M, N, ldy, row_bytes, x_stride; };
#define MV_R 2
#define MV_SG 4

template <int T, int BB, int NT>
kernel void mv(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]], device float* Y [[buffer(2)]],
               constant MvArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
               ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg.x * MV_SG + sg) * MV_R;
    const uint t0 = tg.y * NT;
    if (row0 >= a.M) return;
    const uint nb = a.K / QK;
    float acc[MV_R][NT];
    for (int r = 0; r < MV_R; r++) for (int t = 0; t < NT; t++) acc[r][t] = 0.0f;
    const int j = lane & 15;
    for (uint b = lane >> 4; b < nb; b += 2) {
        float4 q[MV_R][4];
        float dl[MV_R], ml[MV_R];
        for (int r = 0; r < MV_R; r++) {
            if (row0 + r < a.M) deqq<T>(W + (row0 + r) * a.row_bytes + b * BB, j, q[r], dl[r], ml[r]);
            else { for (int k = 0; k < 4; k++) q[r][k] = float4(0.0f); dl[r] = 0.0f; ml[r] = 0.0f; }
        }
        for (int t = 0; t < NT; t++) {
            if (t0 + t < a.N) {
                device const float4* xp = (device const float4*)(X + (t0 + t) * a.x_stride + b * QK + j * 16);
                float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                float sx = dot(x0 + x1 + x2 + x3, float4(1.0f));
                for (int r = 0; r < MV_R; r++)
                    acc[r][t] += dl[r] * (dot(q[r][0], x0) + dot(q[r][1], x1) + dot(q[r][2], x2) + dot(q[r][3], x3)) - ml[r] * sx;
            }
        }
    }
    for (int r = 0; r < MV_R; r++)
        for (int t = 0; t < NT; t++) {
            float s = simd_sum(acc[r][t]);
            if (lane == 0 && row0 + r < a.M && t0 + t < a.N) Y[(t0 + t) * a.ldy + row0 + r] = s;
        }
}

typedef decltype(mv<2, 84, 1>) mv_t;
#define MV_INST(name, T, BB)                                          \
    template [[host_name("mv_" #name "_1")]] kernel mv_t mv<T, BB, 1>; \
    template [[host_name("mv_" #name "_2")]] kernel mv_t mv<T, BB, 2>; \
    template [[host_name("mv_" #name "_4")]] kernel mv_t mv<T, BB, 4>; \
    template [[host_name("mv_" #name "_8")]] kernel mv_t mv<T, BB, 8>;
MV_INST(q2k, 2, 84)
MV_INST(q3k, 3, 110)
MV_INST(q4k, 4, 144)
MV_INST(q5k, 5, 176)
MV_INST(q6k, 6, 210)
MV_INST(q8_0, 8, 272)
MV_INST(q4_0, 40, 144)

// ---------------------------------------------------------------------------
// Fused GEMM for medium batches (9-64 rows: many concurrent agents): 64 weight
// rows x TN tokens per threadgroup, K step 64. Each thread decodes two 16-value
// sub-blocks of one row per step into threadgroup memory (no f16 copy in device
// memory), then 4 simdgroups x 16 rows run 8x8 simdgroup MMAs.
// ---------------------------------------------------------------------------
#define FM_KS 64
#define FM_SA (FM_KS + 4) // padded stride spreads the lanes' stores across banks

template <int T, int BB, int TN>
kernel void mm_fused(device const uchar* W [[buffer(0)]], device const float* X [[buffer(1)]], device float* Y [[buffer(2)]],
                     constant MvArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64 * FM_SA];
    threadgroup half sb[TN * FM_KS];
    const uint r0 = tg.y * 64, t0 = tg.x * TN;
    simdgroup_float8x8 acc[2][TN / 8];
    for (int i = 0; i < 2; i++) for (int jn = 0; jn < TN / 8; jn++) acc[i][jn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint wr = tid >> 1, wp = tid & 1;
    const bool wvalid = r0 + wr < a.M;
    device const uchar* wrow = W + (r0 + wr) * a.row_bytes;
    for (uint ks = 0; ks < a.K / FM_KS; ks++) {
        device const uchar* blk = wrow + (ks / (QK / FM_KS)) * BB;
        for (int s = 0; s < 2; s++) {
            int j = int((ks % (QK / FM_KS)) * (FM_KS / 16) + wp * 2 + s);
            float4 q[4];
            float dl = 0.0f, ml = 0.0f;
            if (wvalid) deqq<T>(blk, j, q, dl, ml);
            else for (int k = 0; k < 4; k++) q[k] = float4(0.0f);
            threadgroup half4* dst = (threadgroup half4*)(sa + wr * FM_SA + (wp * 2 + s) * 16);
            for (int k = 0; k < 4; k++) dst[k] = half4(dl * q[k] - ml);
        }
        for (uint e = tid; e < TN * FM_KS / 4; e += 128) {
            uint t = e / (FM_KS / 4), kk = (e % (FM_KS / 4)) * 4;
            float4 v = (t0 + t < a.N) ? *((device const float4*)(X + (t0 + t) * a.x_stride + ks * FM_KS + kk)) : float4(0.0f);
            *((threadgroup half4*)(sb + t * FM_KS + kk)) = half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < FM_KS / 8; kk++) {
            simdgroup_half8x8 ma[2], mb[TN / 8];
            for (int i = 0; i < 2; i++) simdgroup_load(ma[i], sa + (sg * 16 + i * 8) * FM_SA + kk * 8, FM_SA);
            for (int jn = 0; jn < TN / 8; jn++) simdgroup_load(mb[jn], sb + (jn * 8) * FM_KS + kk * 8, FM_KS, ulong2(0, 0), true);
            for (int i = 0; i < 2; i++) for (int jn = 0; jn < TN / 8; jn++) simdgroup_multiply_accumulate(acc[i][jn], ma[i], mb[jn], acc[i][jn]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* so = (threadgroup float*)sa; // 64 rows x TN tokens fits in the weight tile
    for (int i = 0; i < 2; i++)
        for (int jn = 0; jn < TN / 8; jn++) simdgroup_store(acc[i][jn], so + (jn * 8) * 64 + sg * 16 + i * 8, 64, ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < TN * 64; e += 128) {
        uint t = e / 64, r = e % 64;
        if (t0 + t < a.N && r0 + r < a.M) Y[(t0 + t) * a.ldy + r0 + r] = so[e];
    }
}

typedef decltype(mm_fused<2, 84, 16>) mmf_t;
#define MMF_INST(name, T, BB)                                                  \
    template [[host_name("mmf_" #name "_16")]] kernel mmf_t mm_fused<T, BB, 16>; \
    template [[host_name("mmf_" #name "_32")]] kernel mmf_t mm_fused<T, BB, 32>;
MMF_INST(q2k, 2, 84)
MMF_INST(q3k, 3, 110)
MMF_INST(q4k, 4, 144)
MMF_INST(q5k, 5, 176)
MMF_INST(q6k, 6, 210)
MMF_INST(q8_0, 8, 272)
MMF_INST(q4_0, 40, 144)

// ---------------------------------------------------------------------------
// Affine (MLX) mat-vec for small batches: Y[t][r] = W[r] . X[t] for NT tokens,
// with W[r][k] = scale * q + bias per group of G columns, q packed at BITS.
// Layout follows MLX's qmv_fast (2 simdgroups x 4 rows, each lane 16 (8 for
// 8-bit) consecutive values per pass); codes are decoded once per pass and
// reused for all NT tokens, and the bias term uses sum(x).
// Instantiated at runtime for the (S, G, BITS) combinations the model uses.
// ---------------------------------------------------------------------------
struct AmvArgs { uint K, M, N, ldy, x_stride; };

template <typename S, int G, int BITS, int NT>
kernel void amv(device const uchar* W [[buffer(0)]], device const S* SC [[buffer(1)]], device const S* BI [[buffer(2)]],
                device const float* X [[buffer(3)]], device float* Y [[buffer(4)]], constant AmvArgs& a [[buffer(5)]],
                uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int VPT = BITS == 8 ? 8 : 16; // values per lane per pass
    constexpr int BPT = VPT * BITS / 8;     // packed bytes per lane per pass
    constexpr int R = 4;
    const uint row0 = tg.x * 8 + sg * R;
    const uint t0 = tg.y * NT;
    if (row0 >= a.M) return;
    const uint rb = a.K * BITS / 8, ng = a.K / G;
    float acc[R][NT];
    for (int r = 0; r < R; r++) for (int t = 0; t < NT; t++) acc[r][t] = 0.0f;
    for (uint k = lane * VPT; k < a.K; k += VPT * 32) {
        float q[R][VPT], sc[R], bi[R];
        for (int r = 0; r < R; r++) {
            uint row = min(row0 + r, a.M - 1);
            device const uchar* wp = W + row * rb + k * BITS / 8;
            sc[r] = float(SC[row * ng + k / G]);
            bi[r] = float(BI[row * ng + k / G]);
            if (BITS == 4) {
                uint2 v = *((device const uint2*)wp);
                for (int i = 0; i < 8; i++) { q[r][i] = float((v.x >> (4 * i)) & 15u); q[r][8 + i] = float((v.y >> (4 * i)) & 15u); }
            } else if (BITS == 8) {
                uint2 v = *((device const uint2*)wp);
                float4 lo = float4(as_type<uchar4>(v.x)), hi = float4(as_type<uchar4>(v.y));
                for (int i = 0; i < 4; i++) { q[r][i] = lo[i]; q[r][4 + i] = hi[i]; }
            } else { // 2-bit
                uint v = *((device const uint*)wp);
                for (int i = 0; i < 16; i++) q[r][i] = float((v >> (2 * i)) & 3u);
            }
        }
        for (int t = 0; t < NT; t++) {
            if (t0 + t >= a.N) break;
            device const float4* xp = (device const float4*)(X + (t0 + t) * a.x_stride + k);
            float4 xv[VPT / 4];
            float sx = 0.0f;
            for (int i = 0; i < VPT / 4; i++) { xv[i] = xp[i]; sx += xv[i].x + xv[i].y + xv[i].z + xv[i].w; }
            for (int r = 0; r < R; r++) {
                float d = 0.0f;
                for (int i = 0; i < VPT / 4; i++) d += dot(xv[i], float4(q[r][4 * i], q[r][4 * i + 1], q[r][4 * i + 2], q[r][4 * i + 3]));
                acc[r][t] += sc[r] * d + bi[r] * sx;
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int t = 0; t < NT; t++) {
            float v = simd_sum(acc[r][t]);
            if (lane == 0 && row0 + r < a.M && t0 + t < a.N) Y[(t0 + t) * a.ldy + row0 + r] = v;
        }
}

// ---------------------------------------------------------------------------
// Dequantize rows [r0, r0+rows) of W into a dense f16 matrix D[rows][K].
// One thread per 16-value sub-block.
// ---------------------------------------------------------------------------
struct DqArgs { uint K, rows, row_bytes, r0; };

template <int T, int BB>
kernel void dequant_f16(device const uchar* W [[buffer(0)]], device half* D [[buffer(1)]], constant DqArgs& a [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint sub = gid.x, r = gid.y;
    if (r >= a.rows || sub >= a.K / 16) return;
    float4 q[4];
    float dl, ml;
    deqq<T>(W + (a.r0 + r) * a.row_bytes + (sub / 16) * BB, int(sub % 16), q, dl, ml);
    device half4* dst = (device half4*)(D + (size_t)r * a.K + sub * 16);
    for (int k = 0; k < 4; k++) dst[k] = half4(dl * q[k] - ml);
}

typedef decltype(dequant_f16<2, 84>) dq_t;
template [[host_name("dq_q2k")]] kernel dq_t dequant_f16<2, 84>;
template [[host_name("dq_q3k")]] kernel dq_t dequant_f16<3, 110>;
template [[host_name("dq_q4k")]] kernel dq_t dequant_f16<4, 144>;
template [[host_name("dq_q5k")]] kernel dq_t dequant_f16<5, 176>;
template [[host_name("dq_q6k")]] kernel dq_t dequant_f16<6, 210>;
template [[host_name("dq_q8_0")]] kernel dq_t dequant_f16<8, 272>;
template [[host_name("dq_q4_0")]] kernel dq_t dequant_f16<40, 144>;

// ---------------------------------------------------------------------------
// Dense GEMM: Y[t*ldy + r0 + r] = sum_k D[r][k] * X[t][k], D f16 [rows x K].
// Threadgroup tile 64 rows x 32 tokens, K step 32, 4 simdgroups x 16 rows.
// Grid x = token tiles (fastest) so concurrent threadgroups share a row tile in cache.
// ---------------------------------------------------------------------------
struct GemmArgs { uint K, rows, N, ldy, r0, x_stride; };
#define GS 40 // padded tile stride (halves)

kernel void gemm_f16(device const half* D [[buffer(0)]], device const float* X [[buffer(1)]], device float* Y [[buffer(2)]],
                     constant GemmArgs& a [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[64 * GS];
    threadgroup half sb[32 * GS];
    threadgroup float so[32 * 64];
    const uint t0 = tg.x * 32, rb = tg.y * 64;
    simdgroup_float8x8 acc[2][4];
    for (int i = 0; i < 2; i++) for (int jn = 0; jn < 4; jn++) acc[i][jn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint wr = tid >> 1, wp = tid & 1;   // weight tile: 64 rows x 2 halves-of-32
    const uint xt = tid >> 2, xp = tid & 3;   // activation tile: 32 tokens x 4 quarters-of-32
    for (uint k0 = 0; k0 < a.K; k0 += 32) {
        threadgroup half4* dw = (threadgroup half4*)(sa + wr * GS + wp * 16);
        if (rb + wr < a.rows) {
            device const half4* src = (device const half4*)(D + (size_t)(rb + wr) * a.K + k0 + wp * 16);
            for (int i = 0; i < 4; i++) dw[i] = src[i];
        } else for (int i = 0; i < 4; i++) dw[i] = half4(0.0h);
        threadgroup half4* dx = (threadgroup half4*)(sb + xt * GS + xp * 8);
        if (t0 + xt < a.N) {
            device const float4* src = (device const float4*)(X + (size_t)(t0 + xt) * a.x_stride + k0 + xp * 8);
            dx[0] = half4(src[0]);
            dx[1] = half4(src[1]);
        } else { dx[0] = half4(0.0h); dx[1] = half4(0.0h); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; kk++) {
            simdgroup_half8x8 ma[2], mb[4];
            for (int i = 0; i < 2; i++) simdgroup_load(ma[i], sa + (sg * 16 + i * 8) * GS + kk * 8, GS);
            for (int jn = 0; jn < 4; jn++) simdgroup_load(mb[jn], sb + (jn * 8) * GS + kk * 8, GS, ulong2(0, 0), true);
            for (int i = 0; i < 2; i++) for (int jn = 0; jn < 4; jn++) simdgroup_multiply_accumulate(acc[i][jn], ma[i], mb[jn], acc[i][jn]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 2; i++)
        for (int jn = 0; jn < 4; jn++) simdgroup_store(acc[i][jn], so + (jn * 8) * 64 + sg * 16 + i * 8, 64, ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 32 * 64; e += 128) {
        uint t = e / 64, r = e % 64;
        if (t0 + t < a.N && rb + r < a.rows) Y[(size_t)(t0 + t) * a.ldy + a.r0 + rb + r] = so[e];
    }
}

// ---------------------------------------------------------------------------
// RMS norm. add=0: y[row] = rms(x[src])*w   add=1: y[row] += rms(x[row])*w
// ---------------------------------------------------------------------------
struct NormArgs { uint d; float eps; uint has_w, add, gather, half_out; }; // half_out: y is f16 (matmul input)

kernel void rms_norm(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], device const float* w [[buffer(2)]],
                     constant NormArgs& a [[buffer(3)]], device const uint* idx [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[32];
    uint src = a.gather ? idx[row] : row;
    device const float* xr = x + (size_t)src * a.d;
    device float* yr = y + (size_t)row * a.d;
    float ss = 0.0f;
    for (uint i = tid; i < a.d; i += nt) { float v = xr[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (nt + 31) / 32 ? part[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) part[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = rsqrt(part[0] / float(a.d) + a.eps);
    for (uint i = tid; i < a.d; i += nt) {
        float v = xr[i] * r * (a.has_w ? w[i] : 1.0f);
        if (a.half_out) ((device half*)y)[(size_t)row * a.d + i] = half(v);
        else if (a.add) yr[i] += v;
        else yr[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Fused residual step (one threadgroup of 512 per row, row kept in registers):
//   x += post ? rms(o) * wp : o        (wp may be null: plain add)
//   h  = rms(x) * wn                    (f16 when half_out: next matmul input)
// d must be a multiple of 4 and <= 512 * 4 * RA_V.
// ---------------------------------------------------------------------------
#define RA_V 8
struct AddNormArgs { uint d; float eps_post, eps; uint post, half_out; };

static inline float tg_sum(float v, threadgroup float* part, ushort lane, ushort sg, uint nt) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < (nt + 31) / 32 ? part[lane] : 0.0f;
    return simd_sum(t);
}

kernel void add_rms_norm(device const float* o [[buffer(0)]], device float* x [[buffer(1)]], device float* h [[buffer(2)]],
                         device const float* wp [[buffer(3)]], device const float* wn [[buffer(4)]], constant AddNormArgs& a [[buffer(5)]],
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[32];
    const uint d4 = a.d / 4;
    device const float4* orow = (device const float4*)(o + (size_t)row * a.d);
    device float4* xrow = (device float4*)(x + (size_t)row * a.d);
    float4 v[RA_V];
    float ss = 0.0f;
    for (int j = 0; j < RA_V; j++) {
        uint i = tid + j * nt;
        v[j] = i < d4 ? orow[i] : float4(0.0f);
        ss += dot(v[j], v[j]);
    }
    float ro = a.post ? rsqrt(tg_sum(ss, part, lane, sg, nt) / float(a.d) + a.eps_post) : 1.0f;
    ss = 0.0f;
    for (int j = 0; j < RA_V; j++) {
        uint i = tid + j * nt;
        if (i < d4) {
            float4 add = v[j] * ro;
            if (a.post) add *= ((device const float4*)wp)[i];
            v[j] = xrow[i] + add;
            xrow[i] = v[j];
        }
        ss += dot(v[j], v[j]);
    }
    float r = rsqrt(tg_sum(ss, part, lane, sg, nt) / float(a.d) + a.eps);
    for (int j = 0; j < RA_V; j++) {
        uint i = tid + j * nt;
        if (i >= d4) continue;
        float4 y = v[j] * r * ((device const float4*)wn)[i];
        if (a.half_out) ((device half4*)h)[(size_t)row * d4 + i] = half4(y);
        else ((device float4*)h)[(size_t)row * d4 + i] = y;
    }
}

// ---------------------------------------------------------------------------
// Per-head QK RMS norm (optional) + RoPE (optional; NORM or NEOX), in place.
// grid (rows, n_head + n_kv) threadgroups of 32; n_head may be 0 (K only).
// ---------------------------------------------------------------------------
struct QkArgs { uint nh, nkv, hd; float eps; uint has_qn, has_kn, rope, neox, qs, ks; }; // qs/ks: row strides (floats)

kernel void qk_norm_rope(device float* q [[buffer(0)]], device float* k [[buffer(1)]], device const float* qn [[buffer(2)]],
                         device const float* kn [[buffer(3)]], device const uint* pos [[buffer(4)]],
                         device const float* inv_freq [[buffer(5)]], constant QkArgs& a [[buffer(6)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, h = tg.y, hd = a.hd, E = hd / 32, half_ = hd / 2;
    bool isq = h < a.nh;
    device float* x = isq ? q + row * a.qs + h * hd : k + row * a.ks + (h - a.nh) * hd;
    bool norm = isq ? a.has_qn : a.has_kn;
    device const float* wn = isq ? qn : kn;
    uint ia[4], ib[4], fi[4];
    const uint np = E / 2;
    for (uint i = 0; i < np; i++) {
        if (a.neox) { uint pi = lane * np + i; ia[i] = pi; ib[i] = pi + half_; fi[i] = pi; }
        else { uint e = lane * E + 2 * i; ia[i] = e; ib[i] = e + 1; fi[i] = e / 2; }
    }
    float va[4], vb[4], ss = 0.0f;
    for (uint i = 0; i < np; i++) { va[i] = x[ia[i]]; vb[i] = x[ib[i]]; ss += va[i] * va[i] + vb[i] * vb[i]; }
    if (norm) {
        ss = simd_sum(ss);
        float r = rsqrt(ss / float(hd) + a.eps);
        for (uint i = 0; i < np; i++) { va[i] *= r * wn[ia[i]]; vb[i] *= r * wn[ib[i]]; }
    }
    if (a.rope) {
        float p = float(pos[row]);
        for (uint i = 0; i < np; i++) {
            float ang = p * inv_freq[fi[i]];
            float c = precise::cos(ang), s = precise::sin(ang);
            float x0 = va[i], x1 = vb[i];
            va[i] = x0 * c - x1 * s;
            vb[i] = x0 * s + x1 * c;
        }
    }
    for (uint i = 0; i < np; i++) { x[ia[i]] = va[i]; x[ib[i]] = vb[i]; }
}

// ---------------------------------------------------------------------------
// Write K/V rows into their paged f16 slots. slot[row] = page*page_elems + (pos%PAGE)*hd.
// ---------------------------------------------------------------------------
struct KvwArgs { uint nkv, hd, page, ks; ulong layer_off; uint vs, pad; }; // ks/vs: row strides of k/v

kernel void kv_write(device const float* k [[buffer(0)]], device const float* v [[buffer(1)]], device half* pool [[buffer(2)]],
                     device const ulong* slot [[buffer(3)]], constant KvwArgs& a [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint e = gid.x, row = gid.y, kvd = a.nkv * a.hd;
    if (e >= kvd) return;
    uint h = e / a.hd, dd = e % a.hd;
    ulong base = slot[row] + a.layer_off + h * a.page * a.hd + dd;
    pool[base] = half(k[row * a.ks + e]);
    pool[base + a.nkv * a.page * a.hd] = half(v[row * a.vs + e]);
}

// ---------------------------------------------------------------------------
// Flash-decoding attention. grid (rows, kv_heads, splits); one simdgroup per q
// head of the GQA group (G = nh/nkv <= 16). K/V tiles of AT positions are
// loaded once into threadgroup memory and shared by the group; scores are
// position-parallel (lane = position) with online softmax; partial (m, l, o)
// per split; attn_reduce merges splits and applies the optional sigmoid gate.
// ---------------------------------------------------------------------------
struct AttnArgs { uint nh, nkv, hd; float scale; ulong page_elems, layer_off; uint page, gate, qs, gs, half_out, pad; }; // qs/gs: row strides of q/gate
struct RowInfo { uint ctx_end, start, tab, split; }; // split: positions per split for this row
#define AT 32

template <int HD>
kernel void attn_fd(device const float* q [[buffer(0)]], device const half* pool [[buffer(1)]], device const RowInfo* ri [[buffer(2)]],
                    device const uint* tab [[buffer(3)]], device float* part [[buffer(4)]], constant AttnArgs& a [[buffer(5)]],
                    constant uint2& sp [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                    uint tid [[thread_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]]) {
    constexpr int E = HD / 32;
    constexpr int KS = HD + 2; // padded K row stride: lanes (= positions) hit distinct banks
    const uint ntg = ntg3.x;
    threadgroup float qs[16 * HD];
    threadgroup half ks[AT * KS];
    threadgroup half vs[AT * HD];
    const uint row = tg.x, kvh = tg.y, split = tg.z;
    const uint G = a.nh / a.nkv, h = kvh * G + sg;
    const RowInfo r = ri[row];
    const uint s0 = r.start + split * r.split;
    const uint s1 = min(r.ctx_end + 1, s0 + r.split); // s0 >= s1: empty split (l = 0)
    for (uint i = tid; i < G * HD; i += ntg) qs[i] = q[row * a.qs + kvh * G * HD + i] * a.scale;
    const ulong vofs = (ulong)a.nkv * a.page * HD;
    float m = -INFINITY, l = 0.0f, o[E];
    for (int i = 0; i < E; i++) o[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t0 = s0; t0 < s1; t0 += AT) {
        for (uint i = tid; i < AT * HD; i += ntg) {
            uint j = i / HD, dd = i % HD, p = t0 + j;
            half kv = 0.0h, vv = 0.0h;
            if (p < s1) {
                ulong base = (ulong)tab[r.tab + p / a.page] * a.page_elems + a.layer_off + (ulong)kvh * a.page * HD + (p % a.page) * HD + dd;
                kv = pool[base];
                vv = pool[base + vofs];
            }
            ks[j * KS + dd] = kv;
            vs[j * HD + dd] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint p = t0 + lane;
        float s = -INFINITY;
        if (p < s1) {
            threadgroup const float* qh = qs + sg * HD;
            threadgroup const half* kr = ks + lane * KS;
            float acc = 0.0f;
            for (int d = 0; d < HD; d += 2) acc += qh[d] * float(kr[d]) + qh[d + 1] * float(kr[d + 1]);
            s = acc;
        }
        const float mn = max(m, simd_max(s));
        const float c = exp(m - mn);
        const float pl = p < s1 ? exp(s - mn) : 0.0f;
        l = l * c + simd_sum(pl);
        for (int i = 0; i < E; i++) o[i] *= c;
        const uint nj = min((uint)AT, s1 - t0);
        for (uint j = 0; j < nj; j++) {
            float pj = simd_shuffle(pl, j);
            threadgroup const half* vr = vs + j * HD + lane * E;
            for (int i = 0; i < E; i++) o[i] += pj * float(vr[i]);
        }
        m = mn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* dst = part + ((row * a.nh + h) * sp.y + split) * (HD + 2);
    if (lane == 0) { dst[0] = m; dst[1] = l; }
    for (int i = 0; i < E; i++) dst[2 + lane * E + i] = o[i];
}

typedef decltype(attn_fd<128>) attn_fd_t;
template [[host_name("attn_fd_64")]] kernel attn_fd_t attn_fd<64>;
template [[host_name("attn_fd_128")]] kernel attn_fd_t attn_fd<128>;

kernel void attn_reduce(device const float* part [[buffer(0)]], device const float* g [[buffer(1)]], device float* out [[buffer(2)]],
                        constant AttnArgs& a [[buffer(3)]], constant uint2& sp [[buffer(4)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, h = tg.y, hd = a.hd, E = hd / 32, ns = sp.y;
    device const float* src = part + (row * a.nh + h) * ns * (hd + 2);
    float M = -INFINITY;
    for (uint s = 0; s < ns; s++) M = max(M, src[s * (hd + 2)]);
    float L = 0.0f, o[8];
    for (uint i = 0; i < E; i++) o[i] = 0.0f;
    for (uint s = 0; s < ns; s++) {
        device const float* ps = src + s * (hd + 2);
        float w = ps[1] > 0.0f ? exp(ps[0] - M) : 0.0f;
        L += ps[1] * w;
        for (uint i = 0; i < E; i++) o[i] += ps[2 + lane * E + i] * w;
    }
    for (uint i = 0; i < E; i++) {
        uint idx = row * a.nh * hd + h * hd + lane * E + i;
        float v = o[i] / L;
        if (a.gate) v *= 1.0f / (1.0f + exp(-g[row * a.gs + h * hd + lane * E + i]));
        if (a.half_out) ((device half*)out)[idx] = half(v);
        else out[idx] = v;
    }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------
// out[r][c] = silu(g[r*stride + c]) * u[r*stride + c], c < ff (g/u may be halves of one fused row)
// a.z: write f16
kernel void silu_mul(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* out [[buffer(2)]],
                     constant uint3& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    uint r = i / a.x, c = i % a.x;
    float x = g[r * a.y + c];
    float v = x / (1.0f + exp(-x)) * u[r * a.y + c];
    if (a.z) ((device half*)out)[i] = half(v);
    else out[i] = v;
}

// y[i] += b[i % w] (bias add, or a plain residual add with w = n*d)
kernel void add_bcast(device float* y [[buffer(0)]], device const float* b [[buffer(1)]], constant uint& w [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    y[i] += b[i % w];
}

struct CapArgs { float scale, cap; };
kernel void softcap(device float* y [[buffer(0)]], constant CapArgs& a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float v = y[i] * a.scale;
    y[i] = a.cap > 0.0f ? a.cap * precise::tanh(v / a.cap) : v;
}

// f32 -> f16 copy (activations feeding the NAX GEMM), 4 elements per thread; n % 4 == 0
kernel void to_half(device const float4* x [[buffer(0)]], device half4* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n / 4) y[i] = half4(x[i]);
}

// ---------------------------------------------------------------------------
// Speculative decoding helpers
// ---------------------------------------------------------------------------
// Argmax per row (lowest index among ties, like the CPU sampler); one
// threadgroup of 1024 per row.
kernel void argmax_rows(device const float* x [[buffer(0)]], device int* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    device const float* r = x + (size_t)row * n;
    float best = -INFINITY;
    uint idx = 0xFFFFFFFFu;
    for (uint i = tid; i < n; i += nt) {
        float v = r[i];
        if (v > best) { best = v; idx = i; } // per-thread indices increase, so '>' keeps the lowest
    }
    for (ushort o = 16; o > 0; o >>= 1) {
        float ov = simd_shuffle_down(best, o);
        uint oi = simd_shuffle_down(idx, o);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (lane == 0) { bv[sg] = best; bi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint ns = (nt + 31) / 32;
        best = lane < ns ? bv[lane] : -INFINITY;
        idx = lane < ns ? bi[lane] : 0xFFFFFFFFu;
        for (ushort o = 16; o > 0; o >>= 1) {
            float ov = simd_shuffle_down(best, o);
            uint oi = simd_shuffle_down(idx, o);
            if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
        }
        if (lane == 0) out[row] = int(idx);
    }
}

// dst[r * ld + off + c] = half(src[r * d + c]): capture a residual stream slice
kernel void cap_write(device const float* src [[buffer(0)]], device half* dst [[buffer(1)]], constant uint3& a [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint d = a.x, ld = a.y, off = a.z, c = gid.x, r = gid.y;
    if (c < d) dst[(size_t)r * ld + off + c] = half(src[(size_t)r * d + c]);
}

// dst[i] = src[idx[i]] for rows of w halves
kernel void gather_half(device const half* src [[buffer(0)]], device half* dst [[buffer(1)]], device const uint* idx [[buffer(2)]],
                        constant uint& w [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < w) dst[(size_t)gid.y * w + gid.x] = src[(size_t)idx[gid.y] * w + gid.x];
}
