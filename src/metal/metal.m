// Metal backend (Apple GPU). The mmap'd GGUF is wrapped as one zero-copy
// MTLBuffer (weights and F32 norm vectors are addressed by offset); KV pages
// are f16 in shared buffers so the engine's copy-on-write bookkeeping stays on
// the CPU. One command buffer per forward step.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "backend.h"
#include "kernels_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MV_MAX_ROWS 8            // fused mat-vec up to here
#define FUSED_MAX_ROWS 64        // fused tile-dequant GEMM up to here; above: dequantize once + dense GEMM
#define SCRATCH_BYTES (256u << 20) // f16 dequant scratch; bigger matrices go in row chunks
#define ATTN_SPLIT 256           // positions per flash-decoding split

typedef struct { uint32_t K, M, N, ldy, row_bytes, x_stride; } MvArgs;
typedef struct { uint32_t K, rows, row_bytes, r0; } DqArgs;
typedef struct { uint32_t K, rows, N, ldy, r0, x_stride; } GemmArgs;
typedef struct { uint32_t d; float eps; uint32_t has_w, add, gather; } NormArgs;
typedef struct { uint32_t nh, nkv, hd; float eps; uint32_t has_qn, has_kn, rope, neox; } QkArgs;
typedef struct { uint32_t nkv, hd, page, pad; uint64_t layer_off; } KvwArgs;
typedef struct { uint32_t nh, nkv, hd; float scale; uint64_t page_elems, layer_off; uint32_t page, gate; } AttnArgs;
typedef struct { uint32_t ctx_end, start, tab, pad; } RowInfo;
typedef struct { float scale, cap; } CapArgs;

@interface CIMetal : NSObject {
@public
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pso;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *bufs;
    id<MTLBuffer> wbuf;
    const uint8_t *wbase;
    size_t wlen;
    id<MTLBuffer> invf;
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

static uint64_t woff(CIMetal *mt, const void *p) { return (uint64_t)((const uint8_t *)p - mt->wbase); }

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

// y[t*ldy + r] = W[r] . x[t] for n tokens
static void matmul(CIMetal *mt, id<MTLComputeCommandEncoder> e, const qmat *w, id<MTLBuffer> x, id<MTLBuffer> y, int ldy, int n) {
    const char *ty = qname(w->type);
    if (n <= mv_max()) {
        int nt = n == 1 ? 1 : n == 2 ? 2 : n <= 4 ? 4 : 8;
        MvArgs a = { (uint32_t)w->cols, (uint32_t)w->rows, (uint32_t)n, (uint32_t)ldy, (uint32_t)w->row_bytes, (uint32_t)w->cols };
        [e setComputePipelineState:P(mt, [NSString stringWithFormat:@"mv_%s_%d", ty, nt])];
        [e setBuffer:mt->wbuf offset:woff(mt, w->data) atIndex:0];
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
        [e setBuffer:mt->wbuf offset:woff(mt, w->data) atIndex:0];
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
        [e setBuffer:mt->wbuf offset:woff(mt, w->data) atIndex:0];
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

static void rms(CIMetal *mt, id<MTLComputeCommandEncoder> e, id<MTLBuffer> x, id<MTLBuffer> y, const float *w, int d, float eps, bool add, int rows, id<MTLBuffer> gather) {
    NormArgs a = { (uint32_t)d, eps, w != NULL, add, gather != nil };
    [e setComputePipelineState:P(mt, @"rms_norm")];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:y offset:0 atIndex:1];
    [e setBuffer:mt->wbuf offset:w ? woff(mt, w) : 0 atIndex:2];
    set_bytes(e, &a, sizeof a, 3);
    [e setBuffer:gather ? gather : x offset:0 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void add_bcast(CIMetal *mt, id<MTLComputeCommandEncoder> e, id<MTLBuffer> y, id<MTLBuffer> b, uint64_t b_off, uint32_t w, size_t n) {
    [e setComputePipelineState:P(mt, @"add_bcast")];
    [e setBuffer:y offset:0 atIndex:0];
    [e setBuffer:b offset:b_off atIndex:1];
    set_bytes(e, &w, sizeof w, 2);
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
        id<MTLBuffer> xb = B(mt, @"x", f * n * d), hb = B(mt, @"h", f * n * d), qb = B(mt, @"q", f * n * qd),
                      kb = B(mt, @"k", f * n * kvd), vb = B(mt, @"v", f * n * kvd), gb = B(mt, @"g", f * n * qd),
                      ab = B(mt, @"att", f * n * qd), ob = B(mt, @"o", f * n * d), fgb = B(mt, @"fg", f * n * c->ff),
                      fub = B(mt, @"fu", f * n * c->ff);
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
        uint64_t *slot_full = malloc(8 * n), *slot_swa = malloc(8 * n);
        uint32_t max_full = 1, max_swa = 1;
        for (int i = 0; i < n; i++) {
            const row_t *r = &rows[i];
            const seq_pages *pg = pages[r->seq];
            uint32_t sw = kv->window && r->ctx_end + 1 > kv->window ? (uint32_t)(r->ctx_end + 1 - kv->window) : 0;
            ri_full[i] = (RowInfo){ (uint32_t)r->ctx_end, 0, off_full[r->seq], 0 };
            ri_swa[i] = (RowInfo){ (uint32_t)r->ctx_end, sw, off_swa[r->seq], 0 };
            if ((uint32_t)r->ctx_end + 1 > max_full) max_full = r->ctx_end + 1;
            if ((uint32_t)r->ctx_end + 1 - sw > max_swa) max_swa = r->ctx_end + 1 - sw;
            slot_full[i] = (uint64_t)pg->full[r->pos / KV_PAGE] * kv->full.page_elems + (uint64_t)(r->pos % KV_PAGE) * hd;
            slot_swa[i] = kv->window ? (uint64_t)pg->swa[r->pos / KV_PAGE] * kv->swa.page_elems + (uint64_t)(r->pos % KV_PAGE) * hd : 0;
        }
        write_buf(B(mt, @"ri_full", sizeof(RowInfo) * n), ri_full, sizeof(RowInfo) * n);
        write_buf(B(mt, @"ri_swa", sizeof(RowInfo) * n), ri_swa, sizeof(RowInfo) * n);
        write_buf(B(mt, @"slot_full", 8 * n), slot_full, 8 * n);
        write_buf(B(mt, @"slot_swa", 8 * n), slot_swa, 8 * n);
        int ns = 0;
        uint32_t *sel = malloc(4 * n);
        for (int i = 0; i < n; i++) if (rows[i].logits) sel[ns++] = (uint32_t)i;
        if (ns) write_buf(B(mt, @"sel", 4 * ns), sel, 4 * ns);
        int n_split_max = (int)((max_full > max_swa ? max_full : max_swa) + ATTN_SPLIT - 1) / ATTN_SPLIT;
        id<MTLBuffer> part = B(mt, @"part", (size_t)n * c->n_head * n_split_max * (hd + 2) * 4);
        id<MTLBuffer> full_pool = (__bridge id<MTLBuffer>)kv->full.handle;
        id<MTLBuffer> swa_pool = kv->window ? (__bridge id<MTLBuffer>)kv->swa.handle : full_pool;
        id<MTLBuffer> hsel = ns ? B(mt, @"hsel", f * ns * d) : nil, lg = ns ? B(mt, @"logits", f * (size_t)ns * c->vocab) : nil;

        id<MTLCommandBuffer> cb = [mt->queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        bool fd = c->n_head / c->n_kv <= 16 && (hd == 64 || hd == 128);
        if (!fd) { fprintf(stderr, "metal: head_dim %d / GQA group %d not supported by attention kernel\n", hd, c->n_head / c->n_kv); exit(1); }
        for (int li = 0; li < c->n_layer; li++) {
            const layer *l = &m->layers[li];
            rms(mt, e, xb, hb, l->attn_norm, d, c->eps, false, n, nil);
            matmul(mt, e, &l->wq, hb, qb, qd, n);
            matmul(mt, e, &l->wk, hb, kb, kvd, n);
            matmul(mt, e, &l->wv, hb, vb, kvd, n);
            if (l->wg.data) matmul(mt, e, &l->wg, hb, gb, qd, n);
            if (l->bq) add_bcast(mt, e, qb, mt->wbuf, woff(mt, l->bq), qd, (size_t)n * qd);
            if (l->bk) add_bcast(mt, e, kb, mt->wbuf, woff(mt, l->bk), kvd, (size_t)n * kvd);
            if (l->bv) add_bcast(mt, e, vb, mt->wbuf, woff(mt, l->bv), kvd, (size_t)n * kvd);
            if (l->q_norm || l->k_norm || l->rope) {
                QkArgs a = { (uint32_t)c->n_head, (uint32_t)c->n_kv, (uint32_t)hd, c->eps, l->q_norm != NULL, l->k_norm != NULL, l->rope, c->rope_neox };
                [e setComputePipelineState:P(mt, @"qk_norm_rope")];
                [e setBuffer:qb offset:0 atIndex:0];
                [e setBuffer:kb offset:0 atIndex:1];
                [e setBuffer:mt->wbuf offset:l->q_norm ? woff(mt, l->q_norm) : 0 atIndex:2];
                [e setBuffer:mt->wbuf offset:l->k_norm ? woff(mt, l->k_norm) : 0 atIndex:3];
                [e setBuffer:B(mt, @"pos", 4 * n) offset:0 atIndex:4];
                [e setBuffer:mt->invf offset:0 atIndex:5];
                set_bytes(e, &a, sizeof a, 6);
                [e dispatchThreadgroups:MTLSizeMake(n, c->n_head + c->n_kv, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            const kv_pool *pl = l->swa ? &kv->swa : &kv->full;
            id<MTLBuffer> pool = l->swa ? swa_pool : full_pool;
            uint64_t layer_off = (uint64_t)l->pool_layer * 2 * c->n_kv * KV_PAGE * hd;
            {
                KvwArgs a = { (uint32_t)c->n_kv, (uint32_t)hd, KV_PAGE, 0, layer_off };
                [e setComputePipelineState:P(mt, @"kv_write")];
                [e setBuffer:kb offset:0 atIndex:0];
                [e setBuffer:vb offset:0 atIndex:1];
                [e setBuffer:pool offset:0 atIndex:2];
                [e setBuffer:B(mt, l->swa ? @"slot_swa" : @"slot_full", 8 * n) offset:0 atIndex:3];
                set_bytes(e, &a, sizeof a, 4);
                [e dispatchThreads:MTLSizeMake(kvd, n, 1) threadsPerThreadgroup:MTLSizeMake(kvd < 256 ? kvd : 256, 1, 1)];
            }
            {
                AttnArgs a = { (uint32_t)c->n_head, (uint32_t)c->n_kv, (uint32_t)hd, c->attn_scale, pl->page_elems, layer_off, KV_PAGE, l->wg.data != NULL };
                uint32_t mx = l->swa ? max_swa : max_full;
                uint32_t sp[2] = { ATTN_SPLIT, (mx + ATTN_SPLIT - 1) / ATTN_SPLIT };
                [e setComputePipelineState:P(mt, [NSString stringWithFormat:@"attn_fd_%d", hd])];
                [e setBuffer:qb offset:0 atIndex:0];
                [e setBuffer:pool offset:0 atIndex:1];
                [e setBuffer:B(mt, l->swa ? @"ri_swa" : @"ri_full", sizeof(RowInfo) * n) offset:0 atIndex:2];
                [e setBuffer:B(mt, @"tab", 4 * tab_n) offset:0 atIndex:3];
                [e setBuffer:part offset:0 atIndex:4];
                set_bytes(e, &a, sizeof a, 5);
                set_bytes(e, sp, sizeof sp, 6);
                [e dispatchThreadgroups:MTLSizeMake(n, c->n_kv, sp[1]) threadsPerThreadgroup:MTLSizeMake(32 * (c->n_head / c->n_kv), 1, 1)];
                [e setComputePipelineState:P(mt, @"attn_reduce")];
                [e setBuffer:part offset:0 atIndex:0];
                [e setBuffer:gb offset:0 atIndex:1];
                [e setBuffer:ab offset:0 atIndex:2];
                set_bytes(e, &a, sizeof a, 3);
                set_bytes(e, sp, sizeof sp, 4);
                [e dispatchThreadgroups:MTLSizeMake(n, c->n_head, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            matmul(mt, e, &l->wo, ab, ob, d, n);
            if (l->post_attn_norm) rms(mt, e, ob, xb, l->post_attn_norm, d, c->post_eps, true, n, nil);
            else add_bcast(mt, e, xb, ob, 0, (uint32_t)(n * d), (size_t)n * d);
            rms(mt, e, xb, hb, l->ffn_norm, d, c->eps, false, n, nil);
            matmul(mt, e, &l->w_gate, hb, fgb, c->ff, n);
            matmul(mt, e, &l->w_up, hb, fub, c->ff, n);
            [e setComputePipelineState:P(mt, @"silu_mul")];
            [e setBuffer:fgb offset:0 atIndex:0];
            [e setBuffer:fub offset:0 atIndex:1];
            [e dispatchThreads:MTLSizeMake((size_t)n * c->ff, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            matmul(mt, e, &l->w_down, fgb, ob, d, n);
            if (l->post_ffn_norm) rms(mt, e, ob, xb, l->post_ffn_norm, d, c->post_eps, true, n, nil);
            else add_bcast(mt, e, xb, ob, 0, (uint32_t)(n * d), (size_t)n * d);
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
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        free(toks); free(posv); free(tab); free(off_full); free(off_swa); free(ri_full); free(ri_swa);
        free(slot_full); free(slot_swa); free(sel);
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "metal: command buffer failed (status %ld): %s\n", (long)cb.status, cb.error.localizedDescription.UTF8String);
            return 1;
        }
        if (ns) memcpy(logits, lg.contents, sizeof(float) * (size_t)ns * c->vocab);
        return 0;
    }
}

static void metal_destroy(backend *b) {
    CFBridgingRelease(b->impl);
    free(b);
}

backend *metal_backend_new(const model *m) {
    @autoreleasepool {
        CIMetal *mt = [CIMetal new];
        mt->dev = MTLCreateSystemDefaultDevice();
        if (!mt->dev) { fprintf(stderr, "metal: no device\n"); exit(1); }
        mt->queue = [mt->dev newCommandQueue];
        NSString *src = [[NSString alloc] initWithBytes:kernels_metal length:kernels_metal_len encoding:NSUTF8StringEncoding];
        NSError *err = nil;
        id<MTLLibrary> lib = [mt->dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
        if (!lib) { fprintf(stderr, "metal: kernel compile failed: %s\n", err.localizedDescription.UTF8String); exit(1); }
        mt->pso = [NSMutableDictionary new];
        for (NSString *name in lib.functionNames) {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            id<MTLComputePipelineState> p = [mt->dev newComputePipelineStateWithFunction:fn error:&err];
            if (!p) { fprintf(stderr, "metal: pipeline %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
            mt->pso[name] = p;
        }
        mt->bufs = [NSMutableDictionary new];
        // the whole mmap as one no-copy buffer (page-aligned base, length rounded to 16 KB)
        mt->wbase = m->g->map;
        mt->wlen = (m->g->size + 16383) / 16384 * 16384;
        mt->wbuf = [mt->dev newBufferWithBytesNoCopy:(void *)m->g->map length:mt->wlen options:MTLResourceStorageModeShared deallocator:nil];
        if (!mt->wbuf) { fprintf(stderr, "metal: could not wrap the model mapping (%zu bytes)\n", mt->wlen); exit(1); }
        mt->invf = [mt->dev newBufferWithBytes:m->inv_freq length:sizeof(float) * m->c.hd / 2 options:MTLResourceStorageModeShared];
        backend *b = calloc(1, sizeof *b);
        b->name = "metal";
        b->kv_elem = 2;
        b->alloc_kv = metal_alloc;
        b->forward = metal_forward;
        b->destroy = metal_destroy;
        b->impl = (__bridge_retained void *)mt;
        return b;
    }
}
