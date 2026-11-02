// Shared helpers for the command-line tools.
#pragma once
#include "backend.h"
#include "chat.h"
#include "engine.h"
#include "model.h"
#include "threadpool.h"
#include "tokenizer.h"
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg_str(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static double arg_num(int argc, char **argv, const char *name, double def) {
    const char *s = arg_str(argc, argv, name, NULL);
    return s ? atof(s) : def;
}
static bool arg_flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], name)) return true;
    return false;
}

// Model aliases resolve inside the Hugging Face cache; anything else is a path.
static const char *model_path(const char *alias) {
    static char path[4096];
    const char *pat = NULL;
    if (!strcmp(alias, "glimmer")) pat = "models--unsloth--Muse-Glimmer-30B-GGUF/snapshots/*/Muse-Glimmer-30B-UD-Q2_K_XL.gguf";
    else if (!strcmp(alias, "llama1b")) pat = "models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/*/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    else if (!strcmp(alias, "qwen3")) pat = "models--unsloth--Qwen3-1.7B-GGUF/snapshots/*/Qwen3-1.7B-Q4_K_M.gguf";
    else if (!strcmp(alias, "glimmer-mlx4")) pat = "models--mlx-community--Muse-Glimmer-30B-4bit/snapshots/*";
    else if (!strcmp(alias, "llama1b-mlx4")) pat = "models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/*";
    else if (!strcmp(alias, "llama8b-mlx4")) pat = "models--mlx-community--Meta-Llama-3.1-8B-Instruct-4bit/snapshots/*";
    else if (!strcmp(alias, "glimmer-draft")) pat = "models--meta-models--Muse-Glimmer-30B-assistant/snapshots/*"; // Meta's DFlash drafter
    if (!pat) return alias;
    char g[4096];
    snprintf(g, sizeof g, "%s/.cache/huggingface/hub/%s", getenv("HOME"), pat);
    glob_t gl;
    if (glob(g, 0, NULL, &gl) != 0 || gl.gl_pathc == 0) { fprintf(stderr, "model %s not found (%s)\n", alias, g); exit(1); }
    snprintf(path, sizeof path, "%s", gl.gl_pathv[0]);
    globfree(&gl);
    return path;
}

typedef struct { model *m, *draft; backend *be; tokenizer *tok; engine *e; } runtime;

// Load model + backend + engine; the KV pool gets what the memory budget leaves.
// --draft M enables speculative decoding with a DFlash drafter (--spec-k drafts
// per verify step, --draft-kv-gb for its KV); its weights count against the budget.
// --ngram drafts from repeated text instead of / alongside a drafter (no memory).
static runtime load_runtime(int argc, char **argv) {
    runtime rt;
    double t = now_secs();
    int threads = (int)arg_num(argc, argv, "--threads", 0); // CPU worker threads (0: performance cores)
    if (threads > 0) tp_init(threads);
    const char *path = model_path(arg_str(argc, argv, "--model", "llama1b"));
    rt.m = model_load(path);
    if (!rt.m) exit(1);
    rt.tok = rt.m->g ? tok_load(rt.m->g) : tok_load_hf(rt.m->dir);
    if (!rt.tok) exit(1);
    double budget_gb = arg_num(argc, argv, "--budget-gb", 16);
    size_t streamed = model_streamed_bytes(rt.m);
    const char *dpath = arg_str(argc, argv, "--draft", NULL);
    rt.draft = dpath ? model_load(model_path(dpath)) : NULL;
    if (dpath && !rt.draft) exit(1);
    size_t dkv = (size_t)(arg_num(argc, argv, "--draft-kv-gb", 0.5) * 1e9);
    if (rt.draft) streamed += model_streamed_bytes(rt.draft) + dkv;
    size_t reserve = 600u << 20; // activations, logits, scratch, touched embedding rows
    size_t budget = (size_t)(budget_gb * 1e9);
    if (budget < streamed + reserve + (64u << 20)) { fprintf(stderr, "budget %.1f GB too small for %.2f GB of weights\n", budget_gb, streamed / 1e9); exit(1); }
    // --device auto|cpu|cpu-ref|metal (--backend: older spelling); auto = metal when built with it
    rt.be = backend_create(arg_str(argc, argv, "--device", arg_str(argc, argv, "--backend", "auto")), rt.m);
    size_t pinned = arg_flag(argc, argv, "--no-pin") ? 0 : model_pin(rt.m);
    size_t kv = budget - streamed - reserve;
    double kv_gb = arg_num(argc, argv, "--kv-gb", 0); // cap the KV pool (it otherwise takes the rest of the budget)
    if (kv_gb > 0 && (size_t)(kv_gb * 1e9) < kv) kv = (size_t)(kv_gb * 1e9);
    if (arg_flag(argc, argv, "--kv-int8")) { // 8-bit KV (per-token-per-head scales): ~2x the tokens per GB
        rt.be->kv_elem = 1;
    }
    rt.e = engine_new(rt.m, rt.be, rt.tok, kv);
    rt.e->row_budget = (int)arg_num(argc, argv, "--row-budget", 128);
    rt.e->spec_rows = (int)arg_num(argc, argv, "--spec-rows", 32);
    if (arg_flag(argc, argv, "--ngram")) engine_set_ngram(rt.e, true); // drafts from repeated text (model-free)
    if (arg_flag(argc, argv, "--reuse-spans")) // move KV of text other sequences already hold (lossy, see spans.h)
        engine_set_reuse(rt.e, (int)arg_num(argc, argv, "--reuse-min", 256), (int)arg_num(argc, argv, "--reuse-head", 64), (int)arg_num(argc, argv, "--reuse-tail", 64));
    double swap_gb = arg_num(argc, argv, "--swap-gb", 0); // KV swap space on disk (0: off; evicted KV is recomputed)
    if (swap_gb > 0) {
        rt.e->swap_dir = arg_str(argc, argv, "--swap-dir", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
        rt.e->swap_limit = (size_t)(swap_gb * 1e9);
    }
    if (rt.draft && !engine_set_drafter(rt.e, rt.draft, dkv, (int)arg_num(argc, argv, "--spec-k", 0))) exit(1);
    fprintf(stderr, "[load] %.1fs %s %s, %d layers, weights %.2f GB (pinned %.2f), KV %.2f GB (%u full + %u swa pages), backend %s\n",
            now_secs() - t, arch_name(rt.m->c.arch), path + (strlen(path) > 60 ? strlen(path) - 60 : 0), rt.m->c.n_layer, streamed / 1e9,
            pinned / 1e9, (rt.e->kv.full.bytes + rt.e->kv.swa.bytes) / 1e9, rt.e->kv.full.n_pages, rt.e->kv.swa.n_pages, rt.be->name);
    return rt;
}

static int encode_chat(const tokenizer *tok, const char *user, bool direct, int32_t **ids) {
    chat_msg msg = { .role = "user", .content = user };
    char *text = chat_format(tok, &msg, 1, direct);
    int n = tok_encode(tok, text, strlen(text), ids);
    free(text);
    return n;
}
