// OpenAI-compatible HTTP server over one engine. Every in-flight request is a
// sequence in the engine, so concurrent clients (agents, eval harness workers)
// are continuously batched together, share prompt prefixes and, with a
// drafter, decode speculatively.
//
//   GET  /health, /v1/models
//   POST /v1/chat/completions   messages, max_tokens | max_completion_tokens,
//                               temperature, top_p, seed, stop, stream,
//                               stream_options.include_usage, reasoning_effort
//   POST /v1/completions        prompt (raw text, no chat template)
//
// Reasoning (Glimmer "to=self" channel, Qwen3 <think>) is returned in
// message.reasoning_content, the answer in message.content.
#pragma once
#include "engine.h"

typedef struct {
    const char *host;      // bind address (default 127.0.0.1)
    int port;
    const char *model_id;  // name reported by /v1/models and in responses
    int default_max_tokens;
    const char *reasoning; // default strength; "none" requests a direct-answer prefix
    int max_active;        // requests decoded at once; the rest wait in a queue
    bool admit;            // admit only what fits in KV memory (default with --swap-gb: the
                           // rest wait with their KV resident or swapped, not recomputed)
} server_opts;

// Serve forever (returns only on a socket error).
int server_run(engine *e, const server_opts *o);
