// OpenAI-compatible HTTP server over one engine. Every in-flight request is a
// sequence in the engine, so concurrent clients (agents, eval harness workers)
// are continuously batched together, share prompt prefixes and, with a
// drafter, decode speculatively.
//
//   GET  /health, /v1/models
//   GET  /v1/fleet/capabilities   supported features (tools, reasoning,
//                                 streaming, admission); incremental tool
//                                 streaming is false (calls buffer to a whole)
//   GET  /v1/fleet/stats          live engine numbers: active/queued/served,
//                                 KV free/total (full and SWA pools), shared /
//                                 generated / prefill counters
//   POST /v1/chat/completions   messages, max_tokens | max_completion_tokens,
//                               temperature, top_p, seed, stop, stream,
//                               stream_options.include_usage, reasoning_effort,
//                               tools, tool_choice
//   POST /v1/completions        prompt (raw text, no chat template)
//
// Reasoning (Glimmer "to=self" channel, Qwen3 <think>) is returned in
// message.reasoning_content, the answer in message.content. Glimmer tool
// calls use the native ATEM format: only tool-recipient channels parse;
// echoes in reasoning or the final answer stay text, and truncated
// (length-capped) calls never execute.
#pragma once
#include "engine.h"
#include <stdbool.h>

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

// True when the client's socket read EOF or hard-failed: a nonblocking peek
// that never disturbs a live keep-alive connection. Used to cancel queued and
// buffered (tool) work nobody will read.
bool fleet_peer_gone(int fd);
