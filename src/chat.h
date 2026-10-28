// Chat formatting for the supported template families, detected from the
// vocabulary: Llama 3 headers, ChatML (Qwen; optional no-think block) and Muse
// Glimmer channels. Templates are built in rather than interpreted (no Jinja).
#pragma once
#include "tokenizer.h"
#include <stdbool.h>

typedef struct { const char *name, *arguments; } chat_call; // arguments: a JSON object's text

// role "tool" carries a tool result; an assistant message may carry tool calls
typedef struct { const char *role, *content; const chat_call *calls; int n_calls; } chat_msg;

typedef struct {
    bool direct;           // answer without a reasoning channel (Qwen3 no-think, Glimmer to=user)
    const char *reasoning; // Glimmer reasoning strength ("low" / "medium" / "high"); NULL: template default
    const char *date;      // "YYYY-MM-DD" for templates that state the current date; NULL: omit
    const char **tools; int n_tools; // tool definitions (OpenAI format, JSON text each), shown as the template does
} chat_opts;

typedef enum { CHAT_LLAMA3, CHAT_CHATML, CHAT_GLIMMER, CHAT_PLAIN } chat_family;
chat_family chat_detect(const tokenizer *t);

// Format messages plus the assistant generation prompt. Returns a malloc'd string.
char *chat_format_opts(const tokenizer *t, const chat_msg *msgs, int n, const chat_opts *o);
// Tool calls in a generated reply, in the family's format (Llama 3: a JSON
// {"name", "parameters"} object, optionally after <|python_tag|>; ChatML:
// <tool_call> blocks). Fills up to max calls (malloc'd strings) and returns the
// count; *rest gets the reply text outside the calls (malloc'd).
int chat_parse_calls(chat_family f, const char *reply, chat_call *calls, int max, char **rest);
// Shorthand used by the CLI tools (Glimmer: low reasoning strength).
char *chat_format(const tokenizer *t, const chat_msg *msgs, int n, bool direct);
