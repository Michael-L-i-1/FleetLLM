// Chat formatting for the supported template families, detected from the
// vocabulary: Llama 3 headers, ChatML (Qwen; optional no-think block) and Muse
// Glimmer channels. Templates are built in rather than interpreted (no Jinja).
//
// Glimmer follows the checkpoint's chat_template.jinja, including its native
// ATEM tool contract: tool definitions and the recipient list in the system
// message, tool-recipient assistant blocks for calls (current and historical),
// and role=tool results. Non-tool text-only behavior is unchanged.
#pragma once
#include "tokenizer.h"
#include <stdbool.h>

// arguments: a JSON object's text. id: the OpenAI tool-call id for a
// historical call (NULL when absent); used to resolve role=tool results.
typedef struct { const char *name, *arguments, *id; } chat_call;

// role "tool" carries a tool result; an assistant message may carry tool calls.
// The extra fields mirror the chat template's message keys; they are NULL
// unless the caller sets them (the server fills them from the request JSON).
typedef struct {
    const char *role, *content;
    const chat_call *calls; int n_calls;
    const char *reasoning_content; // assistant to=self channel (Glimmer)
    const char *recipient;         // assistant recipient override (default "user")
    int end_turn;                  // 0: template default (end iff recipient is "user"); 1: force <|eot|>; -1: force <|eom|>
    const char *tool_call_id;      // role=tool: the call this result answers
    const char *name;              // role=tool: the tool name (else resolved via tool_call_id)
} chat_msg;

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
// <tool_call> blocks; Glimmer: ATEM blocks in tool-recipient channels only).
// Fills up to max calls (malloc'd strings) and returns the count; *rest gets
// the reply text outside the calls (malloc'd).
int chat_parse_calls(chat_family f, const char *reply, chat_call *calls, int max, char **rest);
// As above, binding only doubled bare Glimmer names to an offered tool when
// the emitted full name is not itself registered. Unknown names stay intact.
int chat_parse_calls_tools(chat_family f, const char *reply, chat_call *calls, int max, char **rest,
                           const char *const *names, int n_names);
// Shorthand used by the CLI tools (Glimmer: low reasoning strength).
char *chat_format(const tokenizer *t, const chat_msg *msgs, int n, bool direct);
