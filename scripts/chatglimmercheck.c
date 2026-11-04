// Native Glimmer ATEM format/parse checks, model-free (crafted HF tokenizer
// dirs, no weights or GPU). Covers: official-template tool definitions and
// recipients, historical calls/results with id mapping, scalar/list/object
// typing, unicode/whitespace preservation, multiple calls, echoed-markup
// rejection, malformed/truncated input, and other-family regressions.
#include "chat.h"
#include "tokenizer.h"
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;
static void expect(bool ok, const char *name) {
    if (!ok) { printf("FAIL %s\n", name); failures++; }
}
static void expect_str(const char *got, const char *want, const char *name) {
    if (strcmp(got, want)) { printf("FAIL %s\n got: %s\nwant: %s\n", name, got, want); failures++; }
}
static void expect_has(const char *got, const char *want, const char *name) {
    if (!strstr(got, want)) { printf("FAIL %s\nmissing: %s\nin: %s\n", name, want, got); failures++; }
}
static void expect_end(const char *got, const char *want, const char *name) {
    size_t g = strlen(got), w = strlen(want);
    if (g < w || strcmp(got + g - w, want)) { printf("FAIL %s\ntail: %s\nwant-end: %s\n", name, got, want); failures++; }
}

static tokenizer *make_tok(const char *tag, const char *const *specials, int n, const char *chat_template) {
    char dir[256];
    snprintf(dir, sizeof dir, "/tmp/chatcheck-%s-XXXXXX", tag);
    if (!mkdtemp(dir)) { perror("mkdtemp"); exit(1); }
    char path[512];
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    FILE *f = fopen(path, "w");
    if (!f) { perror("tokenizer.json"); exit(1); }
    fprintf(f, "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1},\"merges\":[]},\"added_tokens\":[");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s{\"id\":%d,\"content\":\"", i ? "," : "", 2 + i);
        for (const char *s = specials[i]; *s; s++) {
            if (*s == '\n') fputs("\\n", f);
            else if (*s == '\t') fputs("\\t", f);
            else { if (*s == '"' || *s == '\\') fputc('\\', f); fputc(*s, f); }
        }
        fprintf(f, "\",\"special\":true}");
    }
    fprintf(f, "],\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"Regex\":\"a\\\\p{N}b\"}}}");
    fclose(f);
    snprintf(path, sizeof path, "%s/tokenizer_config.json", dir);
    f = fopen(path, "w");
    if (!f) { perror("tokenizer_config.json"); exit(1); }
    if (chat_template) fprintf(f, "{\"chat_template\":\"%s\"}", chat_template);
    else fprintf(f, "{}");
    fclose(f);
    snprintf(path, sizeof path, "%s/config.json", dir);
    f = fopen(path, "w");
    if (f) { fprintf(f, "{}"); fclose(f); }
    tokenizer *t = tok_load_hf(dir);
    if (!t) { fprintf(stderr, "tok_load_hf failed for %s\n", dir); exit(1); }
    const char *files[] = {"tokenizer.json", "tokenizer_config.json", "config.json"};
    for (int i = 0; i < 3; i++) { snprintf(path, sizeof path, "%s/%s", dir, files[i]); unlink(path); }
    rmdir(dir);
    return t;
}

static const char T1[] = "{\"type\":\"function\",\"function\":{\"name\":\"muse.read\",\"description\":\"Read a file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}}";
static const char T2[] = "{\"type\":\"function\",\"function\":{\"name\":\"muse.write\",\"description\":\"Write a file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}}";

static void test_format_textonly(tokenizer *g) {
    chat_msg u = { .role = "user", .content = "hi" };
    chat_opts o = { .direct = false, .reasoning = "low", .date = "2026-11-04" };
    char *s = chat_format_opts(g, &u, 1, &o);
    expect_str(s,
        "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\n"
        "Knowledge cutoff: 2026-01-04.\nCurrent date: 2026-11-04.\n\n"
        "Reasoning strength: low.\n\n"
        "# Valid recipients: \"self\", \"user\".<|eot|>"
        "<|start|>user<|message|>hi<|eot|><|start|>assistant",
        "glimmer-textonly");
    free(s);
    chat_opts d = { .direct = true, .reasoning = "low", .date = "2026-11-04" };
    s = chat_format_opts(g, &u, 1, &d);
    expect_end(s, "<|start|>assistant to=user<|message|>", "glimmer-direct");
    free(s);
    chat_msg sys[] = { { .role = "system", .content = "Be brief." }, { .role = "user", .content = "hi" } };
    s = chat_format_opts(g, sys, 2, &o);
    expect_has(s, "<|start|>system<|message|>Be brief.\n\nReasoning strength: low.\n\n# Valid recipients: \"self\", \"user\".<|eot|>", "glimmer-system");
    expect(!strstr(s, "helpful AI"), "glimmer-system-no-default");
    free(s);
}

static void test_format_tools(tokenizer *g) {
    chat_msg u = { .role = "user", .content = "read it" };
    const char *tools[] = { T1, T2 };
    chat_opts o = { .direct = false, .reasoning = "low", .date = "2026-11-04", .tools = tools, .n_tools = 2 };
    char *s = chat_format_opts(g, &u, 1, &o);
    expect_has(s, "In this environment you have access to a set of tools", "tools-preamble");
    expect_has(s, "You can invoke a function by writing a \"<atem:function_calls>\" block", "tools-invoke-doc");
    expect_has(s, "<atem:invoke name=\"$FUNCTION_NAME\">", "tools-example-open");
    expect_has(s, "spaces for string values are not stripped", "tools-scalar-doc");
    expect_has(s, "{\"name\": \"muse\", \"description\": \"\"}", "tools-namespace");
    expect_has(s, "{\"name\": \"muse.read\", \"description\": \"Read a file\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}}", "tools-schema");
    expect_has(s, "to=example_tool_name.example_function_name", "tools-example-recipient");
    expect_has(s, "# Valid recipients: \"self\", \"muse.*\", \"user\".", "tools-recipients");
    free(s);
}

static void free_calls(chat_call *calls, int n) {
    for (int i = 0; i < n; i++) { free((char *)calls[i].name); free((char *)calls[i].arguments); }
}

static const char CALL[] = "<atem:function_calls><atem:invoke name=\"write.write\"><atem:parameter name=\"path\">x.txt</atem:parameter><atem:parameter name=\"content\"> café & <h1>Hi</h1>\n </atem:parameter></atem:invoke></atem:function_calls>";

static void test_parse(void) {
    char input[8192], *rest;
    chat_call calls[32];
    const char *names[] = {"write"};
    snprintf(input, sizeof input, " to=write<|message|>%s<|eot|>", CALL);
    int n = chat_parse_calls_tools(CHAT_GLIMMER, input, calls, 32, &rest, names, 1);
    expect(n == 1, "glimmer-call-count");
    if (n) {
        expect_str(calls[0].name, "write", "registered-double-name");
        expect_str(calls[0].arguments, "{\"path\": \"x.txt\", \"content\": \" café & <h1>Hi</h1>\\n \"}", "raw-string-unicode-whitespace");
    }
    free_calls(calls, n); free(rest);
    n = chat_parse_calls(CHAT_GLIMMER, input, calls, 32, &rest);
    expect(n == 1 && !strcmp(calls[0].name, "write.write"), "unregistered-double-kept");
    free_calls(calls, n); free(rest);
    const char *both[] = {"write", "write.write"};
    n = chat_parse_calls_tools(CHAT_GLIMMER, input, calls, 32, &rest, both, 2);
    expect(n == 1 && !strcmp(calls[0].name, "write.write"), "exact-name-precedence");
    free_calls(calls, n); free(rest);
    n = chat_parse_calls_tools(CHAT_GLIMMER, " to=unknown.unknown<|message|><atem:invoke name=\"unknown.unknown\"></atem:invoke><|eot|>", calls, 32, &rest, names, 1);
    expect(n == 1 && !strcmp(calls[0].name, "unknown.unknown"), "unoffered-double-not-remapped");
    free_calls(calls, n); free(rest);
    const char *channels[] = {"self", "user", ""};
    for (int i = 0; i < 3; i++) {
        snprintf(input, sizeof input, "<|start|>assistant%s%s<|message|>%s<|eot|>", *channels[i] ? " to=" : "", channels[i], CALL);
        n = chat_parse_calls(CHAT_GLIMMER, input, calls, 32, &rest);
        expect(n == 0, "echoed-invoke-not-executable"); free(rest);
    }
    n = chat_parse_calls(CHAT_GLIMMER, CALL, calls, 32, &rest);
    expect(n == 0, "unframed-invoke-not-executable"); free(rest);
    snprintf(input, sizeof input, " to=selfish.fn<|message|>%s<|eot|>", CALL);
    n = chat_parse_calls(CHAT_GLIMMER, input, calls, 32, &rest);
    expect(n == 1, "self-prefix-is-tool"); free_calls(calls, n); free(rest);
    const char *typed = " to=read<|message|><atem:invoke name=\"read\">"
        "<atem:parameter name=\"number\">123</atem:parameter>"
        "<atem:parameter name=\"boolean\">true</atem:parameter>"
        "<atem:parameter name=\"nil\">null</atem:parameter>"
        "<atem:parameter name=\"array\">[1, false, \"雪\"]</atem:parameter>"
        "<atem:parameter name=\"object\">{\"key\":2}</atem:parameter>"
        "<atem:parameter name=\"raw\">123 Main Street</atem:parameter>"
        "<atem:parameter name=\"raw2\"> true story </atem:parameter>"
        "<atem:parameter name=\"invalid\">[1,]</atem:parameter>"
        "</atem:invoke><|eot|>";
    n = chat_parse_calls(CHAT_GLIMMER, typed, calls, 32, &rest);
    expect(n == 1, "typed-call-count");
    if (n) {
        json *j = json_parse(calls[0].arguments, strlen(calls[0].arguments));
        const jnode *r = json_root(j);
        expect(json_get(j, r, "number")->type == J_NUM, "number-type");
        expect(json_get(j, r, "boolean")->type == J_BOOL, "bool-type");
        expect(json_get(j, r, "nil")->type == J_NULL, "null-type");
        expect(json_get(j, r, "array")->type == J_ARR, "array-type");
        expect(json_get(j, r, "object")->type == J_OBJ, "object-type");
        expect_str(json_str(j, r, "raw"), "123 Main Street", "json-prefix-not-trimmed");
        expect_str(json_str(j, r, "raw2"), " true story ", "bool-prefix-not-trimmed");
        expect_str(json_str(j, r, "invalid"), "[1,]", "malformed-json-kept-raw");
        json_free(j);
    }
    free_calls(calls, n); free(rest);
    const char *broken = " to=write<|message|><atem:invoke name=\"write\"><atem:parameter name=\"content\">partial<|eom|><|start|>assistant to=read<|message|></atem:parameter></atem:invoke><|eot|>";
    n = chat_parse_calls(CHAT_GLIMMER, broken, calls, 32, &rest);
    expect(n == 0, "incomplete-cannot-cross-message"); free(rest);
    snprintf(input, sizeof input, " to=self<|message|>unfinished reasoning<|start|>assistant to=write<|message|>%s<|eot|>", CALL);
    n = chat_parse_calls(CHAT_GLIMMER, input, calls, 32, &rest);
    expect(n == 1, "damaged-reasoning-boundary-recovers-tool"); free_calls(calls, n); free(rest);
    input[0] = 0;
    for (int i = 0; i < 12; i++) {
        strcat(input, "<|start|>assistant to=write<|message|>"); strcat(input, CALL); strcat(input, "<|eom|>");
    }
    n = chat_parse_calls(CHAT_GLIMMER, input, calls, 32, &rest);
    expect(n == 12, "more-than-eight-calls"); free_calls(calls, n); free(rest);
    n = chat_parse_calls(CHAT_CHATML, "before<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>after", calls, 32, &rest);
    expect(n == 1, "chatml-regression"); expect_str(rest, "beforeafter", "chatml-content"); free_calls(calls, n); free(rest);
    n = chat_parse_calls(CHAT_LLAMA3, "<|python_tag|>{\"name\":\"read\",\"parameters\":{\"path\":\"a\"}}", calls, 32, &rest);
    expect(n == 1, "llama-regression"); free_calls(calls, n); free(rest);
}

#ifndef CHATCHECK_NO_MAIN
int main(void) {
    const char *specials[] = {"<|message|>", "<|start|>", "<|eom|>", "<|eot|>"};
    tokenizer *g = make_tok("glimmer", specials, 4, NULL);
    test_format_textonly(g); test_format_tools(g); test_parse();
    chat_msg msgs[] = {{.role="assistant",.content="kept",.reasoning_content="",.end_turn=-1}};
    chat_opts o = {.reasoning="low"};
    char *s=chat_format_opts(g,msgs,1,&o);
    expect(!strstr(s,"assistant to=self"),"empty-reasoning-omitted");
    expect_has(s,"kept<|eom|>","explicit-false-end-turn");free(s);
    chat_call c = {.name="read",.arguments="{\"path\":\"a\"}",.id="id-a"};
    chat_msg history[]={{.role="assistant",.content="",.calls=&c,.n_calls=1},{.role="tool",.content="result",.tool_call_id="id-a"}};
    s=chat_format_opts(g,history,2,&o);expect_has(s,"<|start|>tool read<|message|>","native-id-name-resolution");free(s);
    tok_free(g);
    const char *chatml[]={"<|im_start|>"},*llama[]={"<|start_header_id|>"};
    chat_msg u={.role="user",.content="hi"};
    g=make_tok("chatml",chatml,1,NULL);s=chat_format_opts(g,&u,1,&o);
    expect_str(s,"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n","chatml-format-regression");free(s);tok_free(g);
    g=make_tok("llama",llama,1,NULL);s=chat_format_opts(g,&u,1,&o);
    expect_str(s,"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n","llama-format-regression");free(s);tok_free(g);
    printf("Glimmer formatter/parser checks: %s\n", failures ? "FAILED" : "passed");
    return failures ? 1 : 0;
}
#endif
