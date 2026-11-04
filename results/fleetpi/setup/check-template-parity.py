#!/usr/bin/env python3
"""Compare native Glimmer formatting to independent HF Jinja golden bytes.

Only chat.c/json.c are compiled, with tokenizer detection stubs. No model,
tokenizer data, engine, Metal code, or inference server is loaded. Fixture
messages are translated independently into C initializers; tool result names
are resolved here from explicit names/IDs, so this does NOT test HTTP ingress
or the server's ID-to-name mapping. Expected output is never derived from C.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCE = HERE / "reference"


def encoded(value):
    """Use fixed-width octal bytes, avoiding C literal/Unicode ambiguity."""
    if value is None:
        return "NULL"
    if "\0" in value:
        raise ValueError("Embedded NUL is outside the chat_msg C-string contract")
    return '"' + ''.join(f"\\{byte:03o}" for byte in value.encode("utf-8")) + '"'


def serialized(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def content_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(part.get("type") == "text" for part in value):
        return "".join(part["text"] for part in value)
    raise ValueError("Only text fixture content is supported")


def result_name(message, messages):
    if message.get("name"):
        return message["name"]
    call_id = message.get("tool_call_id")
    name = call_id or ""
    for previous in messages:
        for call in previous.get("tool_calls", []):
            if call_id is not None and call.get("id") == call_id:
                name = call["function"]["name"]
    return name


def generate_runner(cases):
    lines = [
        '#include "chat.h"', '#include <stdio.h>', '#include <stdlib.h>',
        '#include <string.h>',
        'int32_t tok_lookup(const tokenizer *t, const char *s) { (void)t; return !strcmp(s, "<|message|>") ? 1 : -1; }',
        'bool tok_qwen35(const tokenizer *t) { (void)t; return false; }',
        'const char *tok_chat_template(const tokenizer *t) { (void)t; return ""; }',
    ]
    for index, case in enumerate(cases):
        opts = case["render_options"]
        if opts.get("bos_token") != "<|begin_of_text|>" or opts.get("add_generation_prompt") is not True:
            raise ValueError("Fixtures must use the native BOS and generation prompt")
        if opts.get("tool_namespace_descriptions") or opts.get("knowledge_cutoff"):
            raise ValueError("Fixture options are not representable by chat_opts")
        lines.append(f"static char *case_{index}(void) {{")
        for msg_index, message in enumerate(case["messages"]):
            calls = message.get("tool_calls", [])
            if calls:
                lines.append(f"  const chat_call calls_{msg_index}[] = {{")
                for call in calls:
                    function = call["function"]
                    arguments = function["arguments"]
                    if not isinstance(arguments, dict):
                        raise ValueError("HF golden tool arguments must already be mappings")
                    lines.append("    { .name=%s, .arguments=%s, .id=%s }," % (
                        encoded(function["name"]), encoded(serialized(arguments)), encoded(call.get("id"))))
                lines.append("  };")
        lines.append("  const chat_msg messages[] = {")
        for msg_index, message in enumerate(case["messages"]):
            calls = message.get("tool_calls", [])
            end_turn = 0 if message.get("end_turn") is None else 1 if message["end_turn"] else -1
            name = result_name(message, case["messages"]) if message["role"] == "tool" else message.get("name")
            fields = {
                "role": encoded(message["role"]), "content": encoded(content_text(message.get("content"))),
                "calls": f"calls_{msg_index}" if calls else "NULL", "n_calls": str(len(calls)),
                "reasoning_content": encoded(message.get("reasoning_content")),
                "recipient": encoded(message.get("recipient")), "end_turn": str(end_turn),
                "tool_call_id": encoded(message.get("tool_call_id")), "name": encoded(name),
            }
            lines.append("    { " + ", ".join(f".{key}={value}" for key, value in fields.items()) + " },")
        lines.append("  };")
        tools = case.get("tools", [])
        if tools:
            lines.append("  const char *tools[] = {" + ", ".join(encoded(serialized(tool)) for tool in tools) + "};")
        lines.append("  const chat_opts options = { .direct=false, .reasoning=%s, .date=%s, .tools=%s, .n_tools=%d };" % (
            encoded(opts.get("reasoning_strength")), encoded(opts.get("current_date")), "tools" if tools else "NULL", len(tools)))
        lines.append(f"  return chat_format_opts(NULL, messages, {len(case['messages'])}, &options);")
        lines.append("}")
    lines.append("int main(int argc, char **argv) { if (argc != 2) return 2; char *out = NULL; switch (atoi(argv[1])) {")
    for index in range(len(cases)):
        lines.append(f"case {index}: out=case_{index}(); break;")
    lines.append("default: return 2; } if (!out) return 3; size_t n=strlen(out); int rc=fwrite(out, 1, n, stdout)==n ? 0 : 4; free(out); return rc; }")
    return "\n".join(lines) + "\n"


def generate_extra():
    # This mode uses the installed Transformers template compiler, never C.
    from transformers.utils.chat_template_utils import _compile_jinja_template

    base = json.loads((REFERENCE / "template-golden-fixtures.json").read_text())
    tools = base["cases"][0]["tools"]
    options = dict(base["cases"][0]["render_options"])
    nested = {"z": ["café", "<tag>&'", True, None, {"β": "雪"}], "a": {"count": 3, "enabled": False}}
    call = {"id": "call-nested", "type": "function", "function": {"name": "write", "arguments": nested}}
    cases = [
        {"name": "extra-nested-json-unicode", "messages": [{"role": "system", "content": "Résumé 雪"}, {"role": "assistant", "content": None, "tool_calls": [call]}], "tools": tools},
        {"name": "extra-tool-call-followed-by-assistant", "messages": [{"role": "system", "content": "System"}, {"role": "assistant", "content": None, "tool_calls": [call]}, {"role": "assistant", "content": "Continue"}], "tools": tools},
        {"name": "extra-empty-reasoning", "messages": [{"role": "system", "content": "System"}, {"role": "assistant", "reasoning_content": "", "content": "Hello"}], "tools": []},
        {"name": "extra-explicit-end-turn-false", "messages": [{"role": "assistant", "recipient": "user", "end_turn": False, "content": "Continued message"}], "tools": []},
        {"name": "extra-explicit-end-turn-true", "messages": [{"role": "assistant", "recipient": "self", "end_turn": True, "content": "Planning"}], "tools": []},
        {"name": "extra-no-system-date", "messages": [{"role": "user", "content": "Hello"}], "tools": tools},
        {"name": "extra-late-system", "messages": [{"role": "user", "content": "Hello"}, {"role": "system", "content": "Updated instruction"}, {"role": "assistant", "content": "Understood"}], "tools": tools},
        {"name": "extra-explicit-tool-name-overrides-id", "messages": [{"role": "assistant", "content": None, "tool_calls": [call]}, {"role": "tool", "name": "explicit_name", "tool_call_id": "call-nested", "content": "Result"}], "tools": tools},
        {"name": "extra-text-content-array", "messages": [{"role": "system", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}, {"role": "user", "content": [{"type": "text", "text": " café "}]}], "tools": []},
        {"name": "extra-shared-namespace-and-bare-schema", "messages": [{"role": "user", "content": "Inspect"}], "tools": [{"name": "files.read", "description": "Read café <p>&'", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}, {"name": "files.write", "description": "Write", "parameters": {"type": "object", "properties": {}}}]},
        {"name": "extra-float-values", "messages": [{"role": "assistant", "content": None, "tool_calls": [{"id": "float-call", "type": "function", "function": {"name": "read", "arguments": {"one": 1.0, "negative_zero": -0.0, "small": 1e-7}}}]}], "tools": tools},
    ]
    template = (REFERENCE / "glimmer-chat-template.jinja").read_text()
    compiled = _compile_jinja_template(template)
    for case in cases:
        case["render_options"] = options
        case["expected"] = compiled.render(messages=case["messages"], tools=case["tools"], **options)
    target = REFERENCE / "template-extra-golden-fixtures.json"
    target.write_text(json.dumps({"source": "Independent review cases rendered by HF _compile_jinja_template; no native formatter output", "template_sha256": hashlib.sha256(template.encode()).hexdigest(), "cases": cases}, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(cases)} independent goldens to {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-extra", action="store_true")
    parser.add_argument("--golden", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=HERE / "template-parity-latest")
    parser.add_argument("--cc", default="cc")
    args = parser.parse_args()
    if args.generate_extra:
        generate_extra()
        return 0
    golden_paths = args.golden or [REFERENCE / "template-golden-fixtures.json"]
    extra = REFERENCE / "template-extra-golden-fixtures.json"
    if args.golden is None and extra.exists():
        golden_paths.append(extra)
    cases = []
    inputs = {}
    for path in golden_paths:
        path = path.resolve()
        data = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        cases.extend(json.loads(data)["cases"])
    args.output.mkdir(parents=True, exist_ok=True)
    runner = generate_runner(cases)
    (args.output / "generated-fixtures.c").write_text(runner)
    # Snapshot all headers and the two C files before compiling, so concurrent
    # implementation edits cannot alter the already captured translation units.
    source_paths = sorted((ROOT / "src").glob("*.h")) + [ROOT / "src/chat.c", ROOT / "src/json.c"]
    sources = {path.name: path.read_bytes() for path in source_paths}
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    report = {"scope": "formatter bytes only; helper resolves tool names; server ingestion, tokenization, parsing and execution are not tested", "inputs": inputs, "source_sha256": source_hashes, "results": []}
    log = []
    with tempfile.TemporaryDirectory(prefix="fleetpi-template-parity-") as directory:
        build = Path(directory)
        for name, data in sources.items():
            (build / name).write_bytes(data)
        shutil.copyfile(args.output / "generated-fixtures.c", build / "runner.c")
        command = [args.cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O0", "-g", "-Wall", "-Wextra", "-I", str(build), str(build / "runner.c"), str(build / "chat.c"), str(build / "json.c"), "-o", str(build / "formatter")]
        compiled = subprocess.run(command, capture_output=True, text=True)
        report["compile"] = {"command": command, "returncode": compiled.returncode, "stdout": compiled.stdout, "stderr": compiled.stderr}
        log.append("CPU-only compile: " + str(compiled.returncode))
        if compiled.stdout or compiled.stderr:
            log.append(compiled.stdout + compiled.stderr)
        if compiled.returncode == 0:
            for index, case in enumerate(cases):
                result = subprocess.run([str(build / "formatter"), str(index)], capture_output=True)
                expected = case["expected"].encode("utf-8")
                actual = result.stdout
                passed = result.returncode == 0 and expected == actual
                row = {"name": case["name"], "passed": passed, "returncode": result.returncode, "expected_bytes": len(expected), "actual_bytes": len(actual), "expected_sha256": hashlib.sha256(expected).hexdigest(), "actual_sha256": hashlib.sha256(actual).hexdigest(), "stderr": result.stderr.decode("utf-8", errors="replace")}
                prefix = args.output / f"{index:02d}-{case['name']}"
                prefix.with_suffix(".expected.txt").write_bytes(expected)
                prefix.with_suffix(".actual.txt").write_bytes(actual)
                if not passed:
                    offset = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
                    row["first_difference_byte"] = offset
                    row["expected_context"] = repr(expected[max(0, offset - 50):offset + 100])
                    row["actual_context"] = repr(actual[max(0, offset - 50):offset + 100])
                    diff = "".join(difflib.unified_diff(expected.decode("utf-8").splitlines(True), actual.decode("utf-8", errors="replace").splitlines(True), fromfile="official-jinja", tofile="native-chat-format"))
                    prefix.with_suffix(".diff.txt").write_text(diff)
                    log.append(f"FAIL {case['name']}: first byte {offset}\n  expected {row['expected_context']}\n  actual   {row['actual_context']}")
                else:
                    log.append(f"PASS {case['name']}: {len(expected)} identical bytes")
                report["results"].append(row)
    changed = [path.name for path in source_paths if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[path.name]]
    report["sources_changed_during_run"] = changed
    report["passed"] = sum(row["passed"] for row in report["results"])
    report["total"] = len(cases)
    report["all_passed"] = compiled.returncode == 0 and report["passed"] == len(cases)
    log.append(f"Result: {report['passed']}/{len(cases)} exact-byte fixtures passed. Sources changed during run: {changed}")
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "results.log").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
