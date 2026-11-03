"""Declarative task smoke checks, not a general capability benchmark.

Checks JSON, exact answers and Python ASTs without executing generated code.
Expected results live beside the prompts in the --tasks JSONL file.
"""

import argparse
import ast
import json
from pathlib import Path


def clean(text):
    for stop in ("<|eot|>", "<|eot_id|>", "<|im_end|>", "<|end_of_text|>"):
        text = text.split(stop)[0]
    return text.strip()


def python_ast(text, normalize=False):
    tree = ast.parse(text)
    if normalize:
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                node.annotation = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.returns = None
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def check(row, spec):
    text = clean(row["text"])
    try:
        if spec["kind"] == "exact":
            expected = spec["value"]
            if spec.get("strip_punctuation"):
                text = text.rstrip(".!")
            if spec.get("ignore_case"):
                text, expected = text.lower(), expected.lower()
            return text == expected
        if spec["kind"] == "json":
            return json.loads(text) == spec["value"]
        if spec["kind"] == "json_fields":
            value = json.loads(text)
            return isinstance(value, dict) and all(
                key in value and value[key] == expected
                for key, expected in spec["value"].items()
            )
        if spec["kind"] == "python_ast":
            norm = spec.get("ignore_annotations_and_docstrings", False)
            return python_ast(text, norm) == python_ast(spec["value"], norm)
        if spec["kind"] == "content_smoke":
            text = text.lower()
            if any(x.lower() in text for x in spec.get("forbidden", [])):
                return False
            matched = all(x.lower() in text for x in spec.get("all", [])) and (
                not spec.get("any") or any(x.lower() in text for x in spec["any"])
            )
            # A speed sample may end before the explanation; missing keywords
            # then give no quality verdict. Known explicit errors still fail.
            return True if matched else None
    except (ValueError, SyntaxError, TypeError):
        return False
    raise ValueError(f"Unknown check: {spec['kind']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", nargs="+", type=Path)
    p.add_argument(
        "--tasks", type=Path, default=Path(__file__).with_name("pi-quality.jsonl")
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    tasks = {r["id"]: r for r in map(json.loads, args.tasks.read_text().splitlines())}
    result = {}
    for directory in args.directories:
        for path in directory.glob("*.jsonl"):
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            if not rows:
                continue
            key = str(path)
            result[key] = {
                "checks": {r["id"]: check(r, tasks[r["id"]]["check"]) for r in rows},
                "decode_rates": {r["id"]: r["decode_tokens_per_second"] for r in rows},
            }
            print(key, json.dumps(result[key]["checks"]))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
