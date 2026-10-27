"""Compare build/tokcheck (C tokenizer) against the HF reference tokenizer.

    python scripts/tokcheck.py MODEL.gguf TOKENIZER.json [samples.txt]
"""
import subprocess
import sys

from tokenizers import Tokenizer

SAMPLES = [
    "Hello, world! It's 2026 and we're testing 1234567 tokens.",
    "  leading spaces\n\nand\ttabs, unicode: café naïve 東京 🚀 — dashes",
    "def f(x):\n    return x**2  # comment\n\n\n",
    "I'll we've they'd DON'T CamelCaseWords snake_case_words https://example.com/a/b?c=d",
    "MIXEDcase ALLCAPS Ünïcödé ÀÉÎ ﬁligatures ½ ² ٣ १२३ ٫",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "<|start|>user<|message|>What is 2+2?<|eot|><|start|>assistant to=user<|message|>",
    "   \n  \r\n x  ",
    "a\n\n\nb    c\t\t\td /// ==== !!!? ...",
]

gguf_path, tok_json = sys.argv[1], sys.argv[2]
path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/cinfer_tok_samples.txt"
open(path, "w").write("\n=====\n".join(SAMPLES))
out = subprocess.run(["build/tokcheck", gguf_path, path], capture_output=True, text=True, check=True)
if out.stderr:
    print(out.stderr, end="")
ref = Tokenizer.from_file(tok_json)
ok = 0
for s, line in zip(SAMPLES, out.stdout.strip().split("\n")):
    got = [int(x) for x in line.split()] if line.strip() else []
    want = ref.encode(s, add_special_tokens=False).ids
    same = got == want
    ok += same
    print(("MATCH" if same else "DIFF ") + f" {len(want):3d}  {s[:50]!r}")
    if not same:
        print("   C  ", got)
        print("   ref", want)
print(f"{ok}/{len(SAMPLES)} match")
