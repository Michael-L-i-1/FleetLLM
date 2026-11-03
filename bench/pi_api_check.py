"""Verify the optional Qwen launcher and a direct-answer chat API request."""

import json
from pathlib import Path
import subprocess
import time
import urllib.request

h = Path.home()
out = h / "pi-opt-api-check"
out.mkdir(exist_ok=True)
command = [
    str(h / "run-demo"),
    "qwen",
    "Return JSON only with one key ready set to true.",
    "--n",
    "32",
]
with (
    (out / "launcher.txt").open("w") as stdout,
    (out / "launcher.log").open("w") as stderr,
):
    run = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=180)
assert run.returncode == 0
raw = (out / "launcher.txt").read_text().split("<|im_end|>")[0].strip()
assert json.loads(raw) == {"ready": True}
cmd = [str(h / "serve-demo"), "qwen", "--port", "18095"]

with (out / "server.log").open("w") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 90
        while True:
            if proc.poll() is not None:
                raise RuntimeError("server exited")
            try:
                with urllib.request.urlopen("http://127.0.0.1:18095/health", timeout=1):
                    break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError("startup timeout")
                time.sleep(0.2)
        body = dict(
            model="pi-demo",
            messages=[
                dict(
                    role="user",
                    content="Return JSON only with one key ready set to true.",
                )
            ],
            reasoning_effort="none",
            temperature=0,
            max_tokens=32,
            stream=False,
        )
        request = urllib.request.Request(
            "http://127.0.0.1:18095/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as r:
            response = json.load(r)
        assert json.loads(response["choices"][0]["message"]["content"]) == {
            "ready": True
        }
        with urllib.request.urlopen(request, timeout=180) as r:
            repeated = json.load(r)
        assert json.loads(repeated["choices"][0]["message"]["content"]) == {
            "ready": True
        }
        assert repeated["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        (out / "result.json").write_text(
            json.dumps(
                dict(
                    launcher=command,
                    server=cmd,
                    request=body,
                    response=response,
                    repeated_response=repeated,
                ),
                indent=2,
            )
            + "\n"
        )
        print(
            "Launcher, direct-answer JSON API and repeated-prefix cache checks PASS",
            flush=True,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
