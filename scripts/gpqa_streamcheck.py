"""Local transport fixture: final-answer grading ignores the reasoning channel."""

import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["reasoning_effort"] == "low"
        assert request["stream_options"]["include_usage"]
        prompt = request["messages"][-1]["content"]
        answer = re.search(r"^([A-D])\) RIGHT$", prompt, re.M).group(1)
        wrong = "ABCD"[("ABCD".index(answer) + 1) % 4]
        chunks = [
            dict(
                choices=[
                    dict(
                        index=0,
                        delta=dict(reasoning_content=f"Answer: {wrong}"),
                        finish_reason=None,
                    )
                ]
            ),
            dict(
                choices=[
                    dict(
                        index=0,
                        delta=dict(content=f"Answer: {answer}"),
                        finish_reason=None,
                    )
                ]
            ),
            dict(choices=[dict(index=0, delta={}, finish_reason="length")]),
            dict(
                choices=[],
                usage=dict(prompt_tokens=20, completion_tokens=8, total_tokens=28),
            ),
        ]
        payload = "".join(
            "data: "
            + json.dumps(
                dict(
                    id="fixture",
                    object="chat.completion.chunk",
                    created=0,
                    model="fixture",
                    **c,
                )
            )
            + "\n\n"
            for c in chunks
        )
        payload += "data: [DONE]\n\n"
        data = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    rows = [
        dict(
            Question=f"Synthetic fixture {i}",
            **{
                "Correct Answer": "RIGHT",
                "Incorrect Answer 1": "wrong1",
                "Incorrect Answer 2": "wrong2",
                "Incorrect Answer 3": "wrong3",
            },
        )
        for i in range(4)
    ]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="fleet-gpqa-streamcheck-") as directory:
            args = [
                "gpqa.py",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "fixture",
                "--workers",
                "4",
                "--num-examples",
                "4",
                "--max-tokens",
                "8",
                "--stream",
                "--reasoning-effort",
                "low",
                "--out-dir",
                directory,
                "--label",
                "fixture",
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(pd, "read_csv", return_value=pd.DataFrame(rows)),
            ):
                runpy.run_path(str(ROOT / "evals/gpqa.py"), run_name="__main__")
            out = Path(directory)
            result = json.loads((out / "fixture.json").read_text())
            events = list(
                map(json.loads, (out / "fixture.events.jsonl").read_text().splitlines())
            )
            completed = [e for e in events if e["event"] == "request_completed"]
            graded = [e for e in events if e["event"] == "question_graded"]
            assert result["score"] == 1 and result["questions"] == 4
            assert result["completion_tokens"] == 32 and result["truncated"] == 4
            assert len(completed) == len(graded) == 4
            assert all(
                e["content"] != e["reasoning_content"]
                and e["finish_reason"] == "length"
                for e in completed
            )
            assert all(e["score"] == 1 for e in graded)
            assert len({e["question_id"] for e in completed}) == 4
            assert len([e for e in events if e["event"] == "first_output"]) == 4
            assert len([e for e in events if e["event"] == "first_answer_output"]) == 4
            print(
                "PASS: four concurrent streams, separate reasoning/final content, official final-only grading, usage, truncation and question IDs; no dataset/model requests"
            )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
