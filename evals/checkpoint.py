"""Small, durable CSV checkpoint for an evaluation; no database or service."""

import csv
from dataclasses import asdict
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

FINAL = {"correct", "wrong", "capped"}
FIELDS = [
    "index",
    "id",
    "status",
    "score",
    "attempts",
    "started",
    "updated",
    "seconds",
    "prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "answer_characters",
    "reasoning_characters",
    "error",
]


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2) + "\n")


class Checkpoint:
    def __init__(self, directory, config, examples, workers):
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.lockfile = (self.path / "runner.lock").open("a")
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mu = threading.RLock()
        self.workers = workers
        self.rows = []
        self.by_id = {}
        self.started = time.monotonic()
        self.closed = False
        self.status = "ready"
        self.examples = examples
        self.ids = [
            f"{i + 1:03d}-"
            + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:12]
            for i, row in enumerate(examples)
        ]
        config_file = self.path / "config.json"
        if config_file.exists():
            if json.loads(config_file.read_text()) != config:
                raise ValueError(
                    "Checkpoint settings differ; use the original settings or a new run directory"
                )
        else:
            if (self.path / "questions.csv").exists():
                raise ValueError(
                    "CSV exists without its configuration; refusing to overwrite"
                )
            atomic_json(config_file, config)
        frozen = self.path / "examples.json"
        if frozen.exists() and json.loads(frozen.read_text()) != examples:
            raise ValueError(
                "Question selection/permutations differ from the checkpoint"
            )
        atomic_json(frozen, examples)
        previous = self.path / "state.json"
        self.previous_seconds = (
            json.loads(previous.read_text()).get("elapsed_seconds", 0)
            if previous.exists()
            else 0
        )
        if (self.path / "questions.csv").exists():
            with (self.path / "questions.csv").open(newline="") as f:
                self.rows = list(csv.DictReader(f))
            if [r["id"] for r in self.rows] != self.ids:
                raise ValueError(
                    "Checkpoint CSV does not match frozen question identities"
                )
        else:
            self.rows = [
                dict.fromkeys(FIELDS, "")
                | dict(index=i + 1, id=qid, status="queued", attempts=0)
                for i, qid in enumerate(self.ids)
            ]
        self.by_id = {r["id"]: r for r in self.rows}
        (self.path / "responses").mkdir(exist_ok=True)
        for row in self.rows:
            saved = self.result(row["id"])
            if saved:
                self._grade_row(row, saved)
            elif row["status"] not in FINAL:
                row["status"] = "queued"
        if not (self.path / "control").exists():
            atomic_text(self.path / "control", "run\n")
        self._csv()
        self.heartbeat("ready")

    def _csv(self):
        data = io.StringIO(newline="")
        writer = csv.DictWriter(data, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(self.rows)
        atomic_text(self.path / "questions.csv", data.getvalue())

    def control(self):
        try:
            return (self.path / "control").read_text().strip()
        except FileNotFoundError:
            return "run"

    def heartbeat(self, status=None):
        with self.mu:
            if status:
                self.status = status
            elapsed = self.previous_seconds + time.monotonic() - self.started
            atomic_json(
                self.path / "state.json",
                dict(
                    status=self.status,
                    pid=os.getpid(),
                    updated=time.time(),
                    workers=self.workers,
                    total=len(self.rows),
                    elapsed_seconds=elapsed,
                ),
            )

    def update(self, qid, **fields):
        with self.mu:
            row = self.by_id[qid]
            row.update(fields, updated=time.time())
            self._csv()

    def start(self, qid):
        row = self.by_id[qid]
        previous = self.path / "responses" / (qid + ".json")
        if previous.exists():
            os.replace(
                previous,
                previous.with_name(f"{qid}.attempt-{row['attempts'] or 0}.json"),
            )
        self.update(
            qid,
            status="waiting",
            attempts=int(row["attempts"] or 0) + 1,
            started=time.time(),
            error="",
            answer_characters=0,
            reasoning_characters=0,
        )

    def response(self, qid):
        path = self.path / "responses" / (qid + ".json")
        return json.loads(path.read_text()) if path.exists() else None

    def save_response(self, qid, record, complete=False):
        with self.mu:
            atomic_json(
                self.path / "responses" / (qid + ".json"),
                record | {"complete": complete},
            )
            usage = record.get("usage") or {}
            self.update(
                qid,
                status="grading"
                if complete
                else ("answering" if record["content"] else "reasoning"),
                seconds=record["seconds"],
                answer_characters=len(record["content"]),
                reasoning_characters=len(record["reasoning_content"]),
                prompt_tokens=usage.get("prompt_tokens", ""),
                completion_tokens=usage.get("completion_tokens", ""),
                finish_reason=record.get("finish_reason") or "",
            )

    def result(self, qid):
        path = self.path / "responses" / (qid + ".result.json")
        return json.loads(path.read_text()) if path.exists() else None

    def _grade_row(self, row, saved):
        result, response = saved["result"], saved["response"]
        usage = response.get("usage") or {}
        row.update(
            status="capped"
            if response.get("finish_reason") == "length"
            else ("correct" if result["score"] == 1 else "wrong"),
            score=result["score"],
            seconds=response["seconds"],
            updated=time.time(),
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            finish_reason=response.get("finish_reason") or "",
            answer_characters=len(response["content"]),
            reasoning_characters=len(response["reasoning_content"]),
        )

    def complete(self, qid, result):
        with self.mu:
            response = self.response(qid)
            saved = dict(result=asdict(result), response=response)
            # Save the graded result first. Recovery repairs a stale CSV if the
            # process dies between this durable write and the CSV replacement.
            atomic_json(self.path / "responses" / (qid + ".result.json"), saved)
            self._grade_row(self.by_id[qid], saved)
            self._csv()

    def interrupt(self):
        with self.mu:
            for row in self.rows:
                if row["status"] not in FINAL | {"queued", "error"}:
                    row["status"] = "interrupted"
                    row["updated"] = time.time()
            self._csv()
            self.heartbeat("stopped")

    def close(self, status):
        if not self.closed:
            self.heartbeat(status)
            fcntl.flock(self.lockfile, fcntl.LOCK_UN)
            self.lockfile.close()
            self.closed = True
