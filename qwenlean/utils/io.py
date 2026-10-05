"""Canonical hashes and inspectable, durable run storage."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, records):
    with Path(path).open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def create_run(root, label, config):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = Path(root) / f"{stamp}-{label}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "config.json", config)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    write_json(
        path / "run.json",
        {
            "created_utc": stamp,
            "config_hash": digest(config),
            "git_commit": git.stdout.strip(),
            "git_dirty": bool(status.stdout.strip()),
        },
    )
    return path
