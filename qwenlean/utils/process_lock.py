"""One model workload per repository; OS locks release automatically on process exit."""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

import psutil


@contextmanager
def model_process_lock(path=None):
    path = Path(path or Path(__file__).resolve().parents[2] / ".qwenlean-model.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    acquired = False
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another QwenLean model workload already holds the process lock"
            ) from exc
        handle.seek(0)
        content = handle.read().strip()
        # This also recognizes a baseline started before lock support was added.
        if content:
            try:
                owner = json.loads(content)
            except json.JSONDecodeError:
                # With the OS lease acquired, a half-written dead-owner file is recoverable.
                owner = {"pid": 99999999, "create_time": 0, "active": False}
            try:
                process = psutil.Process(owner["pid"])
                live = abs(process.create_time() - owner["create_time"]) < 0.01
            except psutil.NoSuchProcess:
                live = False
            if live and owner["pid"] != os.getpid() and owner.get("active", True):
                raise RuntimeError(f"QwenLean model process {owner['pid']} is still active")
        handle.seek(0)
        handle.truncate()
        json.dump(
            {"pid": os.getpid(), "create_time": psutil.Process().create_time(), "active": True},
            handle,
        )
        handle.flush()
        yield
        handle.seek(0)
        handle.truncate()
        json.dump(
            {"pid": os.getpid(), "create_time": psutil.Process().create_time(), "active": False},
            handle,
        )
        handle.flush()
    finally:
        if acquired:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        # Never unlink a lock inode: other waiting/open handles must refer to the same file.
