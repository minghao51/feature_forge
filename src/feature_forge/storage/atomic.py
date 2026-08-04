"""Atomic filesystem operations used by durable artifact stores."""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def create_staging_directory(parent: Path, *, prefix: str = ".tmp-") -> Path:
    """Create a unique staging directory beside its eventual destination."""
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON through a same-directory temporary file and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_publish_directory(staging: Path, destination: Path) -> None:
    """Publish a staging directory without replacing an existing namespace."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(f".{destination.name}.commit-lock")
    token = uuid.uuid4().hex
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"artifact namespace is being committed: {destination}") from exc

    atomic_write_json(
        lock / "owner.json",
        {
            "schema_version": "1",
            "token": token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    try:
        if destination.exists():
            raise FileExistsError(f"artifact namespace already exists: {destination}")
        os.rename(staging, destination)
        fsync_directory(destination.parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        _remove_owned_lock(lock, token)


def write_success_marker(directory: Path, manifest_sha256: str) -> Path:
    """Create the completion marker anchored to the committed manifest digest."""
    marker = directory / "_SUCCESS"
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(f"{manifest_sha256}\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(directory)
    return marker


def fsync_directory(path: Path) -> None:
    """Best-effort fsync for directory entry durability on local filesystems."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _remove_owned_lock(lock: Path, token: str) -> None:
    """Remove a lock only when its ownership token still matches."""
    owner = lock / "owner.json"
    try:
        payload = json.loads(owner.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if payload.get("token") != token:
        return
    owner.unlink(missing_ok=True)
    try:
        lock.rmdir()
    except FileNotFoundError:
        pass
