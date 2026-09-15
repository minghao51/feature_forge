"""Safe inspection and retention for Hamilton's disposable node cache."""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from hamilton.caching.stores.file import FileResultStore  # type: ignore[import-untyped]
from hamilton.caching.stores.sqlite import (  # type: ignore[import-untyped]
    SQLiteMetadataStore,
)

# Hamilton 1.x uses URL-safe base64 data versions; tests and older caches may
# contain hexadecimal versions. Match only one hash-like filename component.
_DATA_VERSION = re.compile(r"^[A-Za-z0-9_-]{20,}={0,2}(?:\.[A-Za-z0-9_-]+)?$")
_METADATA_FILES = {"metadata_store.db", "metadata_store.db-shm", "metadata_store.db-wal"}


@dataclass(frozen=True)
class HamiltonCacheEntry:
    """Secret-free metadata for one Hamilton cache key."""

    cache_key: str
    data_version: str
    node: str
    layer: str | None
    created_at: str
    serialized_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


class ConcurrentSQLiteMetadataStore(SQLiteMetadataStore):  # type: ignore[misc]
    """Hamilton metadata store configured for bounded multi-process contention."""

    def __init__(self, path: str) -> None:
        super().__init__(path, connection_kwargs={"timeout": 30.0})
        self.connection.execute("PRAGMA busy_timeout=30000")
        for attempt in range(60):
            try:
                self.connection.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 59:
                    raise
                time.sleep(min(0.05 * (attempt + 1), 0.5))

    def __getstate__(self) -> dict[str, Any]:
        """Retain the shared path when Hamilton clones an adapter."""
        return {"path": str(self._directory)}


class AtomicFileResultStore(FileResultStore):  # type: ignore[misc]
    """Hamilton file result store whose publications are atomic."""

    @staticmethod
    def _write_result(file_path: Path, stored_result: Any) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = stored_result.save()
        fd, temporary = tempfile.mkstemp(prefix=f".{file_path.name}.", dir=file_path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, file_path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def hamilton_cache_stores(path: str | Path) -> tuple[Any, Any]:
    """Build shared stores, quarantining unreadable disposable metadata.

    Durable medallion packages remain authoritative. If SQLite reports corrupt
    metadata while opening the acceleration cache, preserve the bad database
    under a ``.corrupt-*`` name and start with empty metadata so Hamilton can
    recompute instead of failing the experiment.
    """
    resolved = Path(path)
    try:
        metadata = ConcurrentSQLiteMetadataStore(str(resolved))
    except sqlite3.DatabaseError:
        _quarantine_corrupt_metadata(resolved)
        metadata = ConcurrentSQLiteMetadataStore(str(resolved))
    return metadata, AtomicFileResultStore(str(resolved))


def _quarantine_corrupt_metadata(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    lock = path / ".metadata-recovery.lock"
    descriptor: int | None = None
    for attempt in range(120):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            if attempt == 119:
                raise sqlite3.DatabaseError(
                    f"timed out waiting to recover Hamilton cache metadata at {path}"
                ) from None
            time.sleep(0.05)
    try:
        database = path / "metadata_store.db"
        if _sqlite_file_is_readable(database):
            # Another process completed recovery while this process waited.
            return
        suffix = f".corrupt-{time.time_ns()}-{os.getpid()}"
        for name in sorted(_METADATA_FILES):
            source = path / name
            if not source.exists():
                continue
            try:
                os.replace(source, path / f"{name}{suffix}")
            except FileNotFoundError:
                continue
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink(missing_ok=True)


def _sqlite_file_is_readable(database: Path) -> bool:
    if not database.is_file():
        return False
    try:
        with closing(sqlite3.connect(database, timeout=1.0)) as connection:
            row = connection.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError:
        return False
    return row is not None and row[0] == "ok"


class HamiltonCacheManager:
    """Inspect and garbage-collect only a resolved Hamilton cache directory.

    This manager never traverses the artifact package directories or the LLM
    response cache. Deletion is limited to rows in Hamilton's metadata database
    and hash-named serialized result files directly below ``path``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.database = self.path / "metadata_store.db"

    def status(self) -> dict[str, Any]:
        """Return bounded cache metadata without creating the cache."""
        entries = self.inspect(limit=None)
        result_paths = self._result_paths()
        timestamps = [entry.created_at for entry in entries]
        return {
            "path": str(self.path),
            "exists": self.path.is_dir(),
            "entries": len(entries),
            "serialized_results": len(result_paths),
            "serialized_bytes": sum(path.stat().st_size for path in result_paths),
            "total_bytes": self._total_bytes(),
            "oldest_entry": min(timestamps) if timestamps else None,
            "newest_entry": max(timestamps) if timestamps else None,
        }

    def inspect(
        self,
        *,
        limit: int | None = 100,
        node: str | None = None,
        layer: str | None = None,
    ) -> list[HamiltonCacheEntry]:
        """Return newest cache entries, optionally filtered by node or layer."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")
        if not self.database.is_file():
            return []
        query = "SELECT cache_key, data_version, node_name, created_at FROM cache_metadata"
        values: list[Any] = []
        if node is not None:
            query += " WHERE node_name = ?"
            values.append(node)
        query += " ORDER BY created_at DESC, cache_key"
        if limit is not None and layer is None:
            query += " LIMIT ?"
            values.append(limit)
        try:
            with closing(self._connect(read_only=True)) as connection:
                rows = connection.execute(query, values).fetchall()
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"Hamilton cache metadata is unreadable: {exc}") from exc

        from feature_forge.dataflows.inventory import node_layer

        entries: list[HamiltonCacheEntry] = []
        for cache_key, data_version, node_name, created_at in rows:
            entry_layer = node_layer(str(node_name))
            if layer is not None and entry_layer != layer:
                continue
            entries.append(
                HamiltonCacheEntry(
                    cache_key=str(cache_key),
                    data_version=str(data_version),
                    node=str(node_name),
                    layer=entry_layer,
                    created_at=_normalize_timestamp(str(created_at)),
                    serialized_bytes=self.serialized_size(str(data_version)),
                )
            )
            if limit is not None and len(entries) >= limit:
                break
        return entries

    def collect(
        self,
        *,
        max_age_days: float | None = None,
        max_size_mb: float | None = None,
        dry_run: bool = False,
        clear: bool = False,
    ) -> dict[str, Any]:
        """Apply age/size retention and report exactly what was selected."""
        if max_age_days is not None and max_age_days <= 0:
            raise ValueError("max_age_days must be > 0")
        if max_size_mb is not None and max_size_mb <= 0:
            raise ValueError("max_size_mb must be > 0")
        if clear and (max_age_days is not None or max_size_mb is not None):
            raise ValueError("clear cannot be combined with age or size limits")
        if not clear and max_age_days is None and max_size_mb is None:
            raise ValueError("provide max_age_days, max_size_mb, or clear=True")

        entries = self.inspect(limit=None)
        selected: set[str] = set()
        now = datetime.now(UTC)
        if clear:
            selected.update(entry.cache_key for entry in entries)
        elif max_age_days is not None:
            cutoff = now - timedelta(days=max_age_days)
            selected.update(
                entry.cache_key for entry in entries if _parse_timestamp(entry.created_at) <= cutoff
            )

        if not clear and max_size_mb is not None:
            maximum = int(max_size_mb * 1024 * 1024)
            remaining = [entry for entry in entries if entry.cache_key not in selected]
            versions: dict[str, list[HamiltonCacheEntry]] = {}
            for entry in remaining:
                versions.setdefault(entry.data_version, []).append(entry)
            current_size = sum(self.serialized_size(version) for version in versions)
            for version, grouped in sorted(
                versions.items(),
                key=lambda item: (min(entry.created_at for entry in item[1]), item[0]),
            ):
                if current_size <= maximum:
                    break
                selected.update(entry.cache_key for entry in grouped)
                current_size -= self.serialized_size(version)

        selected_entries = [entry for entry in entries if entry.cache_key in selected]
        selected_versions = {entry.data_version for entry in selected_entries}
        reclaimed = sum(self.serialized_size(version) for version in selected_versions)
        if clear:
            reclaimed = sum(path.stat().st_size for path in self._result_paths())

        if not dry_run and self.database.is_file():
            self._delete(selected, selected_versions, clear=clear)

        return {
            "path": str(self.path),
            "dry_run": dry_run,
            "cleared": clear,
            "removed_entries": len(selected_entries),
            "reclaimed_bytes": reclaimed,
            "remaining_entries": len(entries) - len(selected_entries),
        }

    def serialized_size(self, data_version: str) -> int:
        """Return bytes occupied by one hash-named result, without loading it."""
        return sum(path.stat().st_size for path in self._paths_for_version(data_version))

    def _delete(self, keys: set[str], versions: set[str], *, clear: bool) -> None:
        with closing(self._connect(read_only=False)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if keys:
                placeholders = ",".join("?" for _ in keys)
                ordered_keys = sorted(keys)
                connection.execute(
                    f"DELETE FROM history WHERE cache_key IN ({placeholders})", ordered_keys
                )
                connection.execute(
                    f"DELETE FROM cache_metadata WHERE cache_key IN ({placeholders})",
                    ordered_keys,
                )
            if clear:
                connection.execute("DELETE FROM history")
                connection.execute("DELETE FROM cache_metadata")
                connection.execute("DELETE FROM run_ids")
            connection.commit()
            remaining_versions = {
                str(row[0])
                for row in connection.execute("SELECT DISTINCT data_version FROM cache_metadata")
            }
        candidates = (
            self._result_paths()
            if clear
            else [
                path
                for version in versions - remaining_versions
                for path in self._paths_for_version(version)
            ]
        )
        for path in candidates:
            path.unlink(missing_ok=True)

    def _connect(self, *, read_only: bool) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30.0)
        connection.execute("PRAGMA busy_timeout=30000")
        if read_only:
            connection.execute("PRAGMA query_only=ON")
        return connection

    def _paths_for_version(self, data_version: str) -> list[Path]:
        if not _DATA_VERSION.fullmatch(data_version):
            return []
        base = data_version.split(".", maxsplit=1)[0]
        if not self.path.is_dir():
            return []
        return sorted(
            path
            for path in self.path.glob(f"{base}*")
            if path.is_file() and _DATA_VERSION.fullmatch(path.name)
        )

    def _result_paths(self) -> list[Path]:
        if not self.path.is_dir():
            return []
        return sorted(
            path
            for path in self.path.iterdir()
            if path.is_file()
            and path.name not in _METADATA_FILES
            and _DATA_VERSION.fullmatch(path.name)
        )

    def _total_bytes(self) -> int:
        if not self.path.is_dir():
            return 0
        return sum(path.stat().st_size for path in self.path.rglob("*") if path.is_file())


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _normalize_timestamp(value: str) -> str:
    return _parse_timestamp(value).isoformat().replace("+00:00", "Z")
