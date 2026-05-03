"""Deterministic disk cache for LLM responses.

Uses diskcache (SQLite-backed) with SHA-256 keys for reproducible,
high-performance caching of LLM API calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from diskcache import Cache

from feature_forge.exceptions import LLMError


class DiskCache:
    """Disk-backed cache for LLM responses.

    Keys are SHA-256 hashes of normalized request parameters to ensure
    deterministic lookups across process restarts.

    Attributes:
        cache_dir: Directory for SQLite cache files.
        enabled: Whether cache reads/writes are active.
    """

    def __init__(self, cache_dir: str = "memory_files/llm_cache", enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self._cache: Cache | None = None

    def _get_cache(self) -> Cache:
        """Lazy-init the diskcache instance."""
        if self._cache is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = Cache(str(self.cache_dir))
        return self._cache

    def get_key(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Generate deterministic SHA-256 cache key."""
        payload = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            "extra": kwargs,
        }
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve cached response by key.

        Returns None if cache is disabled or key not found.
        """
        if not self.enabled:
            return None
        try:
            return self._get_cache().get(key)
        except Exception as exc:
            raise LLMError(f"Cache read failed for key {key[:16]}...: {exc}") from exc

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store response in cache.

        No-op if cache is disabled.
        """
        if not self.enabled:
            return
        try:
            self._get_cache()[key] = value
        except Exception as exc:
            raise LLMError(f"Cache write failed for key {key[:16]}...: {exc}") from exc

    def close(self) -> None:
        """Close the underlying cache connection."""
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def clear(self) -> None:
        """Clear all cached entries."""
        if self._cache is not None:
            self._cache.clear()

    def __enter__(self) -> DiskCache:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
