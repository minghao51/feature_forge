"""Deterministic disk cache for LLM responses.

Uses diskcache (SQLite-backed) with SHA-256 keys for reproducible,
high-performance caching of LLM API calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from feature_forge.exceptions import LLMError
from feature_forge.observability.structlog_config import get_logger

if TYPE_CHECKING:
    from diskcache import Cache

logger = get_logger(__name__)


def compute_cache_key(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> str:
    payload: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
        "extra": kwargs,
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class DiskCache:
    """Disk-backed cache for LLM responses.

    Keys are SHA-256 hashes of normalized request parameters to ensure
    deterministic lookups across process restarts.

    Attributes:
        cache_dir: Directory for SQLite cache files.
        enabled: Whether cache reads/writes are active.
        ttl_seconds: Optional per-entry time-to-live; entries older than
            this are ignored on read and removed by :meth:`maintain`.
        size_limit_bytes: Optional total-size cap; enforced eagerly by
            diskcache on writes (least-recently-stored eviction) and by
            :meth:`maintain` when the cap is tightened after the fact.
    """

    def __init__(
        self,
        cache_dir: str = "memory_files/llm_cache",
        enabled: bool = True,
        ttl_seconds: float | None = None,
        size_limit_bytes: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.size_limit_bytes = size_limit_bytes
        self._cache: Cache | None = None

    def _get_cache(self) -> Cache:
        """Lazy-init the diskcache instance."""
        if self._cache is None:
            try:
                from diskcache import Cache as DiskCache
            except ImportError as exc:
                raise LLMError("diskcache not installed. Run: uv pip install diskcache") from exc
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {}
            if self.size_limit_bytes is not None:
                kwargs["size_limit"] = int(self.size_limit_bytes)
            self._cache = DiskCache(str(self.cache_dir), **kwargs)
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
        return compute_cache_key(provider, model, messages, temperature, max_tokens, **kwargs)

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            result = self._get_cache().get(key)
            logger.debug("cache_get", key=key[:16], hit=result is not None)
            return result  # type: ignore[no-any-return]
        except Exception as exc:
            raise LLMError(f"Cache read failed for key {key[:16]}...: {exc}") from exc

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._get_cache().set(key, value, expire=self.ttl_seconds)
            logger.debug("cache_set", key=key[:16], ttl_seconds=self.ttl_seconds)
        except Exception as exc:
            raise LLMError(f"Cache write failed for key {key[:16]}...: {exc}") from exc

    def stats(self) -> dict[str, int]:
        """Return entry count and on-disk volume (bytes) without mutating."""
        cache = self._get_cache()
        return {"items": len(cache), "volume_bytes": cache.volume()}

    def maintain(self) -> dict[str, int]:
        """Garbage-collect: drop expired entries, then cull to the size cap.

        diskcache 5.x's ``expire``/``cull`` return removed-item counts; byte
        figures are derived from the volume delta. Safe to call repeatedly;
        a no-op on an empty or healthy cache.
        """
        if not self.enabled:
            return {
                "expired_removed": 0,
                "expired_bytes": 0,
                "culled_removed": 0,
                "culled_bytes": 0,
                "items": 0,
                "volume_bytes": 0,
            }
        cache = self._get_cache()
        volume_before = cache.volume()
        expired_removed = int(cache.expire())
        expired_bytes = volume_before - cache.volume()
        volume_before = cache.volume()
        culled_removed = int(cache.cull())
        culled_bytes = volume_before - cache.volume()
        return {
            "expired_removed": expired_removed,
            "expired_bytes": int(expired_bytes),
            "culled_removed": culled_removed,
            "culled_bytes": int(culled_bytes),
            **self.stats(),
        }

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
