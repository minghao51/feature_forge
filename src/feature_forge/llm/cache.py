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
            try:
                from diskcache import Cache as DiskCache
            except ImportError as exc:
                raise LLMError("diskcache not installed. Run: uv pip install diskcache") from exc
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = DiskCache(str(self.cache_dir))
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
            self._get_cache()[key] = value
            logger.debug("cache_set", key=key[:16])
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
