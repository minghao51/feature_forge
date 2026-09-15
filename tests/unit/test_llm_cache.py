"""Unit tests for LLM cache and client wrappers."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from feature_forge.config import RetryConfig
from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.llm.cache import DiskCache
from feature_forge.llm.providers.deepseek import DeepSeekProvider
from feature_forge.llm.providers.openai import OpenAIProvider


class FakeProvider(LLMClient):
    """Fake LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="fake-model", api_key="fake-key", tracing_enabled=False)
        self.responses = responses or ["hello"]
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, str]:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return {"content": resp}

    def _extract_content(self, raw_response: dict[str, str]) -> str:
        return raw_response["content"]

    def _extract_usage(self, raw_response: dict[str, str]) -> tuple[int, int, int]:
        return 10, 5, 15


class TestDiskCache:
    def test_get_key_deterministic(self) -> None:
        cache = DiskCache(enabled=False)
        key1 = cache.get_key("openai", "gpt-4", [{"role": "user", "content": "hi"}], 0.2, 100)
        key2 = cache.get_key("openai", "gpt-4", [{"role": "user", "content": "hi"}], 0.2, 100)
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex

    def test_get_key_different_inputs(self) -> None:
        cache = DiskCache(enabled=False)
        key1 = cache.get_key("openai", "gpt-4", [{"role": "user", "content": "hi"}], 0.2, 100)
        key2 = cache.get_key("openai", "gpt-4", [{"role": "user", "content": "hello"}], 0.2, 100)
        assert key1 != key2

    def test_cache_disabled_returns_none(self) -> None:
        cache = DiskCache(enabled=False)
        assert cache.get("any-key") is None

    def test_cache_roundtrip(self, tmp_path: pathlib.Path) -> None:
        cache_dir = str(tmp_path / "cache")
        cache = DiskCache(cache_dir=cache_dir, enabled=True)
        key = "test-key"
        value = {"content": "cached", "model": "gpt-4"}
        cache.set(key, value)
        retrieved = cache.get(key)
        assert retrieved == value
        cache.close()

    def test_cache_clear(self, tmp_path: pathlib.Path) -> None:
        cache_dir = str(tmp_path / "cache")
        cache = DiskCache(cache_dir=cache_dir, enabled=True)
        cache.set("k1", {"v": 1})
        cache.clear()
        assert cache.get("k1") is None
        cache.close()

    def test_cache_context_manager(self, tmp_path: pathlib.Path) -> None:
        cache_dir = str(tmp_path / "cache_ctx")
        with DiskCache(cache_dir=cache_dir, enabled=True) as cache:
            cache.set("k", {"v": 1})
            assert cache.get("k") == {"v": 1}
        assert cache._cache is None

    def test_set_disabled_does_nothing(self) -> None:
        cache = DiskCache(enabled=False)
        cache.set("k", {"v": 1})
        assert cache.get("k") is None


class TestLLMClientCaching:
    @pytest.mark.asyncio
    async def test_calls_without_cache(self) -> None:
        fake = FakeProvider(responses=["world"])
        resp = await fake.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "world"
        assert fake.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_disabled_no_hit(self) -> None:
        fake = FakeProvider(responses=["first", "second"])
        cache = DiskCache(enabled=False)
        fake._cache = cache

        resp1 = await fake.complete([{"role": "user", "content": "hi"}])
        assert resp1.content == "first"
        assert fake.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_api_call(self, tmp_path: pathlib.Path) -> None:
        fake = FakeProvider(responses=["expensive"])
        cache_dir = str(tmp_path / "cache")
        cache = DiskCache(cache_dir=cache_dir, enabled=True)
        fake._cache = cache

        msg = [{"role": "user", "content": "test"}]
        resp1 = await fake.complete(msg)
        assert resp1.content == "expensive"
        assert fake.call_count == 1

        resp2 = await fake.complete(msg)
        assert resp2.content == "expensive"
        assert fake.call_count == 1
        cache.close()

    @pytest.mark.asyncio
    async def test_prompt_meta_recorded_in_cache_payload(self, tmp_path: pathlib.Path) -> None:
        fake = FakeProvider(responses=["answer"])
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        fake._cache = cache

        meta = {"prompt_name": "demo", "prompt_version": 2, "prompt_sha256": "abc"}
        msg = [{"role": "user", "content": "test"}]
        await fake.complete(msg, prompt_meta=meta)

        key = fake.build_cache_key(msg, 0.2, 4096)
        payload = cache.get(key)
        assert payload is not None
        assert payload["prompt_meta"] == meta
        cache.close()

    @pytest.mark.asyncio
    async def test_prompt_meta_not_part_of_cache_key(self, tmp_path: pathlib.Path) -> None:
        fake = FakeProvider(responses=["answer"])
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        fake._cache = cache

        msg = [{"role": "user", "content": "test"}]
        meta_a = {"prompt_name": "demo", "prompt_version": 1, "prompt_sha256": "a"}
        meta_b = {"prompt_name": "demo", "prompt_version": 2, "prompt_sha256": "b"}
        r1 = await fake.complete(msg, prompt_meta=meta_a)
        r2 = await fake.complete(msg, prompt_meta=meta_b)  # same text → hit
        assert r1.content == r2.content == "answer"
        assert fake.call_count == 1
        cache.close()

    @pytest.mark.asyncio
    async def test_complete_json_uses_cache(self, tmp_path: pathlib.Path) -> None:
        fake = FakeProvider(responses=['{"result": "ok"}'])
        cache_dir = str(tmp_path / "cache")
        cache = DiskCache(cache_dir=cache_dir, enabled=True)
        fake._cache = cache

        msg = [{"role": "user", "content": "test"}]
        result1 = await fake.complete_json(msg, schema_description="A JSON object")
        assert result1 == {"result": "ok"}
        assert fake.call_count == 1

        result2 = await fake.complete_json(msg, schema_description="A JSON object")
        assert result2 == {"result": "ok"}
        assert fake.call_count == 1
        cache.close()

    @pytest.mark.asyncio
    async def test_complete_json_cache_only_stores_valid_json(self, tmp_path: pathlib.Path) -> None:
        fake = FakeProvider(responses=["NOT VALID JSON", '{"result": "ok"}'])
        cache_dir = str(tmp_path / "cache")
        cache = DiskCache(cache_dir=cache_dir, enabled=True)
        fake._cache = cache
        fake.set_retry_config(RetryConfig(max_retries=3, backoff_base=0.01, backoff_max=0.01))

        msg = [{"role": "user", "content": "test"}]
        result = await fake.complete_json(msg, schema_description="A JSON object")
        assert result == {"result": "ok"}
        assert fake.call_count == 2

        result2 = await fake.complete_json(msg, schema_description="A JSON object")
        assert result2 == {"result": "ok"}
        assert fake.call_count == 2
        cache.close()


class TestProviders:
    def test_openai_provider_name(self) -> None:
        p = OpenAIProvider(api_key="sk-test")
        assert p.provider_name == "openai"

    def test_deepseek_provider_name(self) -> None:
        p = DeepSeekProvider(api_key="sk-test")
        assert p.provider_name == "deepseek"
        assert p.base_url == "https://api.deepseek.com"

    def test_openai_missing_key_raises(self) -> None:
        from feature_forge.exceptions import LLMError

        with pytest.raises(LLMError, match="API key"):
            OpenAIProvider(api_key=None)

    def test_llm_response_repr(self) -> None:
        resp = LLMResponse(content="hello world", model="gpt-4")
        assert "gpt-4" in repr(resp)


class TestCacheGC:
    """TTL expiry, size-cap culling, and maintain() reporting."""

    def _write_entries(self, cache: DiskCache, n: int, payload_size: int = 4096) -> list[str]:
        keys = [f"key-{i}" for i in range(n)]
        for i, key in enumerate(keys):
            cache.set(key, {"content": "x" * payload_size, "i": i})
        return keys

    def test_ttl_expired_entries_not_returned(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Note: diskcache silently drops writes whose expiry is already in the
        # past, so a negative TTL cannot be used here; short TTL + sleep is
        # deterministic (50ms TTL, 150ms wait).
        import time

        cache = DiskCache(cache_dir="gc-ttl", ttl_seconds=0.05)
        try:
            self._write_entries(cache, 2, payload_size=64)
            assert cache.get("key-0") == {"content": "x" * 64, "i": 0}
            time.sleep(0.15)
            assert cache.get("key-0") is None
            assert cache.get("key-1") is None
        finally:
            cache.close()

    def test_ttl_none_keeps_entries(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cache = DiskCache(cache_dir="gc-keep")
        try:
            cache.set("k", {"content": "v"})
            assert cache.get("k") == {"content": "v"}
        finally:
            cache.close()

    def test_maintain_expires_and_reports(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time

        monkeypatch.chdir(tmp_path)
        cache = DiskCache(cache_dir="gc-expire", ttl_seconds=0.05)
        try:
            self._write_entries(cache, 3, payload_size=64)
            time.sleep(0.15)
            stats = cache.maintain()
            assert stats["expired_removed"] == 3
            assert stats["items"] == 0
        finally:
            cache.close()

    def test_maintain_culls_to_tightened_size_limit(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # diskcache enforces size_limit eagerly on every write (evicting
        # least-recently-stored items), so the cull() path in maintain() only
        # has work to do when the cap is tightened after the fact — which is
        # exactly the operational scenario for cache_gc.py.
        writer = DiskCache(cache_dir="gc-cull")
        try:
            self._write_entries(writer, 4, payload_size=20 * 1024)
            assert writer.stats()["items"] == 4
        finally:
            writer.close()

        tightened = DiskCache(cache_dir="gc-cull", size_limit_bytes=64 * 1024)
        try:
            stats = tightened.maintain()
            assert stats["culled_removed"] > 0
            assert stats["volume_bytes"] <= 64 * 1024
        finally:
            tightened.close()

    def test_size_limit_enforced_eagerly_on_write(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cache = DiskCache(cache_dir="gc-eager", size_limit_bytes=64 * 1024)
        try:
            # 8 x 20 KiB against a 64 KiB cap: writes succeed but older
            # entries are evicted as the cap is exceeded.
            self._write_entries(cache, 8, payload_size=20 * 1024)
            assert cache.stats()["volume_bytes"] <= 64 * 1024
        finally:
            cache.close()

    def test_maintain_healthy_cache_is_noop(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cache = DiskCache(cache_dir="gc-healthy")
        try:
            cache.set("k", {"content": "v"})
            stats = cache.maintain()
            assert stats["expired_removed"] == 0
            assert stats["culled_removed"] == 0
            assert cache.get("k") == {"content": "v"}
        finally:
            cache.close()

    def test_disabled_cache_maintain_returns_zeros(self) -> None:
        cache = DiskCache(enabled=False)
        stats = cache.maintain()
        assert stats["expired_removed"] == 0
        assert stats["culled_removed"] == 0


class TestCacheGCConfig:
    """Config fields and factory wiring for TTL / size cap."""

    def test_config_defaults_preserve_current_behavior(self) -> None:
        from feature_forge.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.cache_ttl_days is None
        assert cfg.cache_size_limit_mb is None

    def test_config_accepts_positive_values(self) -> None:
        from feature_forge.config import LLMConfig

        cfg = LLMConfig(cache_ttl_days=30, cache_size_limit_mb=512)
        assert cfg.cache_ttl_days == 30
        assert cfg.cache_size_limit_mb == 512

    def test_config_rejects_nonpositive(self) -> None:
        from feature_forge.config import LLMConfig

        with pytest.raises(Exception, match="greater than 0"):
            LLMConfig(cache_ttl_days=0)
        with pytest.raises(Exception, match="greater than 0"):
            LLMConfig(cache_size_limit_mb=-1)

    def test_factory_passes_ttl_and_size_limit(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import SecretStr

        from feature_forge.config import LLMConfig
        from feature_forge.llm.factory import create_llm_client

        monkeypatch.chdir(tmp_path)
        cfg = LLMConfig(
            model="gpt-4o-mini",
            api_key=SecretStr("sk-test"),
            cache_dir="factory-cache",
            cache_ttl_days=7,
            cache_size_limit_mb=256,
        )
        client = create_llm_client(cfg, tracing_enabled=False)
        assert client._cache is not None
        assert client._cache.ttl_seconds == 7 * 86400.0
        assert client._cache.size_limit_bytes == 256 * 1024 * 1024
