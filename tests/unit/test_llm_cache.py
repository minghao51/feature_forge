"""Unit tests for LLM cache and client wrappers."""

from __future__ import annotations

import pytest

from feature_forge.config import RetryConfig
from feature_forge.exceptions import LLMError
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
        **kwargs,
    ) -> dict[str, str]:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return {"content": resp}

    def _extract_content(self, raw_response: dict[str, str]) -> str:
        return raw_response["content"]

    def _extract_usage(self, raw_response: dict[str, str]) -> tuple[int, int, int]:
        return 10, 5, 15


class TestDiskCache:
    def test_get_key_deterministic(self):
        cache = DiskCache(enabled=False)
        key1 = cache.get_key("openai", "gpt-4", [{"role": "user", "content": "hi"}], 0.2, 100)
        key2 = cache.get_key("openai", "gpt-4", [{"role": "user", "content": "hi"}], 0.2, 100)
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex

    def test_get_key_different_inputs(self):
        cache = DiskCache(enabled=False)
        key1 = cache.get_key("openai", "gpt-4", [{"role": "user", "content": "hi"}], 0.2, 100)
        key2 = cache.get_key("openai", "gpt-4", [{"role": "user", "content": "hello"}], 0.2, 100)
        assert key1 != key2

    def test_cache_disabled_returns_none(self):
        cache = DiskCache(enabled=False)
        assert cache.get("any-key") is None

    def test_cache_roundtrip(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cache = DiskCache(cache_dir=cache_dir, enabled=True)
        key = "test-key"
        value = {"content": "cached", "model": "gpt-4"}
        cache.set(key, value)
        retrieved = cache.get(key)
        assert retrieved == value
        cache.close()

    def test_cache_clear(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cache = DiskCache(cache_dir=cache_dir, enabled=True)
        cache.set("k1", {"v": 1})
        cache.clear()
        assert cache.get("k1") is None
        cache.close()

    def test_cache_context_manager(self, tmp_path):
        cache_dir = str(tmp_path / "cache_ctx")
        with DiskCache(cache_dir=cache_dir, enabled=True) as cache:
            cache.set("k", {"v": 1})
            assert cache.get("k") == {"v": 1}
        assert cache._cache is None

    def test_set_disabled_does_nothing(self):
        cache = DiskCache(enabled=False)
        cache.set("k", {"v": 1})
        assert cache.get("k") is None


class TestLLMClientCaching:
    @pytest.mark.asyncio
    async def test_calls_without_cache(self):
        fake = FakeProvider(responses=["world"])
        resp = await fake.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "world"
        assert fake.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_disabled_no_hit(self):
        fake = FakeProvider(responses=["first", "second"])
        cache = DiskCache(enabled=False)
        fake._cache = cache

        resp1 = await fake.complete([{"role": "user", "content": "hi"}])
        assert resp1.content == "first"
        assert fake.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_api_call(self, tmp_path):
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
    async def test_complete_json_uses_cache(self, tmp_path):
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
    async def test_complete_json_cache_only_stores_valid_json(self, tmp_path):
        fake = FakeProvider(responses=["NOT VALID JSON", '{"result": "ok"}'])
        cache_dir = str(tmp_path / "cache")
        cache = DiskCache(cache_dir=cache_dir, enabled=True)
        fake._cache = cache
        fake.set_retry_config(RetryConfig(max_retries=3, backoff_base=0.01, backoff_max=0.01))

        msg = [{"role": "user", "content": "test"}]
        with pytest.raises(LLMError, match="invalid JSON"):
            await fake.complete_json(msg, schema_description="A JSON object")
        assert fake.call_count == 1

        result = await fake.complete_json(msg, schema_description="A JSON object")
        assert result == {"result": "ok"}
        assert fake.call_count == 2

        result2 = await fake.complete_json(msg, schema_description="A JSON object")
        assert result2 == {"result": "ok"}
        assert fake.call_count == 2
        cache.close()


class TestProviders:
    def test_openai_provider_name(self):
        p = OpenAIProvider(api_key="sk-test")
        assert p.provider_name == "openai"

    def test_deepseek_provider_name(self):
        p = DeepSeekProvider(api_key="sk-test")
        assert p.provider_name == "deepseek"
        assert p.base_url == "https://api.deepseek.com"

    def test_openai_missing_key_raises(self):
        from feature_forge.exceptions import LLMError

        with pytest.raises(LLMError, match="API key"):
            OpenAIProvider(api_key=None)

    def test_llm_response_repr(self):
        resp = LLMResponse(content="hello world", model="gpt-4")
        assert "gpt-4" in repr(resp)
