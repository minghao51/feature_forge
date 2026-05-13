"""Unit tests for LLM cache and client wrappers."""

from __future__ import annotations

import pytest

from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.llm.cache import DiskCache
from feature_forge.llm.langfuse_wrapper import LangfuseLLMWrapper
from feature_forge.llm.providers.deepseek import DeepSeekProvider
from feature_forge.llm.providers.openai import OpenAIProvider


class FakeProvider(LLMClient):
    """Fake LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="fake-model", api_key="fake-key")
        self.responses = responses or ["hello"]
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    async def _do_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return LLMResponse(
            content=resp,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

    async def _do_complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict:
        import json

        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return json.loads(resp)


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
        # After exit, cache should be closed
        assert cache._cache is None

    def test_set_disabled_does_nothing(self):
        cache = DiskCache(enabled=False)
        cache.set("k", {"v": 1})
        # No error, and no cache should exist
        assert cache.get("k") is None


class TestLangfuseLLMWrapper:
    @pytest.mark.asyncio
    async def test_calls_underlying_client(self):
        fake = FakeProvider(responses=["world"])
        wrapper = LangfuseLLMWrapper(fake)
        resp = await wrapper.complete([{"role": "user", "content": "hi"}])
        assert resp.content == "world"
        assert fake.call_count == 1

    @pytest.mark.asyncio
    async def test_caches_response(self):
        fake = FakeProvider(responses=["first", "second"])
        cache = DiskCache(enabled=False)
        wrapper = LangfuseLLMWrapper(fake, cache=cache)

        # Without cache, each call goes to provider
        resp1 = await wrapper.complete([{"role": "user", "content": "hi"}])
        assert resp1.content == "first"
        assert fake.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_api_call(self, tmp_path):
        fake = FakeProvider(responses=["expensive"])
        cache_dir = str(tmp_path / "cache")
        cache = DiskCache(cache_dir=cache_dir, enabled=True)
        wrapper = LangfuseLLMWrapper(fake, cache=cache)

        msg = [{"role": "user", "content": "test"}]
        resp1 = await wrapper.complete(msg)
        assert resp1.content == "expensive"
        assert fake.call_count == 1

        resp2 = await wrapper.complete(msg)
        assert resp2.content == "expensive"
        # Should still be 1 because cache hit
        assert fake.call_count == 1
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
