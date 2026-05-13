"""Tests for LLMClient base class edge cases."""

from __future__ import annotations

import pytest

from feature_forge.config import RetryConfig
from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient, LLMResponse


class SimpleProvider(LLMClient):
    """Minimal LLM provider for testing base class methods."""

    def __init__(self) -> None:
        super().__init__(model="test-model", api_key="test-key")

    @property
    def provider_name(self) -> str:
        return "test"

    async def _call_api(self, messages, temperature, max_tokens, **kwargs):
        return {"content": "response", "usage": {}}

    def _extract_content(self, raw_response):
        return raw_response.get("content", "")

    def _extract_usage(self, raw_response):
        return 10, 5, 15

    def _json_mode_kwargs(self):
        return {"response_format": {"type": "json_object"}}


class TestLLMClientBase:
    """Cover LLMClient base class uncovered paths."""

    @pytest.mark.asyncio
    async def test_complete_json_with_json_mode(self):
        class JsonModeProvider(SimpleProvider):
            async def _do_complete(
                self, messages, temperature=0.2, max_tokens=4096, json_mode=False, **kwargs
            ):
                return LLMResponse(content='{"result": "ok"}', model="test")

        p = JsonModeProvider()
        result = await p.complete_json(
            [{"role": "user", "content": "test"}],
            schema_description="A JSON object with 'result' key",
        )
        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_do_complete_exception_wrapping(self):
        class FailingProvider(SimpleProvider):
            async def _call_api(self, messages, temperature, max_tokens, **kwargs):
                msg = "API connection failed"
                raise ConnectionError(msg)

        p = FailingProvider()
        with pytest.raises(LLMError, match="API error"):
            await p._do_complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_do_complete_llm_error_passthrough(self):
        class LLLErrorProvider(SimpleProvider):
            async def _call_api(self, messages, temperature, max_tokens, **kwargs):
                raise LLMError("rate limited")

        p = LLLErrorProvider()
        with pytest.raises(LLMError, match="rate limited"):
            await p._do_complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_retry_with_config(self):
        class RetryProvider(SimpleProvider):
            def __init__(self):
                super().__init__()
                self.count = 0

            async def _do_complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
                self.count += 1
                return LLMResponse(content="ok", model="test")

        p = RetryProvider()
        p.set_retry_config(RetryConfig(max_retries=1, backoff_base=0.01))
        result = await p.complete([{"role": "user", "content": "hi"}])
        assert result.content == "ok"

    def test_inject_json_schema_no_system(self):
        messages = [{"role": "user", "content": "hello"}]
        enhanced = LLMClient._inject_json_schema(messages, '{"type": "object"}')
        assert enhanced[0]["role"] == "system"
        assert "MUST respond with valid JSON" in enhanced[0]["content"]
        assert enhanced[1]["role"] == "user"

    def test_inject_json_schema_with_system(self):
        messages = [
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": "hello"},
        ]
        enhanced = LLMClient._inject_json_schema(messages, '{"type": "object"}')
        assert enhanced[0]["role"] == "system"
        assert "You are a helpful AI." in enhanced[0]["content"]
        assert "MUST respond with valid JSON" in enhanced[0]["content"]

    def test_parse_json_response_valid(self):
        provider = SimpleProvider()
        result = provider._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_response_empty(self):
        provider = SimpleProvider()
        with pytest.raises(LLMError, match="empty content"):
            provider._parse_json_response("")

    def test_parse_json_response_invalid(self):
        provider = SimpleProvider()
        with pytest.raises(LLMError, match="invalid JSON"):
            provider._parse_json_response("{broken")

    def test_parse_json_response_markdown_fenced(self):
        provider = SimpleProvider()
        with pytest.raises(LLMError, match="invalid JSON"):
            provider._parse_json_response('```json\n{"key": "value"}\n```')

    def test_build_cache_key_deterministic(self):
        provider = SimpleProvider()
        key1 = provider.build_cache_key(
            [{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=100
        )
        key2 = provider.build_cache_key(
            [{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=100
        )
        assert key1 == key2
        assert len(key1) == 64

    def test_build_cache_key_different(self):
        provider = SimpleProvider()
        key1 = provider.build_cache_key(
            [{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=100
        )
        key2 = provider.build_cache_key(
            [{"role": "user", "content": "bye"}], temperature=0.2, max_tokens=100
        )
        assert key1 != key2

    def test_api_key_secret_property(self):
        from pydantic import SecretStr

        provider = SimpleProvider()
        assert provider.api_key_secret is not None
        assert isinstance(provider.api_key_secret, SecretStr)

    @pytest.mark.asyncio
    async def test_retry_no_config(self):
        """_retry calls fn directly when no config set."""
        provider = SimpleProvider()

        async def dummy():
            return "direct"

        result = await provider._retry(dummy)
        assert result == "direct"
