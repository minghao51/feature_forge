"""Tests for LLM retry logic with tenacity."""

from __future__ import annotations

import pytest

from feature_forge.config import RetryConfig
from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient, LLMResponse


class FlakyProvider(LLMClient):
    """Provider that fails N times before succeeding."""

    def __init__(self, fail_count: int = 2) -> None:
        super().__init__(model="fake-model", api_key="fake-key")
        self.fail_count = fail_count
        self.attempts = 0

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
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise LLMError(f"Transient failure (attempt {self.attempts})")
        return LLMResponse(
            content="success",
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
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise LLMError(f"Transient failure (attempt {self.attempts})")
        return {"status": "ok"}


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.backoff_base == 1.0
        assert cfg.backoff_max == 30.0
        assert cfg.backoff_exponent == 2.0

    def test_env_override(self):
        import os

        os.environ["FF_RETRY__MAX_RETRIES"] = "5"
        try:
            from feature_forge.config import Settings

            s = Settings()
            assert s.retry.max_retries == 5
        finally:
            del os.environ["FF_RETRY__MAX_RETRIES"]

    def test_invalid_max_retries(self):
        with pytest.raises(ValueError, match="max_retries"):
            RetryConfig(max_retries=-1)


class TestRetryOnLLMError:
    @pytest.mark.asyncio
    async def test_succeeds_after_retries(self):
        provider = FlakyProvider(fail_count=2)
        provider.set_retry_config(RetryConfig(max_retries=3, backoff_base=0.01, backoff_max=0.1))
        response = await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert response.content == "success"
        assert provider.attempts == 3

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        provider = FlakyProvider(fail_count=0)
        provider.set_retry_config(RetryConfig(max_retries=3, backoff_base=0.01, backoff_max=0.1))
        response = await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert response.content == "success"
        assert provider.attempts == 1

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        provider = FlakyProvider(fail_count=10)
        provider.set_retry_config(RetryConfig(max_retries=2, backoff_base=0.01, backoff_max=0.1))
        with pytest.raises(LLMError, match="Transient failure"):
            await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert provider.attempts == 3

    @pytest.mark.asyncio
    async def test_complete_json_retries(self):
        provider = FlakyProvider(fail_count=1)
        provider.set_retry_config(RetryConfig(max_retries=3, backoff_base=0.01, backoff_max=0.1))
        result = await provider.complete_json(
            messages=[{"role": "user", "content": "hi"}],
            schema_description="test",
        )
        assert result == {"status": "ok"}
        assert provider.attempts == 2

    @pytest.mark.asyncio
    async def test_no_retry_without_config(self):
        provider = FlakyProvider(fail_count=1)
        with pytest.raises(LLMError):
            await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert provider.attempts == 1
