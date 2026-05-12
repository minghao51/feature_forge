"""Langfuse auto-tracing wrapper for LLMClient.

Wraps any LLMClient implementation to automatically trace every
completion call with input/output, token usage, and latency.
"""

from __future__ import annotations

from typing import Any

from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.llm.cache import DiskCache
from feature_forge.observability.langfuse_tracer import trace_generation
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class LangfuseLLMWrapper(LLMClient):
    """Wrapper that adds Langfuse tracing + caching to any LLMClient.

    Usage:
        base = OpenAIProvider(api_key="sk-...")
        client = LangfuseLLMWrapper(base, cache=DiskCache())
        response = await client.complete(messages=[...])
    """

    def __init__(
        self,
        client: LLMClient,
        cache: DiskCache | None = None,
    ) -> None:
        self._client = client
        self._cache = cache
        super().__init__(
            model=client.model,
            api_key=client.api_key,
            base_url=client.base_url,
        )

    @property
    def provider_name(self) -> str:
        return self._client.provider_name

    def _extract_content(self, raw_response: Any) -> str:
        return self._client._extract_content(raw_response)

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        return self._client._extract_usage(raw_response)

    def _json_mode_kwargs(self) -> dict[str, Any]:
        return self._client._json_mode_kwargs()

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Any:
        return await self._client._call_api(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete with cache check, Langfuse tracing, and retry on inner client."""
        cache_key = self.build_cache_key(messages, temperature, max_tokens, **kwargs)

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info(
                    "llm_cache_hit",
                    provider=self.provider_name,
                    model=self.model,
                    cache_key=cache_key[:16],
                )
                return LLMResponse(
                    content=cached["content"],
                    model=cached.get("model", self.model),
                    prompt_tokens=cached.get("prompt_tokens", 0),
                    completion_tokens=cached.get("completion_tokens", 0),
                    total_tokens=cached.get("total_tokens", 0),
                )

        logger.info(
            "llm_cache_miss",
            provider=self.provider_name,
            model=self.model,
            cache_key=cache_key[:16],
        )

        @trace_generation(name=f"{self.provider_name}-completion")
        async def _traced_complete() -> LLMResponse:
            return await self._client.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        response = await _traced_complete()

        if self._cache is not None:
            self._cache.set(
                cache_key,
                {
                    "content": response.content,
                    "model": response.model,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                },
            )

        return response  # type: ignore[no-any-return]

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        return await self._client.complete_json(
            messages=messages,
            schema_description=schema_description,
            temperature=temperature,
            max_tokens=max_tokens,
        )
