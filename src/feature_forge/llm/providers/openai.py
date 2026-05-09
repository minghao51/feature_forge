"""OpenAI-compatible LLM provider.

Works with any OpenAI-compatible API endpoint, including:
- OpenAI (api.openai.com)
- DeepSeek (api.deepseek.com)
- Local inference servers (vLLM, llama.cpp server, etc.)
"""

from __future__ import annotations

import json
import time
from typing import Any

from openai import AsyncOpenAI

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMClient):
    """OpenAI-compatible async LLM client."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = "https://api.openai.com/v1",
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        if not self.api_key:
            raise LLMError("OpenAI provider requires an API key")
        self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        logger.info(
            "llm_request",
            provider=self.provider_name,
            model=self.model,
            num_messages=len(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        t0 = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            logger.error("llm_error", provider=self.provider_name, model=self.model, error=str(exc))
            raise LLMError(f"OpenAI API error: {exc}") from exc

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            "llm_response",
            provider=self.provider_name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            response_preview=content[:200],
        )

        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=response,
        )

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        enhanced = self._inject_json_schema(messages, schema_description)
        logger.info(
            "llm_request",
            provider=self.provider_name,
            model=self.model,
            num_messages=len(enhanced),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        t0 = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=enhanced,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.error("llm_error", provider=self.provider_name, model=self.model, json_mode=True, error=str(exc))
            raise LLMError(f"OpenAI JSON mode error: {exc}") from exc

        content = response.choices[0].message.content or "{}"
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        usage = response.usage
        logger.info(
            "llm_response",
            provider=self.provider_name,
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency_ms,
            json_mode=True,
            response_preview=content[:200],
        )
        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            logger.error("llm_json_parse_error", provider=self.provider_name, model=self.model, response_preview=content[:200])
            raise LLMError(f"Invalid JSON response: {exc}") from exc
