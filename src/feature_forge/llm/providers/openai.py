"""OpenAI-compatible LLM provider.

Works with any OpenAI-compatible API endpoint, including:
- OpenAI (api.openai.com)
- DeepSeek (api.deepseek.com)
- Local inference servers (vLLM, llama.cpp server, etc.)
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient, LLMResponse


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
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
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
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=enhanced,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise LLMError(f"OpenAI JSON mode error: {exc}") from exc

        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid JSON response: {exc}") from exc
