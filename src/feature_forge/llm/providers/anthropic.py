"""Anthropic Claude LLM provider."""

from __future__ import annotations

import json
from typing import Any

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None  # type: ignore[misc,assignment]

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient, LLMResponse


class AnthropicProvider(LLMClient):
    """Anthropic Claude async LLM client."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        if AsyncAnthropic is None:
            raise LLMError("Anthropic SDK not installed. Run: uv pip install anthropic")
        if not self.api_key:
            raise LLMError("Anthropic provider requires an API key")
        self._client = AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        system_msg = ""
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            else:
                conversation.append(msg)

        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg or "You are a helpful assistant.",
                messages=conversation,  # type: ignore[arg-type]
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        content = ""
        if response.content:
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    content += block.text

        usage = response.usage
        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=((usage.input_tokens + usage.output_tokens) if usage else 0),
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
        system_msg = ""
        conversation: list[dict[str, str]] = []
        for msg in enhanced:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            else:
                conversation.append(msg)

        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg or "You are a helpful assistant.",
                messages=conversation,  # type: ignore[arg-type]
            )
        except Exception as exc:
            raise LLMError(f"Anthropic JSON mode error: {exc}") from exc

        content = ""
        if response.content:
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    content += block.text

        if not content.strip():
            raise LLMError("Anthropic returned empty content in JSON mode.")

        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Anthropic returned invalid JSON: {content[:200]}... Parse error: {exc}"
            ) from exc
