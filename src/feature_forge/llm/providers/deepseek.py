"""DeepSeek LLM provider with native JSON mode optimizations.

DeepSeek uses the OpenAI-compatible API with additional optimizations:
- Native `response_format={'type': 'json_object'}` for guaranteed valid JSON
- Schema guidance injected into system prompts for structured output
- Recommended model: deepseek-chat (or deepseek-v4-pro for advanced tasks)
"""

from __future__ import annotations

import json
from typing import Any

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMResponse
from feature_forge.llm.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek async LLM client with native JSON mode optimizations.

    Inherits from OpenAIProvider since DeepSeek's API is fully
    OpenAI-compatible, but adds specialized structured output handling.

    Parameters:
        model: DeepSeek model name. Defaults to 'deepseek-chat'.
        api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
        base_url: DeepSeek API endpoint.
    """

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        import os

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        super().__init__(model=model, api_key=api_key, base_url=base_url)

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """DeepSeek-optimized JSON mode with schema enforcement.

        DeepSeek's JSON mode requires:
        1. `response_format={'type': 'json_object'}`
        2. The word 'json' in the system/user prompt
        3. An example or schema description to guide output structure
        4. Reasonable `max_tokens` to avoid truncated JSON

        Returns:
            Parsed JSON dict matching the described schema.
        """
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
            raise LLMError(f"DeepSeek JSON mode error: {exc}") from exc

        content = response.choices[0].message.content or "{}"
        if not content.strip():
            raise LLMError(
                "DeepSeek returned empty content in JSON mode. "
                "Try rephrasing the prompt or increasing max_tokens."
            )

        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"DeepSeek returned invalid JSON: {content[:200]}... Parse error: {exc}"
            ) from exc

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Standard completion via DeepSeek's OpenAI-compatible API."""
        return await super().complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
