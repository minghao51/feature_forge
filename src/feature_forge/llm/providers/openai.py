"""OpenAI-compatible LLM provider.

Works with any OpenAI-compatible API endpoint, including:
- OpenAI (api.openai.com)
- DeepSeek (api.deepseek.com)
- Local inference servers (vLLM, llama.cpp server, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient

if TYPE_CHECKING:
    from feature_forge.llm.cache import DiskCache


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
        cache: DiskCache | None = None,
        tracing_enabled: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            cache=cache,
            tracing_enabled=tracing_enabled,
        )
        if not self.get_api_key():
            raise LLMError("OpenAI API key is required but was not provided.")
        self._client = AsyncOpenAI(api_key=self.get_api_key(), base_url=self.base_url)

    def _json_mode_kwargs(self) -> dict[str, Any]:
        return {"response_format": {"type": "json_object"}}

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Any:
        return await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _extract_content(self, raw_response: Any) -> str:
        return raw_response.choices[0].message.content or ""

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        usage = raw_response.usage
        if usage is None:
            return 0, 0, 0
        return usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
