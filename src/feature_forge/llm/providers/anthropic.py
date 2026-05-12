"""Anthropic Claude LLM provider."""

from __future__ import annotations

from typing import Any

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment,misc]

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


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

    def _json_mode_kwargs(self) -> dict[str, Any]:
        return {}

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Any:
        system_msg = ""
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            else:
                conversation.append(msg)
        return await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg or "You are a helpful assistant.",
            messages=conversation,  # type: ignore[arg-type]
            **kwargs,
        )

    def _extract_content(self, raw_response: Any) -> str:
        content = ""
        if raw_response.content:
            for block in raw_response.content:
                if getattr(block, "type", None) == "text":
                    content += block.text
        return content

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        usage = raw_response.usage
        if usage is None:
            return 0, 0, 0
        return usage.input_tokens, usage.output_tokens, usage.input_tokens + usage.output_tokens
