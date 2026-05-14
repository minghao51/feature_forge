"""DeepSeek LLM provider.

DeepSeek uses the OpenAI-compatible API. Inherits from OpenAIProvider
which covers all the similarity; this subclass only overrides the
provider name, default connection details, and thinking mode support.
"""

from __future__ import annotations

from typing import Any

from feature_forge.llm.providers.openai import OpenAIProvider
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek async LLM client.

    Inherits from OpenAIProvider since DeepSeek's API is fully
    OpenAI-compatible. Adds thinking mode support via
    reasoning_effort and extra_body parameters.

    Parameters:
        model: DeepSeek model name. Defaults to 'deepseek-chat'.
        api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
        base_url: DeepSeek API endpoint.
        thinking_enabled: Whether to enable thinking/reasoning mode.
        reasoning_effort: Reasoning effort level (low/medium/high/max).
    """

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        thinking_enabled: bool = False,
        reasoning_effort: str = "medium",
    ) -> None:
        import os

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Any:
        if self.thinking_enabled:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
            extra_body = kwargs.pop("extra_body", {})
            extra_body.setdefault("thinking", {"type": "enabled"})
            kwargs["extra_body"] = extra_body
        return await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _extract_content(self, raw_response: Any) -> str:
        message = raw_response.choices[0].message
        return message.content or ""

    def _extract_reasoning_content(self, raw_response: Any) -> str | None:
        message = raw_response.choices[0].message
        return getattr(message, "reasoning_content", None)

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        usage = raw_response.usage
        if usage is None:
            return 0, 0, 0
        return usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
