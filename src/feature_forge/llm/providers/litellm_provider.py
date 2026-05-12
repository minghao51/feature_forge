"""LiteLLM provider — unified interface for all non-native LLMs.

LiteLLM normalises the API for 100+ providers (OpenAI, Anthropic, Google,
Azure, Bedrock, Ollama, etc.) behind a single ``acompletion()`` call.

Model format: "<provider>/<model>" e.g. "openai/gpt-4o", "anthropic/claude-3-sonnet".
See https://docs.litellm.ai/docs/providers for the full list.
"""

from __future__ import annotations

from typing import Any

try:
    import litellm
except ImportError:
    litellm = None

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class LiteLLMProvider(LLMClient):
    """Unified LLM provider powered by LiteLLM.

    Parameters:
        model: LiteLLM model string, e.g. "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet-20241022", "gemini/gemini-2.0-flash".
        api_key: API key for the target provider. Falls back to the
            provider-specific env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
        base_url: Optional custom base URL for the provider.
        provider_env_vars: Dict of env var names → values to set before
            calling LiteLLM. Useful for non-standard providers.
    """

    @property
    def provider_name(self) -> str:
        return "litellm"

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        provider_env_vars: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        if litellm is None:
            raise LLMError("litellm is not installed. Run: uv pip install litellm")
        self.provider_env_vars = provider_env_vars or {}

    def _setup_env(self) -> None:
        import os

        for key, value in self.provider_env_vars.items():
            os.environ[key] = value
        if self.api_key:
            os.environ.setdefault("API_KEY", self.api_key)

    def _json_mode_kwargs(self) -> dict[str, Any]:
        return {"response_format": {"type": "json_object"}}

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Any:
        self._setup_env()
        return await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.base_url,
            **kwargs,
        )

    def _extract_content(self, raw_response: Any) -> str:
        return raw_response.choices[0].message.content or ""

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        usage = raw_response.usage
        if usage is None:
            return 0, 0, 0
        return usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
