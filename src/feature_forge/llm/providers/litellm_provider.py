"""LiteLLM provider — unified interface for all non-DeepSeek LLMs.

LiteLLM normalises the API for 100+ providers (OpenAI, Anthropic, Google,
Azure, Bedrock, Ollama, etc.) behind a single `acompletion()` call.

Model format: "<provider>/<model>" e.g. "openai/gpt-4o", "anthropic/claude-3-sonnet".
See https://docs.litellm.ai/docs/providers for the full list.
"""

from __future__ import annotations

import json
import time
from typing import Any

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class LiteLLMProvider(LLMClient):
    """Unified LLM provider powered by LiteLLM.

    Parameters:
        model: LiteLLM model string, e.g. "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet-20241022", "gemini/gemini-2.0-flash".
        api_key: API key for the target provider.  Falls back to the
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
        """Set provider-specific environment variables before a call."""
        import os

        for key, value in self.provider_env_vars.items():
            os.environ[key] = value
        if self.api_key:
            os.environ.setdefault("API_KEY", self.api_key)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        self._setup_env()
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
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.api_key,
                api_base=self.base_url,
                **kwargs,
            )
        except Exception as exc:
            logger.error("llm_error", provider=self.provider_name, model=self.model, error=str(exc))
            raise LLMError(f"LiteLLM error ({self.model}): {exc}") from exc

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
        self._setup_env()
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
            response = await litellm.acompletion(
                model=self.model,
                messages=enhanced,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.api_key,
                api_base=self.base_url,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.error("llm_error", provider=self.provider_name, model=self.model, json_mode=True, error=str(exc))
            raise LLMError(f"LiteLLM JSON mode error ({self.model}): {exc}") from exc

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
        if not content.strip():
            raise LLMError(
                f"LiteLLM ({self.model}) returned empty content in JSON mode. "
                "Try rephrasing the prompt or increasing max_tokens."
            )
        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            logger.error("llm_json_parse_error", provider=self.provider_name, model=self.model, response_preview=content[:200])
            raise LLMError(
                f"LiteLLM ({self.model}) returned invalid JSON: "
                f"{content[:200]}... Parse error: {exc}"
            ) from exc
