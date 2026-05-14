"""Abstract base class for LLM clients.

All LLM providers must implement LLMClient to ensure a unified interface
across OpenAI, DeepSeek, Anthropic, LiteLLM, and future providers.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from pydantic import SecretStr

from feature_forge.exceptions import LLMError
from feature_forge.observability.structlog_config import get_logger

if TYPE_CHECKING:
    from feature_forge.config import RetryConfig

logger = get_logger(__name__)

JSONValue: TypeAlias = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None


class LLMResponse:
    """Structured LLM response with token usage metadata."""

    def __init__(
        self,
        content: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        raw_response: Any | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.raw_response = raw_response
        self.reasoning_content = reasoning_content

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.total_tokens}, "
            f"content={self.content[:80]!r}...)"
        )


class LLMClient(ABC):
    """Abstract base class for LLM providers.

    Subclasses implement three hooks:
    - ``_call_api()``: Raw API call returning a provider-specific response object.
    - ``_extract_content()``: Extract text from the raw response.
    - ``_extract_usage()``: Extract token counts from the raw response.

    The base class handles logging, timing, token extraction, retry,
    JSON schema injection, and error wrapping.
    """

    def __init__(
        self,
        model: str,
        api_key: str | SecretStr | None,
        base_url: str | None = None,
        thinking_enabled: bool = False,
        reasoning_effort: str = "medium",
    ) -> None:
        self.model = model
        self._api_key_secret: SecretStr | None = None
        if isinstance(api_key, SecretStr):
            self._api_key_secret = api_key
        elif api_key is not None:
            self._api_key_secret = SecretStr(api_key)
        self.base_url = base_url
        self._retry_config: RetryConfig | None = None
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort

    @property
    def api_key(self) -> str | None:
        return self._api_key_secret.get_secret_value() if self._api_key_secret else None

    @property
    def api_key_secret(self) -> SecretStr | None:
        return self._api_key_secret

    def set_retry_config(self, config: RetryConfig) -> None:
        """Attach retry configuration to this client."""
        self._retry_config = config

    async def _retry(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute ``fn`` with retry if config is set, otherwise call directly."""
        if self._retry_config is None:
            return await fn(*args, **kwargs)
        from feature_forge.llm.retry import build_async_retry

        return await build_async_retry(self._retry_config).wraps(fn)(*args, **kwargs)

    # ── provider hooks (override these, or override _do_complete directly) ──

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g. 'openai', 'deepseek', 'litellm')."""

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Any:
        """Call the provider API. Returns the raw response object."""
        raise NotImplementedError

    def _extract_content(self, raw_response: Any) -> str:
        """Extract text content from the provider's raw response."""
        raise NotImplementedError

    def _extract_reasoning_content(self, raw_response: Any) -> str | None:
        """Extract reasoning/thinking content if the provider supports it."""
        return None

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        """Extract (prompt_tokens, completion_tokens, total_tokens) from raw response."""
        raise NotImplementedError

    def _json_mode_kwargs(self) -> dict[str, Any]:
        """Return provider-specific kwargs for JSON mode requests."""
        return {}

    # ── public API with retry ───────────────────────────────────────

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request with automatic retry on transient failures.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific extra parameters.

        Returns:
            LLMResponse with content and token usage.

        Raises:
            LLMError: On API failure after all retries exhausted.
        """
        return await self._retry(self._do_complete, messages, temperature, max_tokens, **kwargs)  # type: ignore[no-any-return]

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> JSONValue:
        """Send a completion request with JSON mode and automatic retry.

        Args:
            messages: OpenAI-style message list.
            schema_description: Human-readable description of the expected
                JSON schema.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Parsed JSON value from the model response.

        Raises:
            LLMError: On API failure or invalid JSON after all retries.
        """
        return await self._retry(  # type: ignore[no-any-return]
            self._do_complete_json, messages, schema_description, temperature, max_tokens
        )

    # ── core completion logic ───────────────────────────────────────

    async def _do_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        logger.info(
            "llm_request",
            provider=self.provider_name,
            model=self.model,
            num_messages=len(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        t0 = time.perf_counter()
        try:
            raw = await self._call_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except LLMError:
            raise
        except Exception as exc:
            logger.error(
                "llm_error",
                provider=self.provider_name,
                model=self.model,
                json_mode=json_mode,
                error=str(exc),
            )
            raise LLMError(f"{self.provider_name} API error: {exc}") from exc

        content = self._extract_content(raw)
        reasoning_content = self._extract_reasoning_content(raw)
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(raw)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            "llm_response",
            provider=self.provider_name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            json_mode=json_mode,
            response_preview=content[:200],
        )

        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=raw,
            reasoning_content=reasoning_content,
        )

    async def _do_complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> JSONValue:
        enhanced = self._inject_json_schema(messages, schema_description)
        json_kwargs = self._json_mode_kwargs()
        response = await self._do_complete(
            enhanced, temperature, max_tokens, json_mode=True, **json_kwargs
        )
        return self._parse_json_response(response.content)

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _inject_json_schema(
        messages: list[dict[str, str]],
        schema_description: str,
    ) -> list[dict[str, str]]:
        """Inject JSON schema guidance into messages."""
        schema_instruction = (
            "You MUST respond with valid JSON matching this schema:\n"
            f"{schema_description}\n"
            "Output ONLY the JSON object, no markdown fences or explanation.\n\n"
        )
        enhanced: list[dict[str, str]] = []
        system_injected = False
        for msg in messages:
            if msg.get("role") == "system" and not system_injected:
                enhanced.append(
                    {
                        "role": "system",
                        "content": schema_instruction + msg.get("content", ""),
                    }
                )
                system_injected = True
            else:
                enhanced.append(msg)
        if not system_injected:
            enhanced.insert(0, {"role": "system", "content": schema_instruction})
        return enhanced

    def _parse_json_response(self, content: str) -> JSONValue:
        """Parse JSON from LLM content with error handling."""
        content = content.strip()
        if not content:
            raise LLMError(
                f"{self.provider_name} returned empty content in JSON mode. "
                "Try rephrasing the prompt or increasing max_tokens."
            )
        try:
            return cast(JSONValue, json.loads(content))
        except json.JSONDecodeError as exc:
            logger.error(
                "llm_json_parse_error",
                provider=self.provider_name,
                model=self.model,
                response_preview=content[:200],
            )
            raise LLMError(
                f"{self.provider_name} returned invalid JSON: {content[:200]}... Parse error: {exc}"
            ) from exc

    def build_cache_key(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Build a deterministic cache key from request parameters."""
        from feature_forge.llm.cache import compute_cache_key

        return compute_cache_key(
            self.provider_name, self.model, messages, temperature, max_tokens, **kwargs
        )
