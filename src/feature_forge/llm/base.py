"""Abstract base class for LLM clients.

All LLM providers must implement LLMClient to ensure a unified interface
across OpenAI, DeepSeek, Anthropic, LiteLLM, and future providers.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


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
    ) -> None:
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.raw_response = raw_response

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.total_tokens}, "
            f"content={self.content[:80]!r}...)"
        )


class LLMClient(ABC):
    """Abstract base class for LLM providers.

    Implementations must provide:
    - `complete()`: Send messages and return structured response
    - `complete_json()`: Send messages with JSON mode, return parsed dict
    - `provider_name`: String identifier for the provider
    """

    def __init__(self, model: str, api_key: str | None, base_url: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g. 'openai', 'deepseek', 'litellm')."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request to the LLM.

        Args:
            messages: OpenAI-style message list, e.g.
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific extra parameters.

        Returns:
            LLMResponse with content and token usage.

        Raises:
            LLMError: On API failure or invalid response.
        """

    @abstractmethod
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a completion request with JSON mode enabled.

        The provider must force valid JSON output and inject the schema
        description into the prompt so the model conforms to the expected
        structure.

        Args:
            messages: OpenAI-style message list.
            schema_description: Human-readable description of the expected
                JSON schema (used as guidance in the system prompt).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Parsed JSON dict from the model response.

        Raises:
            LLMError: On API failure or invalid JSON response.
        """

    @staticmethod
    def _inject_json_schema(
        messages: list[dict[str, str]],
        schema_description: str,
    ) -> list[dict[str, str]]:
        """Inject JSON schema guidance into messages.

        Prepends schema instructions to the system message, or inserts
        a system message if none exists. The word 'json' is included
        to satisfy providers that require it for JSON mode activation.
        """
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

    def build_cache_key(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        """Build a deterministic cache key from request parameters.

        Override if provider-specific kwargs should be included.
        """
        import hashlib

        payload = {
            "provider": self.provider_name,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            "extra": kwargs,
        }
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
