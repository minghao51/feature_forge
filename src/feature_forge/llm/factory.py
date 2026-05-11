"""LLM provider factory — creates the right client from config.

Only imports the provider module that is actually needed, avoiding
unnecessary dependency loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feature_forge.config import LLMConfig
    from feature_forge.llm.base import LLMClient

_DEEPSEEK_PREFIXES = ("deepseek",)
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")
_ANTHROPIC_PREFIXES = ("claude",)


def _infer_provider(model: str) -> str:
    if any(model.startswith(p) for p in _DEEPSEEK_PREFIXES):
        return "deepseek"
    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    if any(model.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    return "litellm"


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client from configuration.

    When ``config.provider`` is ``"auto"`` (default), the provider is
    inferred from the model name. Otherwise the explicit provider is used.

    Only the required provider module is imported.
    """
    provider: str = config.provider
    if provider == "auto":
        provider = _infer_provider(config.model)

    api_key = config.api_key.get_secret_value() if config.api_key else None

    if provider == "deepseek":
        from feature_forge.llm.providers.deepseek import DeepSeekProvider

        return DeepSeekProvider(
            model=config.model,
            api_key=api_key,
            base_url=config.base_url or "https://api.deepseek.com",
        )

    if provider == "openai":
        from feature_forge.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=config.model,
            api_key=api_key,
            base_url=config.base_url or "https://api.openai.com/v1",
        )

    if provider == "anthropic":
        from feature_forge.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=config.model,
            api_key=api_key,
            base_url=config.base_url,
        )

    from feature_forge.llm.providers.litellm_provider import LiteLLMProvider

    return LiteLLMProvider(
        model=config.model,
        api_key=api_key,
        base_url=config.base_url,
    )
