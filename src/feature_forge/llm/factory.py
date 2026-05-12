"""LLM provider factory — creates the right client from config.

Only imports the provider module that is actually needed, avoiding
unnecessary dependency loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feature_forge.config import LLMConfig, RetryConfig
    from feature_forge.llm.base import LLMClient

_DEEPSEEK_PREFIXES = ("deepseek",)
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")
_ANTHROPIC_PREFIXES = ("claude",)

_PROVIDER_REGISTRY: dict[str, tuple[str, str, str | None]] = {
    "deepseek": (
        "feature_forge.llm.providers.deepseek",
        "DeepSeekProvider",
        "https://api.deepseek.com",
    ),
    "openai": ("feature_forge.llm.providers.openai", "OpenAIProvider", "https://api.openai.com/v1"),
    "anthropic": ("feature_forge.llm.providers.anthropic", "AnthropicProvider", None),
    "litellm": ("feature_forge.llm.providers.litellm_provider", "LiteLLMProvider", None),
}


def _infer_provider(model: str) -> str:
    if any(model.startswith(p) for p in _DEEPSEEK_PREFIXES):
        return "deepseek"
    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    if any(model.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    return "litellm"


def create_llm_client(config: LLMConfig, retry_config: RetryConfig | None = None) -> LLMClient:
    """Create an LLM client from configuration.

    When ``config.provider`` is ``"auto"`` (default), the provider is
    inferred from the model name. Otherwise the explicit provider is used.

    Only the required provider module is imported.
    """
    import importlib

    provider: str = config.provider
    if provider == "auto":
        provider = _infer_provider(config.model)

    module_path, class_name, default_url = _PROVIDER_REGISTRY.get(
        provider,
        _PROVIDER_REGISTRY["litellm"],
    )
    mod = importlib.import_module(module_path)
    provider_cls = getattr(mod, class_name)

    api_key = config.api_key.get_secret_value() if config.api_key else None
    client = provider_cls(
        model=config.model,
        api_key=api_key,
        base_url=config.base_url or default_url,
    )

    if retry_config is not None:
        client.set_retry_config(retry_config)

    return client  # type: ignore[no-any-return]
