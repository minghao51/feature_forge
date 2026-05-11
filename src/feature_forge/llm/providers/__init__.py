"""LLM provider implementations."""

from feature_forge.llm.base import LLMClient

__all__ = [
    "AnthropicProvider",
    "DeepSeekProvider",
    "LiteLLMProvider",
    "OpenAIProvider",
]


def __getattr__(name: str) -> type:
    _lazy = {
        "AnthropicProvider": "feature_forge.llm.providers.anthropic",
        "DeepSeekProvider": "feature_forge.llm.providers.deepseek",
        "LiteLLMProvider": "feature_forge.llm.providers.litellm_provider",
        "OpenAIProvider": "feature_forge.llm.providers.openai",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
