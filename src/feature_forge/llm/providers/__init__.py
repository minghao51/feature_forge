"""LLM provider implementations."""

from feature_forge.utils import _create_lazy_getattr

__all__ = [
    "AnthropicProvider",
    "DeepSeekProvider",
    "LiteLLMProvider",
    "OpenAIProvider",
]

__getattr__ = _create_lazy_getattr(
    {
        "AnthropicProvider": "feature_forge.llm.providers.anthropic",
        "DeepSeekProvider": "feature_forge.llm.providers.deepseek",
        "LiteLLMProvider": "feature_forge.llm.providers.litellm_provider",
        "OpenAIProvider": "feature_forge.llm.providers.openai",
    },
    __name__,
)
