"""DeepSeek LLM provider.

DeepSeek uses the OpenAI-compatible API. Inherits from OpenAIProvider
which covers all the similarity; this subclass only overrides the
provider name and default connection details.
"""

from __future__ import annotations

from feature_forge.llm.providers.openai import OpenAIProvider
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek async LLM client.

    Inherits from OpenAIProvider since DeepSeek's API is fully
    OpenAI-compatible.

    Parameters:
        model: DeepSeek model name. Defaults to 'deepseek-chat'.
        api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
        base_url: DeepSeek API endpoint.
    """

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        import os

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        super().__init__(model=model, api_key=api_key, base_url=base_url)
