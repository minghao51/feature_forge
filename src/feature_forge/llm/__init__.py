"""LLM layer for feature_forge.

Provides provider-agnostic LLM clients with optional disk caching
and Langfuse tracing (configured per-provider via ``cache`` and
``tracing_enabled`` constructor args).
"""

from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.llm.cache import DiskCache
from feature_forge.llm.factory import create_llm_client

__all__ = [
    "DiskCache",
    "LLMClient",
    "LLMResponse",
    "create_llm_client",
]
