"""Rich exception hierarchy for feature_forge.

All exceptions inherit from FeatureForgeError for easy catch-all handling.
"""

from __future__ import annotations


class FeatureForgeError(Exception):
    """Base exception for all feature_forge errors."""


class LLMError(FeatureForgeError):
    """LLM API call failed or returned invalid response."""


class TransientLLMError(LLMError):
    """LLM failure explicitly classified as safe for bounded retry."""


class TransientNetworkError(FeatureForgeError):
    """Network failure explicitly classified as safe for bounded retry."""


class ReplayProviderError(FeatureForgeError):
    """Replay attempted to construct or call an LLM provider."""


class CodeExecutionError(FeatureForgeError):
    """Sandboxed code execution failed or was blocked."""


class SandboxValidationError(CodeExecutionError):
    """Code was blocked by static sandbox policy validation."""


class SandboxTimeoutError(CodeExecutionError):
    """Sandbox worker exceeded execution time budget."""


class AgentError(FeatureForgeError):
    """Agent operation failed (e.g., router, memory update)."""


class DatasetError(FeatureForgeError):
    """Dataset loading, ingestion, or validation failed."""


class TrackingError(FeatureForgeError):
    """Experiment tracking backend operation failed."""


class EvaluationError(FeatureForgeError):
    """Feature evaluation or model training failed."""


class PipelineError(FeatureForgeError):
    """Pipeline orchestration failed."""
