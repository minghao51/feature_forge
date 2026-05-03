"""Rich exception hierarchy for feature_forge.

All exceptions inherit from FeatureForgeError for easy catch-all handling.
"""

from __future__ import annotations


class FeatureForgeError(Exception):
    """Base exception for all feature_forge errors."""


class ConfigurationError(FeatureForgeError):
    """Invalid or missing configuration."""


class LLMError(FeatureForgeError):
    """LLM API call failed or returned invalid response."""


class FeatureGenerationError(FeatureForgeError):
    """Feature generation pipeline step failed."""


class CodeExecutionError(FeatureForgeError):
    """Sandboxed code execution failed or was blocked."""


class AgentError(FeatureForgeError):
    """Agent operation failed (e.g., router, memory update)."""


class MemoryError(FeatureForgeError):
    """Memory system operation failed."""


class DatasetError(FeatureForgeError):
    """Dataset loading, ingestion, or validation failed."""


class TrackingError(FeatureForgeError):
    """Experiment tracking backend operation failed."""


class EvaluationError(FeatureForgeError):
    """Feature evaluation or model training failed."""


class PipelineError(FeatureForgeError):
    """Pipeline orchestration failed."""
