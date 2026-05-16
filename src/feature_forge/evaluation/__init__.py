"""Evaluation layer for feature_forge."""

from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.metrics import MetricRegistry, get_metric
from feature_forge.evaluation.model_factory import ModelFactory, ModelRegistry
from feature_forge.evaluation.sandbox import SandboxedExecutor

__all__ = [
    "CVEvaluator",
    "MetricRegistry",
    "ModelFactory",
    "ModelRegistry",
    "SandboxedExecutor",
    "get_metric",
]
