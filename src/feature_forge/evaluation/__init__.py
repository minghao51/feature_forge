"""Evaluation layer for feature_forge."""

from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.kit import EvaluationKit
from feature_forge.evaluation.metrics import MetricRegistry, get_metric
from feature_forge.evaluation.model_factory import ModelFactory, ModelRegistry
from feature_forge.evaluation.prefilter import prefilter_candidate_columns
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.evaluation.scope import row_local_violations

__all__ = [
    "CVEvaluator",
    "EvaluationKit",
    "MetricRegistry",
    "ModelFactory",
    "ModelRegistry",
    "SandboxedExecutor",
    "get_metric",
    "prefilter_candidate_columns",
    "row_local_violations",
]
