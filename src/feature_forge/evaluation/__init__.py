"""Evaluation layer for feature_forge.

Public names are resolved lazily via __getattr__ (PEP 562) so that importing
``feature_forge.evaluation.sandbox`` from a ``multiprocessing.spawn`` worker
does not transitively import CVEvaluator / ModelFactory / etc. and the numpy /
sklearn graph behind them. The spawn worker needs only the sandbox module and
pandas; forcing the rest on it pushes memory-constrained CI runners past OOM
at numpy import time.
"""

from typing import Any

__all__ = [
    "CVEvaluator",
    "EvaluationKit",
    "MetricRegistry",
    "ModelFactory",
    "ModelRegistry",
    "SandboxedExecutor",
    "get_metric",
    "prefilter_candidate_columns",
]


def __getattr__(name: str) -> Any:
    if name == "CVEvaluator":
        from feature_forge.evaluation.cv import CVEvaluator

        return CVEvaluator
    if name == "EvaluationKit":
        from feature_forge.evaluation.kit import EvaluationKit

        return EvaluationKit
    if name in ("MetricRegistry", "get_metric"):
        from feature_forge.evaluation.metrics import MetricRegistry, get_metric

        return MetricRegistry if name == "MetricRegistry" else get_metric
    if name in ("ModelFactory", "ModelRegistry"):
        from feature_forge.evaluation.model_factory import ModelFactory, ModelRegistry

        return ModelFactory if name == "ModelFactory" else ModelRegistry
    if name == "prefilter_candidate_columns":
        from feature_forge.evaluation.prefilter import prefilter_candidate_columns

        return prefilter_candidate_columns
    if name == "SandboxedExecutor":
        from feature_forge.evaluation.sandbox import SandboxedExecutor

        return SandboxedExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
