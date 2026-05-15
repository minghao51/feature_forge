"""Evaluation metrics for feature engineering.

Supports both classification (AUC, ACC, F1) and regression (RMSE, MAE, R2).
"""

from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from feature_forge.exceptions import EvaluationError


def auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute ROC AUC score.

    Supports both binary and multiclass (ovr) classification.
    """
    classes = np.unique(y_true)
    if len(classes) == 2:
        return float(roc_auc_score(y_true, y_pred))
    # multiclass
    try:
        return float(roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro"))
    except Exception as exc:
        raise EvaluationError(f"AUC computation failed: {exc}") from exc


def acc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score.

    For classification, y_pred should be class probabilities or labels.
    """
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return float(accuracy_score(y_true, y_pred))


def f1_score_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score (macro average)."""
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return float(f1_score(y_true, y_pred, average="macro"))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def r2_score_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 score."""
    return float(r2_score(y_true, y_pred))


def nrmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute normalized RMSE (RMSE / (max - min) of y_true)."""
    rmse = rmse_score(y_true, y_pred)
    y_range = float(np.max(y_true) - np.min(y_true))
    if y_range == 0:
        return 0.0
    return rmse / y_range


METRIC_REGISTRY: dict[str, Callable[..., Any]] = {
    "auc": auc_score,
    "acc": acc_score,
    "f1": f1_score_metric,
    "rmse": rmse_score,
    "mae": mae_score,
    "r2": r2_score_metric,
    "nrmse": nrmse_score,
}


class MetricRegistry:
    """Registry for evaluation metrics with entry point discovery.

    Built-in metrics are registered by default. Additional metrics can be
    discovered via the ``feature_forge.metrics`` entry point group, or
    registered programmatically via ``register()``.
    """

    ENTRY_POINT_GROUP = "feature_forge.metrics"

    _builtin: ClassVar[dict[str, Callable[..., Any]]] = dict(METRIC_REGISTRY)
    _discovered: ClassVar[dict[str, Callable[..., Any]] | None] = None

    @classmethod
    def discover(cls) -> dict[str, Callable[..., Any]]:
        """Discover metrics registered via entry points."""
        discovered: dict[str, Callable[..., Any]] = {}
        builtin = cls._builtin
        for ep in importlib.metadata.entry_points(group=cls.ENTRY_POINT_GROUP):
            try:
                loaded = ep.load()
            except Exception as exc:
                warnings.warn(
                    f"Failed to load metric entry point '{ep.name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if ep.name in builtin and builtin[ep.name] is not loaded:
                warnings.warn(
                    f"Entry point metric '{ep.name}' overrides built-in metric.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if ep.name in discovered:
                warnings.warn(
                    f"Duplicate metric entry point name '{ep.name}'. Last registered wins.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            discovered[ep.name] = loaded
        return discovered

    @classmethod
    def get_builtin(cls) -> dict[str, Callable[..., Any]]:
        """Return built-in metrics only."""
        return dict(cls._builtin)

    @classmethod
    def get_all(cls) -> dict[str, Callable[..., Any]]:
        """Return built-in + entry-point discovered metrics."""
        if cls._discovered is None:
            cls._discovered = cls.discover()
        return {**cls._builtin, **cls._discovered}

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """Get a metric function by name."""
        metrics = cls.get_all()
        if name not in metrics:
            raise EvaluationError(f"Unknown metric: {name}. Available: {list(metrics.keys())}")
        return metrics[name]

    @classmethod
    def register(cls, name: str, fn: Callable[..., Any]) -> None:
        """Register a metric programmatically."""
        if name in cls._builtin:
            warnings.warn(
                f"Metric '{name}' already registered. Overwriting.",
                RuntimeWarning,
                stacklevel=2,
            )
        cls._builtin[name] = fn


def get_metric(name: str) -> Callable[..., Any]:
    """Get metric function by name (delegates to MetricRegistry)."""
    return MetricRegistry.get(name)
