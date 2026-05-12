"""Evaluation metrics for feature engineering.

Supports both classification (AUC, ACC, F1) and regression (RMSE, MAE, R2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


def get_metric(name: str) -> Callable[..., Any]:
    """Get metric function by name."""
    if name not in METRIC_REGISTRY:
        raise EvaluationError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]
