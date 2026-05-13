"""K-fold cross-validation evaluator for features.

Evaluates a single feature (or feature set) by adding it to the base
features and measuring the cross-validated performance change.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from feature_forge.config import Settings
from feature_forge.evaluation.metrics import get_metric
from feature_forge.evaluation.model_factory import ModelFactory
from feature_forge.exceptions import EvaluationError
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class CVEvaluator:
    """Cross-validation feature evaluator.

    Measures the gain (improvement) of adding generated features
    compared to the baseline (original features only).
    """

    def __init__(
        self,
        config: Settings | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or Settings()
        self.model_factory = model_factory or ModelFactory()
        self.metric_fn = get_metric(self.config.metric)
        self.cv_folds = self.config.evaluation.cv_folds

    def _get_cv_splitter(self, y: pd.Series) -> Any:
        """Return appropriate CV splitter for task type."""
        rs = self.config.random_state
        if self.config.task == "classification":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=rs)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=rs)

    def evaluate_baseline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str | None = None,
    ) -> float:
        score = self._cv_score(X, y, model_name=model_name)
        logger.info(
            "cv_baseline_score",
            score=round(score, 6),
            metric=self.config.metric,
            folds=self.cv_folds,
        )
        return score

    def evaluate_feature(
        self,
        X_base: pd.DataFrame,
        y: pd.Series,
        feature_df: pd.DataFrame,
        baseline_score: float | None = None,
        model_name: str | None = None,
    ) -> float:
        """Evaluate the gain of adding new features.

        Args:
            X_base: Original feature DataFrame.
            y: Target Series.
            feature_df: Generated feature DataFrame to evaluate.
            baseline_score: Precomputed baseline score (optional).
            model_name: Which model to use.

        Returns:
            Performance gain (new_score - baseline_score).
        """
        if baseline_score is None:
            baseline_score = self.evaluate_baseline(X_base, y, model_name)

        X_with_new = pd.concat([X_base, feature_df], axis=1)
        # Drop any duplicated columns (keep original)
        X_with_new = X_with_new.loc[:, ~X_with_new.columns.duplicated(keep="first")]

        new_score = self._cv_score(X_with_new, y, model_name=model_name)
        gain = new_score - baseline_score
        logger.debug(
            "cv_feature_gain",
            gain=round(gain, 6),
            new_score=round(new_score, 6),
            baseline_score=round(baseline_score, 6),
        )
        return gain

    def _cv_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str | None = None,
    ) -> float:
        """Compute cross-validated metric score."""
        model = self.model_factory.get_model(
            model_name=model_name,
            task=self.config.task,
        )
        cv = self._get_cv_splitter(y)
        scores: list[float] = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Simple preprocessing: fillna for numerical, ordinal encode categoricals
            X_train_proc, train_state = self._preprocess(X_train, fit=True)
            X_val_proc = self._preprocess(X_val, fit=False, ref_state=train_state)  # type: ignore[arg-type]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train_proc, y_train)
                    if hasattr(model, "predict_proba") and self.config.metric == "auc":
                        y_pred = model.predict_proba(X_val_proc)
                        # For binary, use positive class probability
                        if y_pred.ndim > 1 and y_pred.shape[1] == 2:
                            y_pred = y_pred[:, 1]
                    else:
                        y_pred = model.predict(X_val_proc)
                    scores.append(self.metric_fn(y_val.values, y_pred))
            except (ValueError, RuntimeError) as exc:
                raise EvaluationError(f"CV fold failed: {exc}") from exc

        return float(np.mean(scores))

    def _preprocess(
        self,
        X: pd.DataFrame,
        fit: bool = True,
        ref_state: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]] | pd.DataFrame:
        """Minimal preprocessing: fill NA, encode categoricals.

        When ``fit=True``, computes medians and category mappings from ``X``
        and returns ``(processed_df, state_dict)``.

        When ``fit=False``, uses values from ``ref_state`` (the training fold)
        to avoid data leakage between train and validation splits.
        Returns only the processed DataFrame.
        """
        X = X.copy()
        state: dict[str, Any] = {}
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                if fit:
                    cat_series = X[col].astype("category")
                    state[f"{col}_categories"] = cat_series.cat.categories
                    X[col] = cat_series.cat.codes
                else:
                    ref_categories = (ref_state or {}).get(f"{col}_categories")
                    if ref_categories is not None:
                        X[col] = (
                            X[col].astype("category").cat.set_categories(ref_categories).cat.codes
                        )
                    else:
                        X[col] = X[col].astype("category").cat.codes
            else:
                if fit:
                    median = X[col].median()
                    state[f"{col}_median"] = median
                    X[col] = X[col].fillna(median)
                else:
                    fill_value = (ref_state or {}).get(f"{col}_median", X[col].median())
                    X[col] = X[col].fillna(fill_value)
        if fit:
            return X, state
        return X
