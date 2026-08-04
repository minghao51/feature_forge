"""K-fold cross-validation evaluator for features.

Evaluates a single feature (or feature set) by adding it to the base
features and measuring the cross-validated performance change.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from feature_forge.config import Settings
from feature_forge.evaluation.metrics import MetricDirection, get_metric, get_metric_direction
from feature_forge.evaluation.model_factory import ModelFactory
from feature_forge.exceptions import EvaluationError
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FoldEvaluation:
    """Fold metrics and row-keyed predictions for one evaluation arm."""

    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame


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
        # Optimization direction for the configured metric. Selectors should
        # consult `metric_direction` (or call `_directional` / the
        # `*_directional` evaluation methods) instead of assuming
        # "higher is better" — RMSE/MAE/NRMSE improve when the score drops.
        self.metric_direction: MetricDirection = get_metric_direction(self.config.metric)
        self.cv_folds = self.config.evaluation.cv_folds

    def _directional(self, raw_gain: float) -> float:
        """Flip the sign of a raw gain so that positive always means improvement.

        Mirrors the Platinum path's signed-delta convention
        (``dataflows/platinum.py:_paired_summary``). ``evaluate_feature`` and
        ``evaluate_features_batch`` continue to return raw gains for artifact
        transparency; selection decisions should use this helper or the
        ``*_directional`` evaluation methods.
        """
        if self.metric_direction is MetricDirection.MINIMIZE:
            return -raw_gain
        return raw_gain

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

            X_train_proc, train_state = self._fit_preprocess(X_train)
            X_val_proc = self._transform_preprocess(X_val, train_state)

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

    def evaluate_on_folds(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        row_ids: pd.Series,
        folds: pd.DataFrame,
        *,
        arm: str,
        model_name: str | None = None,
        error_context: dict[str, str] | None = None,
        estimator_threads: int = 1,
        blas_threads: int = 1,
    ) -> FoldEvaluation:
        """Evaluate using authoritative persisted validation-fold assignments."""
        if len(X) != len(y) or len(X) != len(row_ids):
            raise EvaluationError("Evaluation features, target, and row IDs are misaligned")
        if row_ids.isna().any() or row_ids.duplicated().any():
            raise EvaluationError("Evaluation row IDs must be unique and non-null")
        required = {"row_id", "fold"}
        if not required.issubset(folds.columns) or folds["row_id"].duplicated().any():
            raise EvaluationError("Fold assignments must contain unique row_id/fold pairs")
        fold_by_id = folds.set_index("row_id")["fold"]
        if set(fold_by_id.index) != set(row_ids):
            raise EvaluationError("Fold assignments do not match evaluation row IDs")
        ordered_folds = fold_by_id.loc[row_ids.tolist()].reset_index(drop=True)
        metric_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        context = error_context or {}
        for fold_id in sorted(ordered_folds.unique().tolist()):
            validation_mask = ordered_folds == fold_id
            train_idx = np.flatnonzero(~validation_mask.to_numpy())
            val_idx = np.flatnonzero(validation_mask.to_numpy())
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            started = time.perf_counter()
            try:
                X_train_proc, train_state = self._fit_preprocess(X_train)
                X_val_proc = self._transform_preprocess(X_val, train_state)
                model = self.model_factory.get_model(model_name=model_name, task=self.config.task)
                params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
                thread_params = {
                    key: estimator_threads for key in ("n_jobs", "thread_count") if key in params
                }
                if thread_params:
                    model.set_params(**thread_params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

                    with threadpool_limits(limits=blas_threads):
                        model.fit(X_train_proc, y_train)
                        if hasattr(model, "predict_proba") and self.config.metric == "auc":
                            y_pred = model.predict_proba(X_val_proc)
                            if y_pred.ndim > 1 and y_pred.shape[1] == 2:
                                y_pred = y_pred[:, 1]
                        else:
                            y_pred = model.predict(X_val_proc)
                score = self.metric_fn(y_val.values, y_pred)
            except (ValueError, RuntimeError) as exc:
                detail = ", ".join(f"{key}={value}" for key, value in sorted(context.items()))
                suffix = f" ({detail})" if detail else ""
                raise EvaluationError(
                    f"CV fold {fold_id} failed for arm={arm}, model={model_name or 'xgboost'}, "
                    f"metric={self.config.metric}{suffix}: {exc}"
                ) from exc
            metric_rows.append(
                {
                    "arm": arm,
                    "fold": int(fold_id),
                    "metric": self.config.metric,
                    "score": float(score),
                    "n_train": len(train_idx),
                    "n_validation": len(val_idx),
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "status": "succeeded",
                    "error": None,
                }
            )
            predictions = np.asarray(y_pred)
            for offset, row_position in enumerate(val_idx):
                predicted = predictions[offset]
                prediction_rows.append(
                    {
                        "arm": arm,
                        "fold": int(fold_id),
                        "row_id": str(row_ids.iloc[row_position]),
                        "target_json": json.dumps(
                            y_val.iloc[offset].item()
                            if hasattr(y_val.iloc[offset], "item")
                            else y_val.iloc[offset]
                        ),
                        "prediction_json": json.dumps(
                            predicted.tolist() if hasattr(predicted, "tolist") else predicted
                        ),
                    }
                )
        return FoldEvaluation(
            fold_metrics=pd.DataFrame(metric_rows),
            predictions=pd.DataFrame(prediction_rows),
        )

    def evaluate_features_batch(
        self,
        X_base: pd.DataFrame,
        y: pd.Series,
        features_df: pd.DataFrame,
        baseline_score: float,
        n_jobs: int = -1,
    ) -> dict[str, float]:
        from joblib import Parallel, delayed  # type: ignore[import-untyped]

        def _eval_single(col: str) -> tuple[str, float]:
            score = self.evaluate_feature(X_base, y, features_df[[col]], baseline_score)
            return (col, score)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval_single)(col) for col in features_df.columns
        )
        return dict(results)

    def evaluate_features_batch_directional(
        self,
        X_base: pd.DataFrame,
        y: pd.Series,
        features_df: pd.DataFrame,
        baseline_score: float,
        n_jobs: int = -1,
    ) -> dict[str, float]:
        """Like ``evaluate_features_batch`` but returns directional gains.

        Each value is ``_directional(raw_gain)`` so that ``value > 0`` always
        means "the feature improves the configured metric" — regardless of
        whether the metric is maximized or minimized. Feature selectors should
        prefer this over the raw batch method.
        """
        raw = self.evaluate_features_batch(
            X_base, y, features_df, baseline_score=baseline_score, n_jobs=n_jobs
        )
        return {col: self._directional(gain) for col, gain in raw.items()}

    def _fit_preprocess(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fit preprocessing: compute medians and category mappings from X.

        Returns (processed_df, state_dict) where state_dict captures
        statistics computed from this training fold to reuse on validation folds.
        """
        X = X.copy()
        state: dict[str, Any] = {}
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                cat_series = X[col].astype("category")
                state[f"{col}_categories"] = cat_series.cat.categories
                X[col] = cat_series.cat.codes
            else:
                median = X[col].median()
                state[f"{col}_median"] = median
                X[col] = X[col].fillna(median)
        return X, state

    def _transform_preprocess(self, X: pd.DataFrame, ref_state: dict[str, Any]) -> pd.DataFrame:
        """Transform preprocessing: apply previously fitted state to avoid data leakage."""
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                ref_categories = ref_state.get(f"{col}_categories")
                if ref_categories is not None:
                    X[col] = X[col].astype("category").cat.set_categories(ref_categories).cat.codes
                else:
                    X[col] = X[col].astype("category").cat.codes
            else:
                fill_value = ref_state.get(f"{col}_median", X[col].median())
                X[col] = X[col].fillna(fill_value)
        return X
