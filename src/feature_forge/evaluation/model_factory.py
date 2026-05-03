"""Model factory for creating sklearn-compatible models.

Supports XGBoost, LightGBM, CatBoost, Random Forest, and MLP.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from feature_forge.exceptions import EvaluationError


class ModelFactory:
    """Factory for creating ML models by name and task."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def get_model(self, model_name: str | None, task: str) -> Any:
        """Create a model instance.

        Args:
            model_name: One of 'xgboost', 'lightgbm', 'catboost', 'random_forest', 'mlp'.
            task: 'classification' or 'regression'.

        Returns:
            sklearn-compatible estimator.
        """
        name = (model_name or "xgboost").lower()

        if name == "xgboost":
            return self._xgboost(task)
        if name == "lightgbm":
            return self._lightgbm(task)
        if name == "catboost":
            return self._catboost(task)
        if name == "random_forest":
            return self._random_forest(task)
        if name == "mlp":
            return self._mlp(task)

        raise EvaluationError(f"Unknown model: {model_name}")

    def _xgboost(self, task: str) -> Any:
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise EvaluationError("xgboost not installed") from exc
        kwargs = {
            "n_estimators": 500,
            "learning_rate": 0.02,
            "max_depth": 6,
            "random_state": self.random_state,
            "tree_method": "hist",
            "n_jobs": 1,
        }
        return XGBClassifier(**kwargs) if task == "classification" else XGBRegressor(**kwargs)

    def _lightgbm(self, task: str) -> Any:
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise EvaluationError("lightgbm not installed") from exc
        kwargs = {
            "n_estimators": 500,
            "learning_rate": 0.02,
            "random_state": self.random_state,
            "n_jobs": 1,
            "verbose": -1,
        }
        return LGBMClassifier(**kwargs) if task == "classification" else LGBMRegressor(**kwargs)

    def _catboost(self, task: str) -> Any:
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:
            raise EvaluationError("catboost not installed") from exc
        kwargs = {
            "iterations": 500,
            "learning_rate": 0.02,
            "verbose": False,
            "random_state": self.random_state,
        }
        return CatBoostClassifier(**kwargs) if task == "classification" else CatBoostRegressor(**kwargs)

    def _random_forest(self, task: str) -> Any:
        kwargs = {"random_state": self.random_state, "n_jobs": 1}
        return RandomForestClassifier(**kwargs) if task == "classification" else RandomForestRegressor(**kwargs)

    def _mlp(self, task: str) -> Any:
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        kwargs = {
            "hidden_layer_sizes": (128, 64),
            "max_iter": 500,
            "random_state": self.random_state,
            "early_stopping": True,
        }
        return MLPClassifier(**kwargs) if task == "classification" else MLPRegressor(**kwargs)
