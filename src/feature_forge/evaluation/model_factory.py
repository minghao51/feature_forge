"""Model factory for creating sklearn-compatible models.

Supports XGBoost, LightGBM, CatBoost, Random Forest, and MLP.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, ClassVar

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from feature_forge.exceptions import EvaluationError


def create_xgboost(task: str, random_state: int = 42) -> Any:
    """Create an XGBoost model."""
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise EvaluationError("xgboost not installed") from exc
    kwargs = {
        "n_estimators": 500,
        "learning_rate": 0.02,
        "max_depth": 6,
        "random_state": random_state,
        "tree_method": "hist",
        "n_jobs": 1,
    }
    return XGBClassifier(**kwargs) if task == "classification" else XGBRegressor(**kwargs)


def create_lightgbm(task: str, random_state: int = 42) -> Any:
    """Create a LightGBM model."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise EvaluationError("lightgbm not installed") from exc
    kwargs: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.02,
        "random_state": random_state,
        "n_jobs": 1,
        "verbose": -1,
    }
    return LGBMClassifier(**kwargs) if task == "classification" else LGBMRegressor(**kwargs)


def create_catboost(task: str, random_state: int = 42) -> Any:
    """Create a CatBoost model."""
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
        raise EvaluationError("catboost not installed") from exc
    kwargs = {
        "iterations": 500,
        "learning_rate": 0.02,
        "verbose": False,
        "random_state": random_state,
    }
    return CatBoostClassifier(**kwargs) if task == "classification" else CatBoostRegressor(**kwargs)


def create_random_forest(task: str, random_state: int = 42) -> Any:
    """Create a Random Forest model."""
    kwargs = {"random_state": random_state, "n_jobs": 1}
    return (
        RandomForestClassifier(**kwargs)
        if task == "classification"
        else RandomForestRegressor(**kwargs)
    )


def create_mlp(task: str, random_state: int = 42) -> Any:
    """Create an MLP model."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    kwargs = {
        "hidden_layer_sizes": (128, 64),
        "max_iter": 500,
        "random_state": random_state,
        "early_stopping": True,
    }
    return MLPClassifier(**kwargs) if task == "classification" else MLPRegressor(**kwargs)


class ModelRegistry:
    """Registry for ML models with entry point discovery.

    Built-in models (xgboost, lightgbm, catboost, random_forest, mlp) are
    registered by default. Additional models can be discovered via the
    ``feature_forge.models`` entry point group, or registered programmatically.
    """

    ENTRY_POINT_GROUP = "feature_forge.models"

    _builtin: ClassVar[dict[str, Callable[..., Any]]] = {}
    _discovered: ClassVar[dict[str, Callable[..., Any]] | None] = None

    @classmethod
    def get_builtin(cls) -> dict[str, Callable[..., Any]]:
        """Return built-in model factory functions."""
        if not cls._builtin:
            cls._builtin = {
                "xgboost": create_xgboost,
                "lightgbm": create_lightgbm,
                "catboost": create_catboost,
                "random_forest": create_random_forest,
                "mlp": create_mlp,
            }
        return dict(cls._builtin)

    @classmethod
    def discover(cls) -> dict[str, Callable[..., Any]]:
        """Discover models registered via entry points."""
        from feature_forge.evaluation.registry_utils import discover_entry_points

        return discover_entry_points(cls.ENTRY_POINT_GROUP, builtins=cls.get_builtin())

    @classmethod
    def get_all(cls) -> dict[str, Callable[..., Any]]:
        """Return built-in + entry-point discovered model factories."""
        if cls._discovered is None:
            cls._discovered = cls.discover()
        return {**cls.get_builtin(), **cls._discovered}

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """Get a model factory function by name."""
        factories = cls.get_all()
        if name not in factories:
            raise EvaluationError(f"Unknown model: {name}. Available: {list(factories.keys())}")
        return factories[name]

    @classmethod
    def register(cls, name: str, factory_fn: Callable[..., Any]) -> None:
        """Register a model factory programmatically."""
        if name in cls._builtin:
            warnings.warn(
                f"Model '{name}' already registered. Overwriting.",
                RuntimeWarning,
                stacklevel=2,
            )
        cls._builtin[name] = factory_fn

    @classmethod
    def list(cls) -> list[str]:
        """Return list of available model names."""
        return list(cls.get_all().keys())


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
        factory = ModelRegistry.get(name)
        return factory(task, self.random_state)
