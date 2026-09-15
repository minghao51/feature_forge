"""Model factory for creating sklearn-compatible models.

Supports XGBoost, LightGBM, CatBoost, Random Forest, and MLP.

Dependency note (PR 6B): the default model is ``random_forest`` (core
scikit-learn, always available). XGBoost, LightGBM, and CatBoost are
optional extras (``pip install 'feature-forge[xgboost]'`` etc.); their
factories raise :class:`EvaluationError` with a precise install hint when
the package is absent.

OpenMP note: when Intel acceleration is enabled (``sklearnex`` -> libiomp5),
the XGBoost/LightGBM pip wheels still use libgomp. To avoid the resulting
OpenMP deadlock, boosting estimators are pinned to ``n_jobs=1`` here so the
libgomp runtime never spawns threads concurrently with libiomp5. See
``docs/decisions/0010-intel-openmp-bootstrap.md``.

Parallel boosters: ``booster_n_jobs`` (``Settings.evaluation``) opts OpenMP-
backed estimators into multi-threaded fits when Intel acceleration is NOT
active. The default of ``1`` is deliberate: 2026-09-06 benchmarks on a busy
14-core host measured 16-234x REGRESSIONS for nested ``n_jobs>1`` fits
(OMP spin-wait collapse under co-located workloads). Tune only on a quiet,
measured host. When Intel acceleration is active, ``n_jobs`` is forced to 1
regardless of configuration (deadlock safety trumps performance).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, ClassVar

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from feature_forge.exceptions import EvaluationError
from feature_forge.observability.structlog_config import get_logger
from feature_forge.runtime.intel_bootstrap import is_intel_active

logger = get_logger(__name__)

_OPENMP_BACKED_FACTORIES = {"xgboost", "lightgbm", "random_forest"}
_pin_warned = False


def _booster_n_jobs(requested: int | None) -> int:
    """Resolve n_jobs for OpenMP-backed estimators (see module docstring).

    ``None``/``1`` keeps the historical single-threaded pin. Any explicit
    value is forced back to 1 when Intel acceleration is active, because the
    libgomp/libiomp5 fork deadlock (ADR 0010) must stay impossible regardless
    of configuration.
    """
    global _pin_warned
    if requested is None or requested == 1:
        return 1
    if is_intel_active():
        if not _pin_warned:
            logger.warning(
                "booster_n_jobs_forced_single_thread",
                requested=requested,
                reason="Intel acceleration active: n_jobs>1 can deadlock (ADR 0010)",
            )
            _pin_warned = True
        return 1
    return requested


def create_xgboost(task: str, random_state: int = 42, n_jobs: int | None = None) -> Any:
    """Create an XGBoost model (requires the ``xgboost`` extra)."""
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise EvaluationError(
            "xgboost is an optional dependency; install it with: "
            "pip install 'feature-forge[xgboost]'"
        ) from exc
    kwargs = {
        "n_estimators": 500,
        "learning_rate": 0.02,
        "max_depth": 6,
        "random_state": random_state,
        "tree_method": "hist",
        "n_jobs": _booster_n_jobs(n_jobs),
    }
    return XGBClassifier(**kwargs) if task == "classification" else XGBRegressor(**kwargs)


def create_lightgbm(task: str, random_state: int = 42, n_jobs: int | None = None) -> Any:
    """Create a LightGBM model."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise EvaluationError(
            "lightgbm is an optional dependency; install it with: "
            "pip install 'feature-forge[lightgbm]'"
        ) from exc
    kwargs: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.02,
        "random_state": random_state,
        "n_jobs": _booster_n_jobs(n_jobs),
        "verbose": -1,
    }
    return LGBMClassifier(**kwargs) if task == "classification" else LGBMRegressor(**kwargs)


def create_catboost(task: str, random_state: int = 42) -> Any:
    """Create a CatBoost model."""
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
        raise EvaluationError(
            "catboost is an optional dependency; install it with: "
            "pip install 'feature-forge[catboost]'"
        ) from exc
    kwargs = {
        "iterations": 500,
        "learning_rate": 0.02,
        "verbose": False,
        "random_state": random_state,
    }
    return CatBoostClassifier(**kwargs) if task == "classification" else CatBoostRegressor(**kwargs)


def create_random_forest(task: str, random_state: int = 42, n_jobs: int | None = None) -> Any:
    """Create a Random Forest model."""
    kwargs = {"random_state": random_state, "n_jobs": _booster_n_jobs(n_jobs)}
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
    def clear_cache(cls) -> None:
        """Clear cached discovered entry points."""
        cls._discovered = None

    @classmethod
    def refresh(cls) -> dict[str, Callable[..., Any]]:
        """Force re-discovery of entry-point model factories."""
        cls.clear_cache()
        return cls.get_all()

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

    def __init__(self, random_state: int = 42, booster_n_jobs: int | None = None) -> None:
        self.random_state = random_state
        self.booster_n_jobs = booster_n_jobs

    def get_model(self, model_name: str | None, task: str) -> Any:
        """Create a model instance.

        Args:
            model_name: One of 'xgboost', 'lightgbm', 'catboost', 'random_forest', 'mlp'.
                ``None`` selects the core default ``'random_forest'``. The
                boosting models are optional extras (see module docstring).
            task: 'classification' or 'regression'.

        Returns:
            sklearn-compatible estimator.
        """
        name = (model_name or "random_forest").lower()
        factory = ModelRegistry.get(name)
        if self.booster_n_jobs is not None and name in _OPENMP_BACKED_FACTORIES:
            return factory(task, self.random_state, n_jobs=self.booster_n_jobs)
        return factory(task, self.random_state)
