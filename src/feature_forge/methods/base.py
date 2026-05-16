"""Abstract base class for feature engineering methods and registry."""

from __future__ import annotations

import importlib.metadata
import warnings
from abc import abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter
from feature_forge.artifacts.storage import DataFrameStorage
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class _FeatureExecutionProtocol(Protocol):
    def execute(self, code: str, df: pd.DataFrame) -> pd.DataFrame: ...


@runtime_checkable
class MethodProtocol(Protocol):
    """Protocol for any method with no dependency on feature_forge internals.

    Third-party methods can satisfy this protocol without importing from feature_forge.
    """

    name: str

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> MethodProtocol:
        """Fit the method."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted method."""

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Fit and transform training data."""

    @property
    def generated_scripts(self) -> list[str]:
        """Return generated code blocks."""

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        """Return feature metadata."""

    def get_artifacts(self) -> dict[str, Any]:
        """Return all collected artifacts."""


class BaseMethod(ArtifactExporter):
    """Abstract base for feature engineering methods.

    All methods must implement fit/transform for sklearn compatibility.
    Inherits from ArtifactExporter for unified artifact access.
    """

    def __init__(
        self,
        name: str,
        artifact_config: ArtifactConfig | None = None,
    ) -> None:
        super().__init__(artifact_config=artifact_config)
        self.name = name
        self._storage = DataFrameStorage(artifact_config or ArtifactConfig())
        self._artifacts: dict[str, Any] = {}

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseMethod:
        """Fit the method."""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted method."""

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Fit and transform training data."""
        self.fit(X_train, y_train)
        return self.transform(X_train)

    def get_artifacts(self) -> dict[str, Any]:
        """Return all collected artifacts."""
        return dict(self._artifacts)

    @property
    def generated_scripts(self) -> list[str]:
        """Return generated code blocks."""
        code = self._artifacts.get("generated_code", "")
        return [code] if code else []

    def _iterative_feature_metadata(self, method_name: str) -> list[dict[str, Any]]:
        iterations = self._artifacts.get("iterations")
        if not iterations:
            return []
        meta: list[dict[str, Any]] = []
        for it in iterations:
            for col, gain in it.get("gains", {}).items():
                meta.append(
                    {
                        "name": col,
                        "method": method_name,
                        "iteration": it.get("iteration"),
                        "gain": gain,
                        "kept": gain > 0,
                        "code": it.get("generated_code", ""),
                    }
                )
        return meta

    def _iterative_provenance_records(self, method_name: str) -> list[dict[str, Any]]:
        iterations = self._artifacts.get("iterations")
        if not iterations:
            return []
        records: list[dict[str, Any]] = []
        for it in iterations:
            for col, gain in it.get("gains", {}).items():
                records.append(
                    {
                        "feature_name": col,
                        "source_method": method_name,
                        "iteration_index": it.get("iteration"),
                        "generated_code": it.get("generated_code", ""),
                        "cv_gain": gain,
                    }
                )
        return records

    def _should_raise_on_feature_error(self) -> bool:
        return (
            hasattr(self, "evaluator")
            and self.evaluator is not None
            and self.evaluator.config.evaluation.fail_on_feature_error
        )

    def _transform_via_iteration_codes(self, X: pd.DataFrame) -> pd.DataFrame:
        iteration_codes = getattr(self, "_iteration_codes", None)
        if not iteration_codes:
            raise RuntimeError(f"{self.name} not fitted yet")
        sandbox = getattr(self, "sandbox", None)
        if sandbox is None or not hasattr(sandbox, "execute"):
            raise RuntimeError(f"{self.name} missing sandbox executor")
        result = X.copy()
        for code in iteration_codes:
            try:
                sandbox_exec: _FeatureExecutionProtocol = sandbox
                features = sandbox_exec.execute(code, result)
                for col in features.columns:
                    if col not in result.columns:
                        result[col] = features[col].values
            except Exception as exc:
                logger.warning("transform_step_failed", method=self.name, error=str(exc))
                if self._should_raise_on_feature_error():
                    raise
        new_cols = [c for c in result.columns if c not in X.columns]
        return result[new_cols]


class MethodRegistry:
    """Discover methods via Python entry points."""

    ENTRY_POINT_GROUP = "feature_forge.methods"

    _discovered: ClassVar[dict[str, type[BaseMethod]] | None] = None

    @classmethod
    def discover(cls) -> dict[str, type[BaseMethod]]:
        """Discover all registered methods from entry points."""
        discovered: dict[str, type[BaseMethod]] = {}
        for ep in importlib.metadata.entry_points(group=cls.ENTRY_POINT_GROUP):
            try:
                loaded = ep.load()
            except Exception as exc:
                warnings.warn(
                    f"Failed to load method entry point '{ep.name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if not isinstance(loaded, type):
                warnings.warn(
                    f"Method entry point '{ep.name}' is not a class. Skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            required = {
                "fit",
                "transform",
                "fit_transform",
                "generated_scripts",
                "feature_metadata",
                "get_artifacts",
            }
            if not required.issubset(dir(loaded)):
                warnings.warn(
                    f"Method entry point '{ep.name}' does not satisfy MethodProtocol. Skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if ep.name in discovered:
                warnings.warn(
                    f"Duplicate method entry point name '{ep.name}'. Last registered wins.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            discovered[ep.name] = loaded
        return discovered

    @classmethod
    def get_builtin_methods(cls) -> dict[str, type[BaseMethod]]:
        """Return built-in methods without entry point discovery."""
        from feature_forge.methods.caafe.method import CAAFEMethod
        from feature_forge.methods.llmfe.method import LLMFEMethod
        from feature_forge.methods.malmas.method import MALMASMethod
        from feature_forge.methods.malmus.method import MalmusMethod
        from feature_forge.methods.openfe.method import OpenFEMethod

        return {
            "malmas": MALMASMethod,
            "openfe": OpenFEMethod,
            "caafe": CAAFEMethod,
            "llmfe": LLMFEMethod,
            "malmus": MalmusMethod,
        }

    @classmethod
    def get_all_methods(cls) -> dict[str, type[BaseMethod]]:
        """Return built-in + entry-point registered methods."""
        if cls._discovered is None:
            cls._discovered = cls.discover()
        methods = cls.get_builtin_methods()
        for name, method_cls in cls._discovered.items():
            if name in methods and methods[name] is method_cls:
                continue
            methods[name] = method_cls
        return methods
