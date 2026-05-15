"""Abstract base class for feature engineering methods and registry."""

from __future__ import annotations

import importlib.metadata
import warnings
from abc import abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter
from feature_forge.artifacts.storage import DataFrameStorage


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
