"""Abstract base class for baseline methods and registry."""

from __future__ import annotations

import importlib.metadata
import warnings
from abc import abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter
from feature_forge.artifacts.storage import DataFrameStorage


@runtime_checkable
class BaselineProtocol(Protocol):
    """Protocol for any baseline method — no dependency on feature_forge internals.

    3rd-party baselines can satisfy this protocol without importing from feature_forge.
    """

    name: str

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaselineProtocol:
        """Fit the baseline method."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted baseline."""

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


class Baseline(ArtifactExporter):
    """Abstract base for baseline feature engineering methods.

    All baselines must implement fit/transform for sklearn compatibility.
    Inherits from ArtifactExporter for unified artifact access.
    """

    def __init__(
        self,
        name: str,
        artifact_config: ArtifactConfig | None = None,
    ) -> None:
        super().__init__(artifact_config=artifact_config)
        self.name = name
        self.feature_names: list[str] = []
        self._storage = DataFrameStorage(artifact_config or ArtifactConfig())
        self._artifacts: dict[str, Any] = {}

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Baseline:
        """Fit the baseline method."""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted baseline."""

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


class BaselineRegistry:
    """Discover baselines via Python entry points."""

    ENTRY_POINT_GROUP = "feature_forge.baselines"

    _discovered: ClassVar[dict[str, type[Baseline]] | None] = None

    @classmethod
    def discover(cls) -> dict[str, type[Baseline]]:
        """Discover all registered baselines from entry points."""
        discovered: dict[str, type[Baseline]] = {}
        for ep in importlib.metadata.entry_points(group=cls.ENTRY_POINT_GROUP):
            try:
                loaded = ep.load()
            except Exception as exc:
                warnings.warn(
                    f"Failed to load baseline entry point '{ep.name}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if not isinstance(loaded, type):
                warnings.warn(
                    f"Baseline entry point '{ep.name}' is not a class. Skipping.",
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
                    f"Baseline entry point '{ep.name}' does not satisfy BaselineProtocol. "
                    f"Skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if ep.name in discovered:
                warnings.warn(
                    f"Duplicate baseline entry point name '{ep.name}'. Last registered wins.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            discovered[ep.name] = loaded
        return discovered

    @classmethod
    def get_builtin_baselines(cls) -> dict[str, type[Baseline]]:
        """Return built-in baselines without entry point discovery."""
        from feature_forge.baselines.caafe import CAAFEBaseline
        from feature_forge.baselines.llmfe import LLMFEBaseline
        from feature_forge.baselines.malmus import MalmusBaseline
        from feature_forge.baselines.openfe import OpenFEBaseline

        return {
            "openfe": OpenFEBaseline,
            "caafe": CAAFEBaseline,
            "llmfe": LLMFEBaseline,
            "malmus": MalmusBaseline,
        }

    @classmethod
    def get_all_baselines(cls) -> dict[str, type[Baseline]]:
        """Return built-in + entry-point registered baselines."""
        if cls._discovered is None:
            cls._discovered = cls.discover()
        baselines = cls.get_builtin_baselines()
        for name, bl_cls in cls._discovered.items():
            if name in baselines:
                warnings.warn(
                    f"Entry point baseline '{name}' overrides built-in baseline.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            baselines[name] = bl_cls
        return baselines
