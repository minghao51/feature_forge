"""Abstract base class for baseline methods and registry."""

from __future__ import annotations

import importlib.metadata
from abc import abstractmethod
from typing import Any

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter
from feature_forge.artifacts.storage import DataFrameStorage


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

    @classmethod
    def discover(cls) -> dict[str, type[Baseline]]:
        """Discover all registered baselines from entry points."""
        baselines: dict[str, type[Baseline]] = {}
        for ep in importlib.metadata.entry_points(group=cls.ENTRY_POINT_GROUP):
            baselines[ep.name] = ep.load()
        return baselines

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
        baselines = cls.get_builtin_baselines()
        baselines.update(cls.discover())
        return baselines
