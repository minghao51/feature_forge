"""Experiment tracker abstraction.

Supports WandB (default) and MLflow (optional) backends.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class ExperimentTracker(ABC):
    """Abstract base for experiment tracking backends."""

    def __init__(self, project: str, entity: str | None = None) -> None:
        self.project = project
        self.entity = entity

    @abstractmethod
    def init_run(self, run_name: str, config: dict[str, Any]) -> None:
        """Initialize a new experiment run."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters."""

    @abstractmethod
    def log_artifact(self, path: str, artifact_type: str = "dataset") -> None:
        """Log an artifact."""

    @abstractmethod
    def finish(self) -> None:
        """Finish the current run."""

    def log_artifacts_dict(
        self,
        artifacts: dict[str, Any],
        prefix: str = "",
    ) -> None:
        """Log a dictionary of artifacts with optional key prefix."""
        for key, value in artifacts.items():
            full_key = f"{prefix}{key}" if prefix else key
            self._log_artifact_item(full_key, value)

    def _log_artifact_item(self, key: str, value: Any) -> None:
        """Type-dispatch logging for a single artifact item."""
        if value is None:
            self.log_params({key: "None"})
        elif isinstance(value, bool):
            self.log_params({key: str(value)})
        elif isinstance(value, (int, float)):
            self.log_metrics({key: float(value)})
        elif isinstance(value, pd.DataFrame):
            self._log_dataframe(key, value)
        elif hasattr(value, "load") and callable(value.load):
            try:
                self._log_dataframe(key, value.load())
            except Exception:
                self.log_params({key: f"<LazyDataFrameRef: {getattr(value, 'path', '?')}>"})
        elif isinstance(value, str) and self._looks_like_code(key, value):
            self._log_code(key, value)
        elif isinstance(value, (list, tuple)):
            self.log_params({key: json.dumps(value)})
        elif isinstance(value, dict):
            self.log_params({key: json.dumps(value)})
        elif isinstance(value, str):
            self.log_params({key: value})

    @staticmethod
    def _looks_like_code(key: str, value: str) -> bool:
        """Heuristic: does this string look like Python code?"""
        code_hints = ("def ", "import ", "class ")
        return any(hint in value for hint in code_hints) or "code" in key.lower()

    @abstractmethod
    def _log_dataframe(self, key: str, df: pd.DataFrame) -> None:
        """Log a DataFrame as a table/artifact."""

    @abstractmethod
    def _log_code(self, key: str, code: str) -> None:
        """Log a code string."""


class NoOpTracker(ExperimentTracker):
    """No-op tracker for when tracking is disabled."""

    def init_run(self, run_name: str, config: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_artifact(self, path: str, artifact_type: str = "dataset") -> None:
        pass

    def finish(self) -> None:
        pass

    def _log_dataframe(self, key: str, df: pd.DataFrame) -> None:
        pass

    def _log_code(self, key: str, code: str) -> None:
        pass
