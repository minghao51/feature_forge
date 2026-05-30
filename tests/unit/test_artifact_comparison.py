"""Tests for method artifact comparison utility."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.artifacts.base import ArtifactExporter
from feature_forge.artifacts.comparison import compare_methods


class _SimpleMethod:
    def __init__(self, should_fail_transform: bool = False) -> None:
        self.should_fail_transform = should_fail_transform
        self.fit_called = False
        self.artifact_config: Any | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.fit_called = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.should_fail_transform:
            raise RuntimeError("transform failed")
        return X.assign(new_col=1)

    def get_artifacts(self) -> dict[str, Any]:
        return {"score": 0.8}


class _BrokenMethod:
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        raise RuntimeError("fit failed")

    def get_artifacts(self) -> dict[str, Any]:
        return {}


class _Tracker:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], str]] = []

    def log_artifacts_dict(self, artifacts: dict[str, Any], prefix: str = "") -> None:
        self.calls.append((artifacts, prefix))


class _ExporterMethod(_SimpleMethod, ArtifactExporter):
    @property
    def generated_scripts(self) -> list[str]:
        return ["print('ok')"]

    def get_artifacts(self) -> dict[str, Any]:
        return {"code": "print('ok')"}


def test_compare_methods_success_and_transform_output() -> None:
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"a": [4, 5]})

    method = _SimpleMethod()
    results = compare_methods({"m": method}, X_train, y_train, X_test=X_test)

    assert method.fit_called
    assert "m" in results
    assert "score" in results["m"]
    assert "transformed_test" in results["m"]
    transformed = results["m"]["transformed_test"]
    assert isinstance(transformed, pd.DataFrame)
    assert "new_col" in transformed.columns


def test_compare_methods_captures_transform_errors() -> None:
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"a": [4, 5]})

    results = compare_methods(
        {"m": _SimpleMethod(should_fail_transform=True)},
        X_train,
        y_train,
        X_test=X_test,
    )

    assert "transformed_test_error" in results["m"]
    assert "transform failed" in results["m"]["transformed_test_error"]


def test_compare_methods_captures_fit_failures() -> None:
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])

    results = compare_methods({"broken": _BrokenMethod()}, X_train, y_train)

    assert "error" in results["broken"]
    assert "fit failed" in results["broken"]["error"]


def test_compare_methods_logs_artifacts_for_exporters() -> None:
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])
    tracker = _Tracker()

    results = compare_methods({"exp": _ExporterMethod()}, X_train, y_train, tracker=tracker)

    assert "exp" in results
    assert tracker.calls
    _, prefix = tracker.calls[0]
    assert prefix == "exp_"
