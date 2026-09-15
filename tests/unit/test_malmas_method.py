"""Tests for MALMAS method adapter behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.methods.malmas.method import MALMASMethod


class _FakeForge:
    def __init__(self, config: Any, **kwargs: Any) -> None:
        self.config = config
        self.kwargs = kwargs
        self.fit_calls = 0
        self.fit_transform_calls = 0
        self.transform_calls = 0
        self.generated_scripts = ["script_a"]
        self.feature_metadata = [{"name": "f1"}]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        self.fit_calls += 1
        self.fit_kwargs = kwargs

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.transform_calls += 1
        return pd.DataFrame({"f1": [1] * len(X)}, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit_transform_calls += 1
        self.fit(X, y)
        return pd.DataFrame({"f1": [1] * len(X)}, index=X.index)

    def get_artifacts(self) -> dict[str, Any]:
        return {"generated_code": "def generate_features(df): ..."}


def _settings() -> Settings:
    return Settings(task="classification", metric="auc")


def test_transform_raises_when_not_fitted() -> None:
    method = MALMASMethod(config=_settings())
    X = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(RuntimeError, match="not fitted yet"):
        method.transform(X)


def test_fit_and_transform_delegate_to_feature_forge() -> None:
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    with patch("feature_forge.methods.malmas.method.FeatureForge", _FakeForge):
        method = MALMASMethod(config=_settings())
        method.fit(X, y)
        out = method.transform(X)

    assert list(out.columns) == ["f1"]
    assert len(out) == len(X)
    assert method.generated_scripts == ["script_a"]
    assert method.feature_metadata == [{"name": "f1"}]
    assert "generated_code" in method.get_artifacts()


def test_mode_and_client_are_forwarded_to_feature_forge() -> None:
    client = object()
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])

    with patch("feature_forge.methods.malmas.method.FeatureForge", _FakeForge):
        method = MALMASMethod(config=_settings(), mode="unary", llm_client=client)  # type: ignore[arg-type]
        method.fit(X, y)

    assert method._forge is not None
    assert method._forge.kwargs["mode"] == "unary"
    assert method._forge.kwargs["llm_client"] is client


def test_fit_transform_returns_feature_forge_output_once() -> None:
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])

    with patch("feature_forge.methods.malmas.method.FeatureForge", _FakeForge):
        method = MALMASMethod(config=_settings())
        enhanced = method.fit_transform(X, y)

    assert list(enhanced.columns) == ["f1"]
    assert enhanced.shape == (2, 1)
    assert method._forge is not None
    assert method._forge.fit_transform_calls == 1
    assert method._forge.transform_calls == 0
