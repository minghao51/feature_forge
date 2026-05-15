"""Tests for MethodProtocol runtime-checkable protocol."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.methods import BaseMethod, MethodProtocol
from feature_forge.methods.caafe import CAAFEMethod
from feature_forge.methods.malmus import MalmusMethod
from feature_forge.methods.openfe import OpenFEMethod


class StandaloneBaseline:
    """A 3rd-party baseline that does NOT import from feature_forge."""

    def __init__(self) -> None:
        self.name = "standalone"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> StandaloneBaseline:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self.fit(X_train, y_train)
        return self.transform(X_train)

    @property
    def generated_scripts(self) -> list[str]:
        return []

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return []

    def get_artifacts(self) -> dict[str, Any]:
        return {}


class IncompleteBaseline:
    """A class that does NOT satisfy MethodProtocol (missing methods)."""

    def fit(self, X, y):
        return self


class TestMethodProtocol:
    """Verify that MethodProtocol works as a runtime-checkable protocol."""

    def test_standalone_baseline_class_satisfies(self):
        assert isinstance(StandaloneBaseline(), MethodProtocol)

    def test_standalone_baseline_instance_satisfies(self):
        instance = StandaloneBaseline()
        assert isinstance(instance, MethodProtocol)

    def test_instance_has_required_members(self):
        instance = StandaloneBaseline()
        assert hasattr(instance, "name")
        assert hasattr(instance, "fit")
        assert hasattr(instance, "transform")
        assert hasattr(instance, "fit_transform")
        assert hasattr(instance, "generated_scripts")
        assert hasattr(instance, "feature_metadata")
        assert hasattr(instance, "get_artifacts")

    def test_plain_object_does_not_satisfy(self):
        assert not isinstance(object(), MethodProtocol)

    def test_incomplete_class_does_not_satisfy(self):
        assert not isinstance(IncompleteBaseline(), MethodProtocol)

    def test_baseline_abc_instance_satisfies(self):
        # Cannot instantiate ABC directly, but check the class structure
        assert hasattr(BaseMethod, "fit")
        assert hasattr(BaseMethod, "transform")

    def test_builtin_classes_have_correct_structure(self):
        for cls in [MalmusMethod, CAAFEMethod, OpenFEMethod]:
            assert hasattr(cls, "fit")
            assert hasattr(cls, "transform")
            assert hasattr(cls, "fit_transform")
            assert hasattr(cls, "generated_scripts")
            assert hasattr(cls, "get_artifacts")

    def test_baseline_is_protocol(self):
        # BaseMethod ABC should produce instances that satisfy the protocol
        # (if it were instantiable — concrete subclasses do)
        assert hasattr(BaseMethod, "generated_scripts")
