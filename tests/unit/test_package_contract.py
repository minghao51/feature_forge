"""Package dependency contract tests (PR 6B).

The core install (``uv sync`` / ``pip install feature-forge``, no extras)
must not require optional heavyweight method/model libraries:

* ``pyproject.toml`` keeps them out of ``project.dependencies`` and exposes
  dedicated extras (also included in ``all``);
* the standard default model (``random_forest``) stays usable core-only;
* optional method/model factories fail with a precise
  ``feature-forge[...]`` install hint when the package is absent.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from feature_forge.evaluation.model_factory import (
    ModelFactory,
    create_catboost,
    create_lightgbm,
    create_xgboost,
)
from feature_forge.exceptions import EvaluationError
from feature_forge.methods.caafe.method import CAAFEMethod
from feature_forge.methods.openfe.method import OpenFEMethod

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

OPTIONAL_ML_PACKAGES = ("caafe", "openfe", "xgboost", "lightgbm", "catboost")


def _load_pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


class TestDependencyPlacement:
    """Heavyweight method/model packages live only in extras."""

    def test_core_dependencies_exclude_optional_packages(self) -> None:
        deps = _load_pyproject()["project"]["dependencies"]
        core_names = {d.split(">=")[0].split(">")[0].split("==")[0].strip() for d in deps}
        for package in OPTIONAL_ML_PACKAGES:
            assert package not in core_names, (
                f"{package} must be an optional extra, not a core dependency"
            )

    @pytest.mark.parametrize("package", OPTIONAL_ML_PACKAGES)
    def test_package_has_dedicated_extra(self, package: str) -> None:
        extras = _load_pyproject()["project"]["optional-dependencies"]
        assert package in extras, f"missing extra: {package}"
        assert any(pkg.split(">")[0].split("<")[0].strip() == package for pkg in extras[package])

    def test_all_extra_includes_optional_packages(self) -> None:
        extras = _load_pyproject()["project"]["optional-dependencies"]
        all_names = {pkg.split(">")[0].split("<")[0].strip() for pkg in extras["all"]}
        for package in OPTIONAL_ML_PACKAGES:
            assert package in all_names, f"'all' extra must include {package}"


class TestDefaultModelCoreOnly:
    """The standard default model must not need any extras."""

    def test_default_model_is_random_forest(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        model = ModelFactory(random_state=42).get_model(None, "classification")
        assert isinstance(model, RandomForestClassifier)

    def test_default_model_fit_predict(self) -> None:
        factory = ModelFactory(random_state=42)
        model = factory.get_model(None, "classification")
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1])
        model.fit(X, y)
        assert len(model.predict(X)) == 4


class TestOptionalInstallHints:
    """Optional factories fail with a precise `feature-forge[...]` hint."""

    def test_xgboost_missing_gives_install_hint(self) -> None:
        with patch.dict("sys.modules", {"xgboost": None}):
            with pytest.raises(EvaluationError, match=r"feature-forge\[xgboost\]"):
                create_xgboost("classification")

    @pytest.mark.parametrize(
        ("package", "factory"),
        [("lightgbm", create_lightgbm), ("catboost", create_catboost)],
    )
    def test_optional_model_missing_gives_install_hint(self, package: str, factory: Any) -> None:
        with patch.dict("sys.modules", {package: None}):
            with pytest.raises(EvaluationError, match=rf"feature-forge\[{package}\]"):
                factory("classification")

    def test_caafe_fidelity_missing_gives_install_hint(self) -> None:
        method = CAAFEMethod(variant="fidelity")
        with patch.dict("sys.modules", {"caafe": None}):
            with pytest.raises(EvaluationError, match=r"feature-forge\[caafe\]"):
                method.fit(pd.DataFrame({"a": [1, 2, 3]}), pd.Series([0, 1, 0]))

    def test_openfe_missing_gives_install_hint_on_fit(self) -> None:
        method = OpenFEMethod()
        with patch.dict("sys.modules", {"openfe": None}):
            with pytest.raises(EvaluationError, match=r"feature-forge\[openfe\]"):
                method.fit(pd.DataFrame({"a": [1, 2, 3]}), pd.Series([0, 1, 0]))

    @pytest.mark.parametrize("package", OPTIONAL_ML_PACKAGES)
    def test_optional_packages_absent_from_core_install(self, package: str) -> None:
        """Meaningful only in a core-only environment (as in the CI job)."""
        if importlib.util.find_spec(package) is not None:
            pytest.skip(
                f"{package} is installed (extras present); core-only CI job verifies absence"
            )
