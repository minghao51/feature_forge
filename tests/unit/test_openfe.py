"""Tests for OpenFE baseline method.

OpenFE is an optional dependency. These tests cover the wrapper logic,
artifact extraction helpers, and error handling — without requiring
the actual openfe package installed (except for integration paths).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from feature_forge.exceptions import EvaluationError
from feature_forge.methods.base import BaseMethod
from feature_forge.methods.openfe.method import OpenFEMethod

pytestmark = pytest.mark.contract


class TestOpenFEContract:
    def test_is_subclass_of_baseline(self):
        assert issubclass(OpenFEMethod, BaseMethod)

    def test_has_required_members(self):
        for attr in ("fit", "transform", "fit_transform", "generated_scripts", "feature_metadata"):
            assert hasattr(OpenFEMethod, attr)

    def test_generated_scripts_empty(self):
        method = OpenFEMethod()
        assert method.generated_scripts == []

    def test_feature_metadata_empty_when_no_artifacts(self):
        method = OpenFEMethod()
        assert method.feature_metadata == []

    def test_default_name(self):
        method = OpenFEMethod()
        assert method.name == "openfe"


class TestOpenFEErrors:
    def test_transform_before_fit_raises(self):
        method = OpenFEMethod()
        with pytest.raises(EvaluationError, match="not fitted"):
            method.transform(pd.DataFrame({"a": [1]}))

    def test_fit_missing_openfe_raises(self):
        method = OpenFEMethod()
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        with patch.dict("sys.modules", {"openfe": None}):
            with pytest.raises(EvaluationError, match="openfe not installed"):
                method.fit(X, y)


class TestOperatorNames:
    """Cover _operator_names static method (lines 102-116)."""

    def test_none_returns_none(self):
        assert OpenFEMethod._operator_names(None) is None

    def test_node_with_name(self):
        node = MagicMock()
        node.name = "add(a, b)"
        result = OpenFEMethod._operator_names([node])
        assert result == ["add(a, b)"]

    def test_node_with_repr(self):
        node = MagicMock()
        del node.name
        node.__repr__ = MagicMock(return_value="CustomNode(a, b)")
        result = OpenFEMethod._operator_names([node])
        assert result == ["CustomNode(a, b)"]

    def test_node_with_repr_fallback(self):
        node = MagicMock()
        del node.name
        node.__repr__ = MagicMock(return_value="ReprNode(x, y)")
        result = OpenFEMethod._operator_names([node])
        assert result == ["ReprNode(x, y)"]

    def test_multiple_operators(self):
        nodes = []
        for name in ("add", "multiply", "log"):
            node = MagicMock()
            node.name = name
            nodes.append(node)
        result = OpenFEMethod._operator_names(nodes)
        assert result == ["add", "multiply", "log"]


class TestImportanceDf:
    """Cover _importance_df static method (lines 118-128)."""

    def test_numpy_array_returns_dataframe(self):
        arr = np.array([0.1, 0.2, 0.3])
        result = OpenFEMethod._importance_df(arr)
        assert isinstance(result, pd.DataFrame)
        assert list(result["rank"]) == [0, 1, 2]
        assert list(result["importance"]) == [0.1, 0.2, 0.3]

    def test_invalid_importance_returns_empty(self):
        result = OpenFEMethod._importance_df("not-an-array")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_empty_array(self):
        arr = np.array([])
        result = OpenFEMethod._importance_df(arr)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestExtractArtifacts:
    """Cover _extract_artifacts with mocked OpenFE internals."""

    def test_extracts_selected_operators(self):
        method = OpenFEMethod()
        mock_node = MagicMock()
        mock_node.name = "add(a, b)"
        method._features = [mock_node]

        method._extract_artifacts()
        assert method._artifacts["selected_operators"] == ["add(a, b)"]

    def test_candidate_features_none_warns(self):
        method = OpenFEMethod()
        mock_node = MagicMock()
        mock_node.name = "add(a, b)"
        method._features = [mock_node]
        method._ofe = MagicMock()
        method._ofe.candidate_features_list = None

        with pytest.warns(UserWarning, match="candidate_features_list"):
            method._extract_artifacts()
        assert method._artifacts["candidate_operators"] is None

    def test_importances_none_warns(self):
        method = OpenFEMethod()
        mock_node = MagicMock()
        mock_node.name = "add(a, b)"
        method._features = [mock_node]
        method._ofe = MagicMock()
        method._ofe.candidate_features_list = [mock_node]
        method._ofe.feature_importances_ = None

        with pytest.warns(UserWarning, match="feature_importances_"):
            method._extract_artifacts()
        assert method._artifacts["feature_importances"] is None

    def test_full_artifact_extraction(self):
        method = OpenFEMethod()
        node = MagicMock()
        node.name = "mul(a, b)"
        method._features = [node]
        method._ofe = MagicMock()
        method._ofe.candidate_features_list = [node]
        method._ofe.feature_importances_ = np.array([0.5])

        method._extract_artifacts()
        assert method._artifacts["selected_operators"] == ["mul(a, b)"]
        assert method._artifacts["candidate_operators"] == ["mul(a, b)"]
        assert isinstance(method._artifacts["feature_importances"], pd.DataFrame)
