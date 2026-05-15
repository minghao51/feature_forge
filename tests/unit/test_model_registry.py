"""Tests for ModelRegistry with entry point discovery."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from feature_forge.evaluation.model_factory import ModelFactory, ModelRegistry
from feature_forge.exceptions import EvaluationError


class TestModelRegistry:
    """Verify ModelRegistry built-in + entry point discovery."""

    def test_get_builtin_returns_all_models(self):
        builtin = ModelRegistry.get_builtin()
        assert "xgboost" in builtin
        assert "lightgbm" in builtin
        assert "catboost" in builtin
        assert "random_forest" in builtin
        assert "mlp" in builtin
        assert len(builtin) >= 5

    def test_get_all_includes_builtin(self):
        all_models = ModelRegistry.get_all()
        assert "xgboost" in all_models

    def test_list_returns_names(self):
        names = ModelRegistry.list()
        assert "xgboost" in names
        assert len(names) >= 5

    def test_get_returns_factory(self):
        factory = ModelRegistry.get("xgboost")
        assert callable(factory)

    def test_get_raises_on_unknown(self):
        with pytest.raises(EvaluationError, match="Unknown model"):
            ModelRegistry.get("nonexistent_model")

    def test_register_adds_model(self):
        ModelRegistry.register("test_model", lambda task, rs: None)
        all_models = ModelRegistry.get_all()
        assert "test_model" in all_models

    def test_factory_creates_xgboost(self):
        factory = ModelFactory(random_state=42)
        model = factory.get_model("xgboost", "classification")
        assert model is not None

    def test_factory_defaults_to_xgboost(self):
        factory = ModelFactory(random_state=42)
        model = factory.get_model(None, "classification")
        assert model is not None

    def test_factory_creates_regression_model(self):
        factory = ModelFactory(random_state=42)
        model = factory.get_model("random_forest", "regression")
        assert model is not None

    def test_factory_raises_on_unknown(self):
        factory = ModelFactory(random_state=42)
        with pytest.raises(EvaluationError, match="Unknown model"):
            factory.get_model("nonexistent", "classification")

    @patch("importlib.metadata.entry_points")
    def test_entry_point_discovery(self, mock_entry_points):
        mock_entry_points.return_value.select.return_value = []
        discovered = ModelRegistry.discover()
        assert isinstance(discovered, dict)
