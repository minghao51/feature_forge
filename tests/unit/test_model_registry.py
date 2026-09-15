"""Tests for ModelRegistry with entry point discovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from feature_forge.evaluation.model_factory import ModelFactory, ModelRegistry
from feature_forge.exceptions import EvaluationError


class TestModelRegistry:
    """Verify ModelRegistry built-in + entry point discovery."""

    def test_get_builtin_returns_all_models(self) -> None:
        builtin = ModelRegistry.get_builtin()
        assert "xgboost" in builtin
        assert "lightgbm" in builtin
        assert "catboost" in builtin
        assert "random_forest" in builtin
        assert "mlp" in builtin
        assert len(builtin) >= 5

    def test_get_all_includes_builtin(self) -> None:
        all_models = ModelRegistry.get_all()
        assert "xgboost" in all_models

    def test_list_returns_names(self) -> None:
        names = ModelRegistry.list()
        assert "xgboost" in names
        assert len(names) >= 5

    def test_get_returns_factory(self) -> None:
        factory = ModelRegistry.get("xgboost")
        assert callable(factory)

    def test_get_raises_on_unknown(self) -> None:
        with pytest.raises(EvaluationError, match="Unknown model"):
            ModelRegistry.get("nonexistent_model")

    def test_register_adds_model(self) -> None:
        ModelRegistry.register("test_model", lambda task, rs: None)
        all_models = ModelRegistry.get_all()
        assert "test_model" in all_models

    def test_clear_cache_and_refresh(self) -> None:
        ModelRegistry.get_all()
        first = ModelRegistry._discovered
        ModelRegistry.clear_cache()
        assert ModelRegistry._discovered is None
        ModelRegistry.refresh()
        assert ModelRegistry._discovered is not None
        assert ModelRegistry._discovered is not first

    def test_factory_creates_xgboost(self) -> None:
        pytest.importorskip("xgboost")
        factory = ModelFactory(random_state=42)
        model = factory.get_model("xgboost", "classification")
        assert model is not None

    def test_factory_defaults_to_random_forest(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        factory = ModelFactory(random_state=42)
        model = factory.get_model(None, "classification")
        assert isinstance(model, RandomForestClassifier)

    def test_factory_creates_regression_model(self) -> None:
        factory = ModelFactory(random_state=42)
        model = factory.get_model("random_forest", "regression")
        assert model is not None

    def test_factory_raises_on_unknown(self) -> None:
        factory = ModelFactory(random_state=42)
        with pytest.raises(EvaluationError, match="Unknown model"):
            factory.get_model("nonexistent", "classification")

    @patch("importlib.metadata.entry_points")
    def test_entry_point_discovery(self, mock_entry_points: MagicMock) -> None:
        mock_entry_points.return_value.select.return_value = []
        discovered = ModelRegistry.discover()
        assert isinstance(discovered, dict)


class TestBoosterNJobs:
    """Opt-in parallel boosters with the Intel deadlock guard (ADR 0010)."""

    def test_default_pin_is_single_threaded(self) -> None:
        from feature_forge.evaluation.model_factory import create_lightgbm, create_xgboost

        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        assert create_xgboost("classification").get_params()["n_jobs"] == 1
        assert create_lightgbm("classification").get_params()["n_jobs"] == 1

    def test_explicit_n_jobs_respected_when_intel_inactive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import feature_forge.evaluation.model_factory as mf

        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        monkeypatch.setattr(mf, "is_intel_active", lambda: False)
        assert mf.create_xgboost("classification", n_jobs=4).get_params()["n_jobs"] == 4
        assert mf.create_lightgbm("regression", n_jobs=-1).get_params()["n_jobs"] == -1
        assert mf.create_random_forest("classification", n_jobs=4).get_params()["n_jobs"] == 4

    def test_intel_active_forces_single_thread(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import feature_forge.evaluation.model_factory as mf

        pytest.importorskip("xgboost")
        monkeypatch.setattr(mf, "is_intel_active", lambda: True)
        monkeypatch.setattr(mf, "_pin_warned", False)  # reset once-per-process warning
        assert mf.create_xgboost("classification", n_jobs=8).get_params()["n_jobs"] == 1

    def test_factory_plumbs_booster_n_jobs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import feature_forge.evaluation.model_factory as mf

        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")
        monkeypatch.setattr(mf, "is_intel_active", lambda: False)
        factory = ModelFactory(random_state=42, booster_n_jobs=4)
        assert factory.get_model("xgboost", "classification").get_params()["n_jobs"] == 4
        assert factory.get_model("lightgbm", "classification").get_params()["n_jobs"] == 4
        assert factory.get_model("random_forest", "classification").get_params()["n_jobs"] == 4
        # mlp is not OpenMP-backed; the knob must not reach it (would TypeError)
        factory_none = ModelFactory(random_state=42)
        assert factory_none.get_model("mlp", "classification") is not None

    def test_config_booster_n_jobs_validation(self) -> None:
        from feature_forge.config import EvaluationConfig

        assert EvaluationConfig().booster_n_jobs == 1
        assert EvaluationConfig(booster_n_jobs=-1).booster_n_jobs == -1
        assert EvaluationConfig(booster_n_jobs=8).booster_n_jobs == 8
        with pytest.raises(Exception, match=r"-1 \(all cores\) or >= 1"):
            EvaluationConfig(booster_n_jobs=0)

    def test_kit_wires_booster_n_jobs(self) -> None:
        from feature_forge.config import EvaluationConfig, Settings
        from feature_forge.evaluation.kit import EvaluationKit

        kit = EvaluationKit.from_settings(Settings(evaluation=EvaluationConfig(booster_n_jobs=4)))
        assert kit.model_factory.booster_n_jobs == 4
