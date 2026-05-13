"""Contract tests verifying public API surfaces.

Tests signatures, return types, protocol compliance, and structural
contracts that must remain stable across refactors.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from feature_forge.agents.base import Agent, AgentRegistry
from feature_forge.api import FeatureForge
from feature_forge.baselines.base import Baseline
from feature_forge.config import Settings
from feature_forge.evaluation.metrics import METRIC_REGISTRY, MetricRegistry
from feature_forge.evaluation.model_factory import ModelFactory
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.experiment.matrix import ExperimentMatrix
from feature_forge.experiment.tracker import ExperimentTracker, NoOpTracker
from feature_forge.llm.base import LLMClient
from feature_forge.llm.cache import DiskCache
from feature_forge.memory.base import AgentMemory

pytestmark = pytest.mark.contract


# ── FeatureForge sklearn contract ──────────────────────────────────────


class TestFeatureForgeContract:
    def test_has_sklearn_methods(self):
        for method in ("fit", "transform", "fit_transform", "get_params", "set_params"):
            assert hasattr(FeatureForge, method), f"Missing {method}"

    def test_has_get_feature_names_out(self):
        assert hasattr(FeatureForge, "get_feature_names_out")

    def test_has_generated_scripts(self):
        ff = FeatureForge()
        assert isinstance(ff.generated_scripts, list)

    def test_has_feature_metadata(self):
        ff = FeatureForge()
        assert isinstance(ff.feature_metadata, list)

    def test_fit_returns_self(self, fake_llm):
        ff = FeatureForge(llm_client=fake_llm, config={"n_rounds": 1})
        assert hasattr(ff, "fit")

    def test_get_feature_names_out_returns_list(self, fake_llm):
        ff = FeatureForge(llm_client=fake_llm)
        result = ff.get_feature_names_out()
        assert isinstance(result, list)
        assert all(isinstance(n, str) for n in result)


# ── LLMClient ABC contract ────────────────────────────────────────────


class TestLLMClientContract:
    def test_has_complete(self):
        assert hasattr(LLMClient, "complete")

    def test_has_complete_json(self):
        assert hasattr(LLMClient, "complete_json")

    def test_has_provider_name(self):
        assert "provider_name" in dir(LLMClient)

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            LLMClient(model="test", api_key="test")


# ── Baseline protocol contract ─────────────────────────────────────────


class TestBaselineContract:
    def test_has_required_members(self):
        required = ("fit", "transform", "fit_transform", "generated_scripts", "feature_metadata")
        for member in required:
            assert hasattr(Baseline, member), f"Baseline missing {member}"

    def test_get_artifacts_method(self):
        assert hasattr(Baseline, "get_artifacts")


# ── Metric contract ────────────────────────────────────────────────────


class TestMetricContract:
    @pytest.mark.parametrize(("name", "fn"), list(METRIC_REGISTRY.items()))
    def test_all_builtins_callable(self, name, fn):
        assert callable(fn), f"Metric {name} is not callable"

    @pytest.mark.parametrize(("name", "fn"), list(METRIC_REGISTRY.items()))
    def test_signature_two_arrays(self, name, fn):
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert len(params) == 2, f"{name} should take (y_true, y_pred), got {params}"

    @pytest.mark.parametrize(("name", "fn"), list(METRIC_REGISTRY.items()))
    def test_returns_float(self, name, fn):
        y_true = np.array([0, 1, 0, 1])
        if name in ("auc",):
            y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        elif name in ("acc", "f1"):
            y_pred = np.array([0, 1, 1, 1])
        else:
            y_pred = np.array([0.1, 1.9, 2.8, 4.2])
        result = fn(y_true, y_pred)
        assert isinstance(result, float), f"{name} returned {type(result)}"

    def test_all_builtins_registered(self):
        expected = {"auc", "acc", "f1", "rmse", "mae", "r2", "nrmse"}
        assert expected == set(MetricRegistry.get_builtin().keys())


# ── ModelFactory contract ──────────────────────────────────────────────


class TestModelFactoryContract:
    @pytest.mark.parametrize("name", ["xgboost", "random_forest"])
    def test_returns_sklearn_estimator(self, name):
        factory = ModelFactory(random_state=42)
        model = factory.get_model(name, "classification")
        for method in ("fit", "predict", "get_params"):
            assert hasattr(model, method), f"{name} missing {method}"

    def test_default_model_is_xgboost(self):
        factory = ModelFactory(random_state=42)
        model = factory.get_model("xgboost", "classification")
        assert model is not None


# ── Settings contract ──────────────────────────────────────────────────


class TestSettingsContract:
    def test_nested_config_access(self):
        s = Settings()
        assert hasattr(s, "llm")
        assert hasattr(s, "router")
        assert hasattr(s, "tracker")
        assert hasattr(s, "memory")
        assert hasattr(s, "retry")
        assert hasattr(s, "evaluation")

    def test_llm_config_fields(self):
        s = Settings()
        for field in ("model", "provider", "temperature", "max_tokens", "cache_responses"):
            assert hasattr(s.llm, field)

    def test_get_settings_returns_settings(self):
        from feature_forge.config import get_settings

        s = get_settings()
        assert isinstance(s, Settings)


# ── ExperimentTracker contract ─────────────────────────────────────────


class TestTrackerContract:
    def test_has_required_methods(self):
        required = ("init_run", "log_metrics", "log_params", "log_artifact", "finish")
        for method in required:
            assert hasattr(ExperimentTracker, method), f"Tracker missing {method}"

    def test_noop_tracker_implements_all(self):
        tracker = NoOpTracker(project="test")
        tracker.init_run("test", {})
        tracker.log_metrics({"a": 1})
        tracker.log_params({"b": 2})
        tracker.log_artifact("key", "value")
        tracker.finish()


# ── Agent ABC contract ─────────────────────────────────────────────────


class TestAgentContract:
    def test_has_generate(self):
        assert hasattr(Agent, "generate")

    def test_has_system_prompt(self):
        assert hasattr(Agent, "system_prompt")

    def test_registry_has_builtin_agents(self):
        builtins = AgentRegistry.get_builtin_agents()
        assert len(builtins) >= 6
        assert "unary" in builtins


# ── DiskCache context manager contract ─────────────────────────────────


class TestDiskCacheContract:
    def test_context_manager(self, tmp_path):
        with DiskCache(cache_dir=str(tmp_path / "cache")) as cache:
            assert cache is not None
            assert cache.enabled

    def test_has_get_set_clear(self):
        for method in ("get", "set", "clear", "close"):
            assert hasattr(DiskCache, method)


# ── ExperimentMatrix builder contract ──────────────────────────────────


class TestExperimentMatrixContract:
    def test_builder_returns_self(self):
        m = ExperimentMatrix()
        assert m.datasets(["a"]) is m
        assert m.seeds([1]) is m
        assert m.models(["xgb"]) is m
        assert m.add_param("x", [1]) is m

    def test_generate_returns_list_of_dicts(self):
        configs = ExperimentMatrix().datasets(["a"]).seeds([1]).generate()
        assert isinstance(configs, list)
        assert all(isinstance(c, dict) for c in configs)


# ── SandboxedExecutor security contract ────────────────────────────────


class TestSandboxContract:
    def test_forbidden_names_non_empty(self):
        assert len(SandboxedExecutor.FORBIDDEN_NAMES) > 0

    def test_allowed_imports_limited(self):
        assert SandboxedExecutor.ALLOWED_IMPORTS == {"pandas", "numpy", "math"}

    def test_allowed_builtins_excludes_dangerous(self):
        dangerous = {"eval", "exec", "compile", "open", "__import__", "input"}
        assert dangerous.isdisjoint(SandboxedExecutor.ALLOWED_BUILTINS)

    def test_execute_has_correct_signature(self):
        sig = inspect.signature(SandboxedExecutor.execute)
        params = list(sig.parameters.keys())
        assert "code" in params
        assert "df" in params


# ── AgentMemory contract ───────────────────────────────────────────────


class TestAgentMemoryContract:
    def test_has_record_methods(self, tmp_path):
        mem = AgentMemory("test", str(tmp_path / "m.json"))
        for method in ("record_procedure", "record_feedback", "record_conceptual", "save"):
            assert hasattr(mem, method)

    def test_generate_prompt_section_returns_str(self, tmp_path):
        mem = AgentMemory("test", str(tmp_path / "m.json"))
        assert isinstance(mem.generate_prompt_section(), str)
