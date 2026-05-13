"""Metamorphic relation tests.

Tests input-output relations without ground-truth oracles:
- Perfect prediction scores, monotonicity, determinism,
  invariance, round-trip fidelity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_forge.agents.router import RouterAgent
from feature_forge.config import Settings
from feature_forge.evaluation.metrics import (
    auc_score,
    mae_score,
    nrmse_score,
    r2_score_metric,
    rmse_score,
)
from feature_forge.evaluation.sandbox import SandboxedExecutor, _to_parquet_safe
from feature_forge.experiment.matrix import ExperimentMatrix
from feature_forge.memory.base import AgentMemory
from feature_forge.utils import strip_markdown_fences

pytestmark = pytest.mark.metamorphic


# ── Metric metamorphic relations ───────────────────────────────────────


class TestMetricMetamorphic:
    def test_auc_perfect_prediction(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = y_true.astype(float)
        assert auc_score(y_true, y_pred) == pytest.approx(1.0)

    def test_rmse_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse_score(y, y) == pytest.approx(0.0)

    def test_r2_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2_score_metric(y, y) == pytest.approx(1.0)

    def test_mae_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mae_score(y, y) == pytest.approx(0.0)

    def test_nrmse_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nrmse_score(y, y) == pytest.approx(0.0)

    def test_nrmse_constant_target_returns_zero(self):
        y = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0])
        assert nrmse_score(y, y_pred) == 0.0

    def test_rmse_monotonic_with_error_magnitude(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        small_error = y + np.array([0.1, -0.1, 0.1, -0.1, 0.1])
        large_error = y + np.array([10.0, -10.0, 10.0, -10.0, 10.0])
        assert rmse_score(y, small_error) < rmse_score(y, large_error)

    def test_auc_better_prediction_higher_score(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        rng = np.random.RandomState(42)
        mediocre_pred = np.clip(0.5 + rng.randn(10) * 0.1, 0.0, 1.0)
        good_pred = y_true.astype(float)
        assert auc_score(y_true, good_pred) >= auc_score(y_true, mediocre_pred)

    def test_mae_monotonic_with_error(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        small = y + 0.1
        large = y + 1.0
        assert mae_score(y, small) < mae_score(y, large)


# ── RouterAgent metamorphic relations ──────────────────────────────────


class TestRouterMetamorphic:
    def _make_router(self, **overrides):
        return RouterAgent(Settings(**overrides))

    async def test_warmup_selects_all_agents(self):
        router = self._make_router()
        selected = await router.select_agents(round_idx=0)
        assert len(selected) == len(router.agent_names)

    async def test_post_warmup_respects_min_agents(self):
        router = self._make_router(router={"min_agents": 3})
        router.dataset_characteristics = router.analyze_dataset(
            pd.DataFrame({"a": [1]}), {"a": {"type": "numerical"}}
        )
        for strategy in ["data_driven", "performance_driven", "hybrid"]:
            router.strategy = strategy
            selected = await router.select_agents(round_idx=1)
            assert len(selected) >= 3

    async def test_max_agents_cap(self):
        router = self._make_router(router={"max_agents": 2})
        router.dataset_characteristics = router.analyze_dataset(
            pd.DataFrame({"a": [1], "b": [2]}), {"a": {"type": "numerical"}}
        )
        selected = await router.select_agents(round_idx=1)
        assert len(selected) <= 2

    def test_performance_history_bounded(self):
        router = self._make_router()
        for i in range(20):
            router.update_performance("unary", float(i))
        assert len(router.agent_performance["unary"]) <= 10

    def test_update_performance_window_is_last_n(self):
        router = self._make_router()
        for i in range(15):
            router.update_performance("unary", float(i))
        history = router.agent_performance["unary"]
        assert history == [float(i) for i in range(5, 15)]


# ── Sandbox determinism ────────────────────────────────────────────────


class TestSandboxMetamorphic:
    def test_deterministic_execution(self):
        code = """
def generate_features(df):
    result = pd.DataFrame()
    result['doubled'] = df.iloc[:, 0] * 2
    return result
"""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        executor = SandboxedExecutor(timeout_seconds=10.0)
        result1 = executor.execute(code, df)
        result2 = executor.execute(code, df)
        pd.testing.assert_frame_equal(result1, result2)

    def test_same_code_different_numeric_data(self):
        code = """
def generate_features(df):
    result = pd.DataFrame()
    result['sum'] = df.sum(axis=1)
    return result
"""
        df2 = pd.DataFrame({"a": [10.0, 20.0], "b": [30.0, 40.0]})
        executor = SandboxedExecutor(timeout_seconds=10.0)
        r2 = executor.execute(code, df2)
        assert r2["sum"].tolist() == [40.0, 60.0]


# ── _to_parquet_safe invariance ────────────────────────────────────────


class TestToParquetSafeMetamorphic:
    def test_numeric_passthrough_identity(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4, 5, 6]})
        result = _to_parquet_safe(df)
        pd.testing.assert_frame_equal(result, df)


# ── Memory save/load round-trip ────────────────────────────────────────


class TestMemoryMetamorphic:
    def test_save_load_roundtrip(self, tmp_path):
        mem = AgentMemory("test", str(tmp_path / "mem.json"))
        mem.record_procedure(["col_a"], "x*2", "feat_1", "numerical", "desc", 0)
        mem.record_feedback("feat_1", "auc", 0.05, True, 0, ["col_a"], "numerical")
        mem.save()
        loaded = AgentMemory("test", str(tmp_path / "mem.json"))
        assert len(loaded.procedural) == 1
        assert len(loaded.feedback) == 1
        assert loaded.procedural[0]["feature_name"] == "feat_1"

    def test_compute_stats_grows_with_feedback(self, tmp_path):
        mem = AgentMemory("test", str(tmp_path / "mem.json"))
        stats0 = mem.compute_stats()
        mem.record_procedure(["a"], "x*2", "f1", "numerical", "d", 0)
        mem.record_feedback("f1", "auc", 0.05, True, 0, ["a"], "numerical")
        stats1 = mem.compute_stats(min_effective=1)
        assert len(stats1["effective_transforms"]) >= len(stats0["effective_transforms"])


# ── strip_markdown_fences invariance ───────────────────────────────────


class TestStripMarkdownFencesMetamorphic:
    def test_plain_code_identity(self):
        code = "def foo():\n    return 42"
        assert strip_markdown_fences(code) == code

    def test_double_fenced_reduces_to_single_strip(self):
        inner = "def foo():\n    return 42"
        fenced = f"```python\n{inner}\n```"
        once = strip_markdown_fences(fenced)
        twice = strip_markdown_fences(strip_markdown_fences(fenced))
        assert once == twice


# ── ExperimentMatrix completeness ──────────────────────────────────────


class TestExperimentMatrixMetamorphic:
    def test_all_combinations_present(self):
        datasets = ["a", "b"]
        seeds = [1, 2, 3]
        configs = ExperimentMatrix().datasets(datasets).seeds(seeds).generate()
        for d in datasets:
            for s in seeds:
                assert any(c["dataset"] == d and c["seed"] == s for c in configs)

    def test_adding_params_increases_combinations(self):
        base = ExperimentMatrix().datasets(["a", "b"]).seeds([1])
        configs1 = base.generate()
        configs2 = base.models(["xgb", "rf"]).generate()
        assert len(configs2) == len(configs1) * 2
