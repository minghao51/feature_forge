"""Differential tests comparing implementation against known references.

Verifies that custom implementations match sklearn/numpy/pandas
references within floating-point tolerance.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from feature_forge.evaluation.metrics import (
    acc_score,
    auc_score,
    f1_score_metric,
    mae_score,
    r2_score_metric,
    rmse_score,
)
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.experiment.matrix import ExperimentMatrix
from feature_forge.llm.cache import compute_cache_key
from feature_forge.utils import strip_markdown_fences

pytestmark = pytest.mark.differential


# ── Metric vs sklearn differential ─────────────────────────────────────


class TestMetricsDifferential:
    def test_auc_vs_sklearn(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=200)
        y_pred = rng.rand(200)
        ours = auc_score(y_true, y_pred)
        ref = float(roc_auc_score(y_true, y_pred))
        assert ours == pytest.approx(ref, abs=1e-10)

    def test_acc_vs_sklearn_labels(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        ours = acc_score(y_true, y_pred)
        ref = float(accuracy_score(y_true, y_pred))
        assert ours == pytest.approx(ref, abs=1e-10)

    def test_f1_vs_sklearn(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        ours = f1_score_metric(y_true, y_pred)
        ref = float(f1_score(y_true, y_pred, average="macro"))
        assert ours == pytest.approx(ref, abs=1e-10)

    def test_rmse_vs_sklearn(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(100)
        y_pred = y_true + rng.randn(100) * 0.1
        ours = rmse_score(y_true, y_pred)
        ref = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        assert ours == pytest.approx(ref, abs=1e-10)

    def test_mae_vs_sklearn(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(100)
        y_pred = y_true + rng.randn(100) * 0.1
        ours = mae_score(y_true, y_pred)
        ref = float(mean_absolute_error(y_true, y_pred))
        assert ours == pytest.approx(ref, abs=1e-10)

    def test_r2_vs_sklearn(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(100)
        y_pred = y_true + rng.randn(100) * 0.1
        ours = r2_score_metric(y_true, y_pred)
        ref = float(r2_score(y_true, y_pred))
        assert ours == pytest.approx(ref, abs=1e-10)


# ── ExperimentMatrix vs itertools.product differential ─────────────────


class TestExperimentMatrixDifferential:
    def test_vs_itertools_product(self):
        datasets = ["titanic", "houses"]
        seeds = [0, 1, 2]
        models = ["xgboost", "rf"]
        ours = ExperimentMatrix().datasets(datasets).seeds(seeds).models(models).generate()
        keys = ["dataset", "seed", "model"]
        ref = [
            dict(zip(keys, combo, strict=True))
            for combo in itertools.product(datasets, seeds, models)
        ]
        assert len(ours) == len(ref)
        for ref_combo in ref:
            assert any(
                c["dataset"] == ref_combo["dataset"]
                and c["seed"] == ref_combo["seed"]
                and c["model"] == ref_combo["model"]
                for c in ours
            )


# ── Sandbox vs direct pandas differential ──────────────────────────────


class TestSandboxDifferential:
    def test_multiply_vs_direct(self):
        code = """
def generate_features(df):
    result = pd.DataFrame()
    result['doubled'] = df['a'] * 2
    return result
"""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        executor = SandboxedExecutor(timeout_seconds=10.0)
        sandbox_result = executor.execute(code, df)
        direct_result = df.copy()
        direct_result["doubled"] = df["a"] * 2
        np.testing.assert_array_almost_equal(
            sandbox_result["doubled"].values, direct_result["doubled"].values
        )

    def test_add_columns_vs_direct(self):
        code = """
def generate_features(df):
    result = pd.DataFrame()
    result['sum'] = df['a'] + df['b']
    return result
"""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        executor = SandboxedExecutor(timeout_seconds=10.0)
        sandbox_result = executor.execute(code, df)
        direct = df["a"] + df["b"]
        np.testing.assert_array_almost_equal(sandbox_result["sum"].values, direct.values)

    def test_log_transform_vs_direct(self):
        code = """
import numpy as np
def generate_features(df):
    result = pd.DataFrame()
    result['log_a'] = np.log1p(df['a'].abs())
    return result
"""
        df = pd.DataFrame({"a": [1.0, 10.0, 100.0, 0.0]})
        executor = SandboxedExecutor(timeout_seconds=10.0)
        sandbox_result = executor.execute(code, df)
        direct = np.log1p(df["a"].abs())
        np.testing.assert_array_almost_equal(sandbox_result["log_a"].values, direct.values)


# ── strip_markdown_fences vs regex differential ────────────────────────


class TestStripMarkdownFencesDifferential:
    @staticmethod
    def _regex_impl(code: str) -> str:
        import re

        return re.sub(r"^```(?:python)?\n?", "", re.sub(r"\n?```$", "", code))

    def test_python_fence_vs_regex(self):
        code = "```python\nprint('hello')\n```"
        assert strip_markdown_fences(code) == self._regex_impl(code)

    def test_bare_fence_vs_regex(self):
        code = "```\nprint('hello')\n```"
        assert strip_markdown_fences(code) == self._regex_impl(code)

    def test_no_fence_vs_regex(self):
        code = "print('hello')"
        assert strip_markdown_fences(code) == self._regex_impl(code)


# ── compute_cache_key vs manual SHA-256 ────────────────────────────────


class TestCacheKeyDifferential:
    def test_vs_manual_sha256(self):
        provider = "openai"
        model = "gpt-4"
        messages = [{"role": "user", "content": "hello"}]
        temperature = 0.7
        max_tokens = 100
        kwargs = {}
        ours = compute_cache_key(provider, model, messages, temperature, max_tokens, **kwargs)
        payload = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            "extra": kwargs,
        }
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        ref = hashlib.sha256(data.encode("utf-8")).hexdigest()
        assert ours == ref
