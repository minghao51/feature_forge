"""Property-based tests using Hypothesis.

Tests invariants that must hold for all valid inputs:
- Idempotency, determinism, range bounds, type preservation,
  round-trip, deduplication, size invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from feature_forge.config import EvaluationConfig, LLMConfig, RetryConfig
from feature_forge.evaluation.metrics import (
    auc_score,
    mae_score,
    nrmse_score,
    rmse_score,
)
from feature_forge.evaluation.sandbox import _to_parquet_safe
from feature_forge.experiment.matrix import ExperimentMatrix
from feature_forge.llm.cache import compute_cache_key
from feature_forge.memory.base import AgentMemory
from feature_forge.types import FeatureSpec
from feature_forge.utils import strip_markdown_fences
from tests.strategies import (
    binary_classification_data,
    feature_specs,
    llm_message_lists,
    markdown_fenced_code,
    numeric_pd_dataframes,
    pd_dataframes,
    regression_data,
)

pytestmark = pytest.mark.property


# ── strip_markdown_fences ──────────────────────────────────────────────


class TestStripMarkdownFencesProperties:
    @given(code=st.text())
    @settings(max_examples=200)
    def test_idempotent(self, code: str):
        assert strip_markdown_fences(strip_markdown_fences(code)) == strip_markdown_fences(code)

    @given(code=st.text())
    @settings(max_examples=200)
    def test_never_adds_content(self, code: str):
        assert len(strip_markdown_fences(code)) <= len(code)

    @given(code=st.text(min_size=1, alphabet="abcdefghijklmnopqrstuvwxyz0123456789 =+*"))
    @settings(max_examples=100)
    def test_plain_code_unchanged(self, code: str):
        assume(not code.startswith("```"))
        assert strip_markdown_fences(code) == code

    @given(code=markdown_fenced_code())
    @settings(max_examples=100)
    def test_output_never_starts_with_triple_backtick(self, code: str):
        result = strip_markdown_fences(code)
        assert not result.startswith("```")


# ── compute_cache_key ──────────────────────────────────────────────────


class TestCacheKeyProperties:
    @given(messages=llm_message_lists())
    @settings(max_examples=50)
    def test_deterministic(self, messages):
        kwargs = {"provider": "openai", "model": "gpt-4", "temperature": 0.2, "max_tokens": 100}
        key1 = compute_cache_key(messages=messages, **kwargs)
        key2 = compute_cache_key(messages=messages, **kwargs)
        assert key1 == key2

    @given(
        messages=llm_message_lists(),
        temperature=st.floats(0.0, 2.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_different_temperature_different_key(self, messages, temperature):
        assume(len(messages) > 0)
        key1 = compute_cache_key("openai", "gpt-4", messages, 0.2, 100)
        key2 = compute_cache_key("openai", "gpt-4", messages, temperature, 100)
        if abs(temperature - 0.2) > 1e-10:
            assert key1 != key2

    @given(messages=llm_message_lists())
    @settings(max_examples=50)
    def test_key_is_valid_hex_sha256(self, messages):
        key = compute_cache_key("test", "model", messages, 0.5, 100)
        assert len(key) == 64
        int(key, 16)


# ── ExperimentMatrix ───────────────────────────────────────────────────


class TestExperimentMatrixProperties:
    @given(
        datasets=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
        seeds=st.lists(st.integers(0, 100), min_size=1, max_size=4),
    )
    @settings(max_examples=50)
    def test_len_equals_product(self, datasets, seeds):
        m = ExperimentMatrix().datasets(datasets).seeds(seeds)
        assert len(m) == len(datasets) * len(seeds)

    @given(
        datasets=st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=3),
        seeds=st.lists(st.integers(0, 10), min_size=1, max_size=3),
    )
    @settings(max_examples=50)
    def test_generate_size_matches_len(self, datasets, seeds):
        m = ExperimentMatrix().datasets(datasets).seeds(seeds)
        configs = m.generate()
        assert len(configs) == len(m)

    @given(
        values=st.lists(st.integers(0, 5), min_size=1, max_size=3),
    )
    @settings(max_examples=30)
    def test_empty_matrix_has_len_zero(self, values):
        m = ExperimentMatrix()
        assert len(m) == 0
        assert m.generate() == []

    @given(
        datasets=st.lists(st.text(min_size=1), min_size=1, max_size=3),
        seeds=st.lists(st.integers(0, 10), min_size=1, max_size=3),
        models=st.lists(st.text(min_size=1), min_size=1, max_size=3),
    )
    @settings(max_examples=30)
    def test_generate_contains_all_combinations(self, datasets, seeds, models):
        m = ExperimentMatrix().datasets(datasets).seeds(seeds).models(models)
        configs = m.generate()
        expected = len(datasets) * len(seeds) * len(models)
        assert len(configs) == expected
        assert all("dataset" in c and "seed" in c and "model" in c for c in configs)


# ── FeatureSpec round-trip ─────────────────────────────────────────────


class TestFeatureSpecProperties:
    @given(spec=feature_specs())
    @settings(max_examples=50)
    def test_json_roundtrip(self, spec: FeatureSpec):
        data = spec.model_dump()
        restored = FeatureSpec(**data)
        assert restored == spec

    @given(spec=feature_specs())
    @settings(max_examples=50)
    def test_model_dump_json_roundtrip(self, spec: FeatureSpec):
        json_str = spec.model_dump_json()
        restored = FeatureSpec.model_validate_json(json_str)
        assert restored == spec

    @given(spec=feature_specs())
    @settings(max_examples=50)
    def test_base_columns_always_list(self, spec: FeatureSpec):
        assert isinstance(spec.base_columns, list)


# ── Metric range bounds ────────────────────────────────────────────────


class TestMetricRangeBounds:
    @given(data=binary_classification_data(min_rows=50))
    @settings(max_examples=30)
    def test_auc_range(self, data):
        _X, y = data
        assume(len(np.unique(y)) == 2)
        score = auc_score(y, y.astype(float))
        assert 0.0 <= score <= 1.0

    @given(data=regression_data(min_rows=20))
    @settings(max_examples=30)
    def test_rmse_non_negative(self, data):
        _, y = data
        score = rmse_score(y, y + np.random.randn(len(y)) * 0.1)
        assert score >= 0.0

    @given(data=regression_data(min_rows=20))
    @settings(max_examples=30)
    def test_mae_non_negative(self, data):
        _, y = data
        score = mae_score(y, y + np.random.randn(len(y)) * 0.1)
        assert score >= 0.0

    @given(data=regression_data(min_rows=20))
    @settings(max_examples=30)
    def test_nrmse_non_negative(self, data):
        _, y = data
        y_range = float(np.max(y) - np.min(y))
        assume(y_range > 1e-10)
        score = nrmse_score(y, y + np.random.randn(len(y)) * 0.1)
        assert score >= 0.0


# ── _to_parquet_safe ───────────────────────────────────────────────────


class TestToParquetSafeProperties:
    @given(df=pd_dataframes(min_rows=2, max_rows=30, min_cols=1, max_cols=5))
    @settings(max_examples=50)
    def test_row_count_preserved(self, df: pd.DataFrame):
        result = _to_parquet_safe(df)
        assert len(result) == len(df)

    @given(df=pd_dataframes(min_rows=2, max_rows=30, min_cols=1, max_cols=5))
    @settings(max_examples=50)
    def test_column_count_preserved(self, df: pd.DataFrame):
        result = _to_parquet_safe(df)
        assert len(result.columns) == len(df.columns)

    @given(df=numeric_pd_dataframes(min_rows=2, max_rows=30, min_cols=1, max_cols=5))
    @settings(max_examples=50)
    def test_numeric_passthrough(self, df: pd.DataFrame):
        result = _to_parquet_safe(df)
        for col in df.columns:
            assert result[col].dtype in [np.float64, np.int64, float, int]


# ── AgentMemory deduplication ──────────────────────────────────────────


class TestAgentMemoryProperties:
    @given(
        feature_name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_record_procedure_deduplication(self, feature_name, tmp_path):
        mem = AgentMemory("test", str(tmp_path / "mem.json"))
        mem.record_procedure(["col_a"], "x * 2", feature_name, "numerical", "desc", 0)
        count_before = len(mem.procedural)
        mem.record_procedure(["col_a"], "x * 2", feature_name, "numerical", "desc", 0)
        assert len(mem.procedural) == count_before

    @given(
        feature_name=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_record_feedback_deduplication(self, feature_name, tmp_path):
        mem = AgentMemory("test", str(tmp_path / "mem.json"))
        mem.record_feedback(feature_name, "auc", 0.05, True, 0, ["col_a"], "numerical")
        count_before = len(mem.feedback)
        mem.record_feedback(feature_name, "auc", 0.05, True, 0, ["col_a"], "numerical")
        assert len(mem.feedback) == count_before

    def test_positive_negative_disjoint(self, tmp_path):
        mem = AgentMemory("test", str(tmp_path / "mem.json"))
        mem.record_procedure(["col_a"], "x*2", "feat_good", "numerical", "desc", 0)
        mem.record_feedback("feat_good", "auc", 0.05, True, 0, ["col_a"], "numerical")
        mem.record_unused_procedure(["col_b"], "x+1", "feat_bad", "numerical", "desc", 0)
        positive, negative = mem.get_positive_negative_features()
        assert set(positive).isdisjoint(set(negative))


# ── Config validators ──────────────────────────────────────────────────


class TestConfigValidators:
    @given(temp=st.floats(0.0, 2.0))
    @settings(max_examples=20)
    def test_temperature_valid_range(self, temp):
        cfg = LLMConfig(temperature=temp)
        assert cfg.temperature == temp

    @given(temp=st.one_of(st.floats(max_value=-0.01), st.floats(min_value=2.01)))
    @settings(max_examples=20)
    def test_temperature_invalid_raises(self, temp):
        with pytest.raises(ValueError, match="temperature"):
            LLMConfig(temperature=temp)

    @given(folds=st.integers(2, 100))
    @settings(max_examples=20)
    def test_cv_folds_valid(self, folds):
        cfg = EvaluationConfig(cv_folds=folds)
        assert cfg.cv_folds == folds

    @given(folds=st.integers(max_value=1))
    @settings(max_examples=10)
    def test_cv_folds_invalid_raises(self, folds):
        with pytest.raises(ValueError, match="cv_folds"):
            EvaluationConfig(cv_folds=folds)

    @given(test_size=st.floats(0.01, 0.99))
    @settings(max_examples=20)
    def test_test_size_valid(self, test_size):
        cfg = EvaluationConfig(test_size=test_size)
        assert cfg.test_size == test_size

    @given(retries=st.integers(0, 20))
    @settings(max_examples=10)
    def test_max_retries_valid(self, retries):
        cfg = RetryConfig(max_retries=retries)
        assert cfg.max_retries == retries
