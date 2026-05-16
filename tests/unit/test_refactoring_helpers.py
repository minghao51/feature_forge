"""Unit tests for helpers introduced in the code simplification refactoring."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import CodeExecutionError
from feature_forge.methods.base import BaseMethod
from feature_forge.methods.malmas.pipeline.iterative import IterativePipeline
from feature_forge.types import FeatureSpec


class _ConcreteMethod(BaseMethod):
    """Minimal concrete subclass for testing BaseMethod helpers."""

    def __init__(self, name: str = "test_method") -> None:
        super().__init__(name=name)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseMethod:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class TestFromEvaluatorFactory:
    def test_from_evaluator_none_returns_defaults(self):
        executor = SandboxedExecutor.from_evaluator(None)
        assert executor.limits.timeout_seconds == 5.0
        assert executor.limits.max_memory_mb == 512

    def test_from_evaluator_with_config(self):
        config = Settings(
            task="classification",
            metric="auc",
            evaluation={"sandbox_timeout_seconds": 10.0, "sandbox_max_memory_mb": 256},
        )
        evaluator = CVEvaluator(config=config)
        executor = SandboxedExecutor.from_evaluator(evaluator)
        assert executor.limits.timeout_seconds == 10.0
        assert executor.limits.max_memory_mb == 256

    def test_from_evaluator_returns_sandboxed_executor(self):
        executor = SandboxedExecutor.from_evaluator(None)
        assert isinstance(executor, SandboxedExecutor)


class TestIterativeFeatureMetadata:
    def test_empty_artifacts_returns_empty(self):
        method = _ConcreteMethod()
        assert method._iterative_feature_metadata("test") == []

    def test_single_iteration(self):
        method = _ConcreteMethod()
        method._artifacts["iterations"] = [
            {
                "iteration": 0,
                "gains": {"feat_a": 0.05, "feat_b": -0.02},
                "kept": True,
                "generated_code": "df['feat_a'] = df['x'] * 2",
            }
        ]
        meta = method._iterative_feature_metadata("caafe")
        assert len(meta) == 2
        assert meta[0]["name"] == "feat_a"
        assert meta[0]["method"] == "caafe"
        assert meta[0]["gain"] == 0.05
        assert meta[1]["name"] == "feat_b"
        assert meta[1]["gain"] == -0.02

    def test_multiple_iterations(self):
        method = _ConcreteMethod()
        method._artifacts["iterations"] = [
            {"iteration": 0, "gains": {"a": 0.1}, "generated_code": "code0"},
            {"iteration": 1, "gains": {"b": 0.2}, "generated_code": "code1"},
        ]
        meta = method._iterative_feature_metadata("llmfe")
        assert len(meta) == 2
        assert meta[0]["iteration"] == 0
        assert meta[1]["iteration"] == 1

    def test_kept_is_derived_per_feature_gain(self):
        method = _ConcreteMethod()
        method._artifacts["iterations"] = [
            {
                "iteration": 0,
                "gains": {"feat_good": 0.1, "feat_bad": -0.1},
                "kept": True,
                "generated_code": "code0",
            }
        ]
        meta = method._iterative_feature_metadata("llmfe")
        by_name = {m["name"]: m for m in meta}
        assert by_name["feat_good"]["kept"] is True
        assert by_name["feat_bad"]["kept"] is False

    def test_missing_gains_key(self):
        method = _ConcreteMethod()
        method._artifacts["iterations"] = [{"iteration": 0}]
        meta = method._iterative_feature_metadata("test")
        assert meta == []


class TestIterativeProvenanceRecords:
    def test_empty_artifacts_returns_empty(self):
        method = _ConcreteMethod()
        assert method._iterative_provenance_records("test") == []

    def test_single_iteration(self):
        method = _ConcreteMethod()
        method._artifacts["iterations"] = [
            {
                "iteration": 0,
                "gains": {"feat_a": 0.05},
                "generated_code": "df['feat_a'] = df['x'] * 2",
            }
        ]
        records = method._iterative_provenance_records("malmus")
        assert len(records) == 1
        assert records[0]["feature_name"] == "feat_a"
        assert records[0]["source_method"] == "malmus"
        assert records[0]["cv_gain"] == 0.05
        assert records[0]["generated_code"] == "df['feat_a'] = df['x'] * 2"

    def test_provenance_includes_iteration_index(self):
        method = _ConcreteMethod()
        method._artifacts["iterations"] = [
            {"iteration": 3, "gains": {"x": 0.1}, "generated_code": ""},
        ]
        records = method._iterative_provenance_records("test")
        assert records[0]["iteration_index"] == 3


class TestTransformViaIterationCodes:
    def test_raises_when_not_fitted(self):
        method = _ConcreteMethod()
        with pytest.raises(RuntimeError, match="not fitted yet"):
            method._transform_via_iteration_codes(pd.DataFrame({"a": [1, 2]}))

    def test_raises_when_empty_codes(self):
        method = _ConcreteMethod()
        method._iteration_codes = []
        with pytest.raises(RuntimeError, match="not fitted yet"):
            method._transform_via_iteration_codes(pd.DataFrame({"a": [1, 2]}))

    def test_applies_codes_and_returns_new_cols(self):
        method = _ConcreteMethod()
        method._iteration_codes = [
            "import pandas as pd\ndef generate_features(df):\n    return pd.DataFrame({'double_a': df['a'] * 2}, index=df.index)"
        ]
        sandbox = SandboxedExecutor(timeout_seconds=5.0)
        method.sandbox = sandbox
        X = pd.DataFrame({"a": [1, 2, 3]})
        result = method._transform_via_iteration_codes(X)
        assert "double_a" in result.columns
        assert result["double_a"].tolist() == [2, 4, 6]

    def test_skips_bad_code_and_returns_good_results(self):
        method = _ConcreteMethod()
        method._iteration_codes = [
            "bad syntax {{{",
            "import pandas as pd\ndef generate_features(df):\n    return pd.DataFrame({'b': df['a'] + 1}, index=df.index)",
        ]
        method.sandbox = SandboxedExecutor(timeout_seconds=5.0)
        X = pd.DataFrame({"a": [1, 2, 3]})
        result = method._transform_via_iteration_codes(X)
        assert "b" in result.columns
        assert result["b"].tolist() == [2, 3, 4]

    def test_raises_on_bad_code_when_fail_on_error(self):
        method = _ConcreteMethod()
        method._iteration_codes = ["bad syntax {{{"]
        method.sandbox = SandboxedExecutor(timeout_seconds=5.0)
        mock_evaluator = MagicMock()
        mock_evaluator.config.evaluation.fail_on_feature_error = True
        method.evaluator = mock_evaluator
        X = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(CodeExecutionError, match="Invalid syntax"):
            method._transform_via_iteration_codes(X)


class TestShouldRaiseOnFeatureError:
    def test_returns_false_when_no_evaluator(self):
        method = _ConcreteMethod()
        assert method._should_raise_on_feature_error() is False

    def test_returns_false_when_evaluator_is_none(self):
        method = _ConcreteMethod()
        method.evaluator = None
        assert method._should_raise_on_feature_error() is False

    def test_returns_false_when_fail_on_error_is_false(self):
        method = _ConcreteMethod()
        mock_evaluator = MagicMock()
        mock_evaluator.config.evaluation.fail_on_feature_error = False
        method.evaluator = mock_evaluator
        assert method._should_raise_on_feature_error() is False

    def test_returns_true_when_fail_on_error_is_true(self):
        method = _ConcreteMethod()
        mock_evaluator = MagicMock()
        mock_evaluator.config.evaluation.fail_on_feature_error = True
        method.evaluator = mock_evaluator
        assert method._should_raise_on_feature_error() is True


class TestRecordFeatureInMemory:
    def test_records_effective_feature(self):
        memory = MagicMock()
        spec = FeatureSpec(
            name="feat_a",
            type="numerical",
            transform="df['x'] * 2",
            logic="double x",
            base_columns=["x"],
            agent_name="unary",
        )
        IterativePipeline._record_feature_in_memory(
            memory, spec, gain=0.05, round_idx=0, metric="auc"
        )
        memory.record_procedure.assert_called_once_with(
            base_columns=["x"],
            transform="df['x'] * 2",
            feature_name="feat_a",
            ty="numerical",
            description="double x",
            round_idx=0,
        )
        memory.record_feedback.assert_called_once_with(
            feature_name="feat_a",
            metric="auc",
            value=0.05,
            effective=True,
            round_idx=0,
            base=["x"],
            ty="numerical",
        )
        memory.record_unused_procedure.assert_not_called()

    def test_records_ineffective_feature_with_unused(self):
        memory = MagicMock()
        spec = FeatureSpec(
            name="feat_b",
            type="numerical",
            transform="df['y'] + 1",
            logic="increment y",
            base_columns=["y"],
            agent_name="unary",
        )
        IterativePipeline._record_feature_in_memory(
            memory, spec, gain=-0.03, round_idx=1, metric="rmse"
        )
        memory.record_procedure.assert_called_once()
        memory.record_feedback.assert_called_once_with(
            feature_name="feat_b",
            metric="rmse",
            value=-0.03,
            effective=False,
            round_idx=1,
            base=["y"],
            ty="numerical",
        )
        memory.record_unused_procedure.assert_called_once_with(
            base_columns=["y"],
            transform="df['y'] + 1",
            feature_name="feat_b",
            ty="numerical",
            description="increment y",
            round_idx=1,
        )

    def test_zero_gain_is_ineffective(self):
        memory = MagicMock()
        spec = FeatureSpec(name="zero_feat", base_columns=["a"])
        IterativePipeline._record_feature_in_memory(
            memory, spec, gain=0.0, round_idx=0, metric="auc"
        )
        memory.record_feedback.assert_called_once_with(
            feature_name="zero_feat",
            metric="auc",
            value=0.0,
            effective=False,
            round_idx=0,
            base=["a"],
            ty="numerical",
        )
        memory.record_unused_procedure.assert_called_once()
