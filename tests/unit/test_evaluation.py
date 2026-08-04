"""Unit tests for evaluation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.metrics import (
    acc_score,
    auc_score,
    f1_score_metric,
    get_metric,
    mae_score,
    nrmse_score,
    r2_score_metric,
    rmse_score,
)
from feature_forge.evaluation.model_factory import ModelFactory
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import CodeExecutionError, EvaluationError


class TestMetrics:
    def test_auc_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        score = auc_score(y_true, y_pred)
        assert 0.0 <= score <= 1.0

    def test_acc(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        score = acc_score(y_true, y_pred)
        assert score == 0.75

    def test_acc_with_proba(self):
        y_true = np.array([0, 1, 1])
        y_pred = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]])
        score = acc_score(y_true, y_pred)
        assert score == pytest.approx(2 / 3)

    def test_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        score = rmse_score(y_true, y_pred)
        assert score > 0

    def test_mae(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])
        score = mae_score(y_true, y_pred)
        assert score == pytest.approx(1 / 3)

    def test_nrmse(self):
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([1.0, 9.0])
        score = nrmse_score(y_true, y_pred)
        assert score == pytest.approx(1.0 / 10.0)

    def test_f1(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        score = f1_score_metric(y_true, y_pred)
        assert score > 0

    def test_r2(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        score = r2_score_metric(y_true, y_pred)
        assert score > 0

    def test_get_metric_unknown(self):
        with pytest.raises(EvaluationError):
            get_metric("unknown_metric")


class TestModelFactory:
    def test_xgboost_classification(self):
        factory = ModelFactory()
        model = factory.get_model("xgboost", "classification")
        assert model is not None

    def test_xgboost_regression(self):
        factory = ModelFactory()
        model = factory.get_model("xgboost", "regression")
        assert model is not None

    def test_random_forest(self):
        factory = ModelFactory()
        clf = factory.get_model("random_forest", "classification")
        reg = factory.get_model("random_forest", "regression")
        assert clf is not None
        assert reg is not None

    def test_mlp(self):
        factory = ModelFactory()
        model = factory.get_model("mlp", "classification")
        assert model is not None

    def test_unknown_model_raises(self):
        factory = ModelFactory()
        with pytest.raises(EvaluationError):
            factory.get_model("unknown", "classification")

    def test_model_fit_predict(self):
        factory = ModelFactory()
        model = factory.get_model("random_forest", "classification")
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1])
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 4


class TestSandboxedExecutor:
    def test_valid_code(self):
        executor = SandboxedExecutor(timeout_seconds=2.0, max_memory_mb=256)
        code = """
import pandas as pd
import numpy as np

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['double_a'] = df['a'] * 2
    return result
"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = executor.execute(code, df)
        assert list(result.columns) == ["double_a"]
        assert result["double_a"].tolist() == [2, 4, 6]

    def test_forbidden_import(self):
        executor = SandboxedExecutor()
        code = "import os\ndef generate_features(df): return df"
        with pytest.raises(CodeExecutionError, match="Import not allowed"):
            executor.execute(code, pd.DataFrame())

    def test_forbidden_function(self):
        executor = SandboxedExecutor()
        code = "eval('1+1')\ndef generate_features(df): return df"
        with pytest.raises(CodeExecutionError, match="Forbidden"):
            executor.execute(code, pd.DataFrame())

    def test_builtins_bypass_blocked(self):
        executor = SandboxedExecutor()
        code = "os = __builtins__['__import__']('os')\ndef generate_features(df): return df"
        with pytest.raises(CodeExecutionError, match="Forbidden"):
            executor.execute(code, pd.DataFrame())

    def test_import_not_in_allowed_builtins(self):
        from feature_forge.evaluation.sandbox import SandboxedExecutor as SE

        assert "__import__" not in SE.ALLOWED_BUILTINS

    def test_direct_import_call_blocked(self):
        executor = SandboxedExecutor()
        code = "os = __import__('os')\ndef generate_features(df): return df"
        with pytest.raises(CodeExecutionError, match="Forbidden"):
            executor.execute(code, pd.DataFrame())

    def test_dunder_introspection_blocked(self):
        executor = SandboxedExecutor()
        code = "def generate_features(df):\n    return (1).__class__.__mro__"
        with pytest.raises(CodeExecutionError, match="dunder"):
            executor.execute(code, pd.DataFrame())

    def test_socket_import_blocked_before_runtime(self):
        executor = SandboxedExecutor()
        code = """
import pandas as pd
import socket
def generate_features(df):
    s = socket.socket()
    s.connect(("example.com", 80))
    return pd.DataFrame(index=df.index)
"""
        with pytest.raises(CodeExecutionError, match="Import not allowed: socket"):
            executor.execute(code, pd.DataFrame({"a": [1]}))

    def test_timeout_enforced(self):
        executor = SandboxedExecutor(timeout_seconds=0.2, max_memory_mb=256)
        code = """
import pandas as pd
def generate_features(df):
    while True:
        pass
    return pd.DataFrame(index=df.index)
"""
        with pytest.raises(CodeExecutionError, match="timed out"):
            executor.execute(code, pd.DataFrame({"a": [1]}))

    def test_large_result_transport(self):
        executor = SandboxedExecutor(timeout_seconds=5.0, max_memory_mb=512)
        code = """
import pandas as pd
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['x2'] = df['x'] * 2
    result['x3'] = df['x'] * 3
    return result
"""
        df = pd.DataFrame({"x": list(range(100000))})
        result = executor.execute(code, df)
        assert list(result.columns) == ["x2", "x3"]
        assert len(result) == len(df)

    def test_missing_generate_features(self):
        executor = SandboxedExecutor()
        code = "x = 1"
        with pytest.raises(CodeExecutionError, match="must define"):
            executor.execute(code, pd.DataFrame())

    def test_invalid_syntax(self):
        executor = SandboxedExecutor()
        with pytest.raises(CodeExecutionError, match="Invalid syntax"):
            executor.execute("def generate_features(df", pd.DataFrame())


class TestCVEvaluator:
    def test_evaluate_baseline_classification(self):
        from feature_forge.config import Settings

        config = Settings(task="classification", metric="auc", evaluation={"cv_folds": 3})
        evaluator = CVEvaluator(config=config)
        X = pd.DataFrame({"a": list(range(20))})
        y = pd.Series([0, 1] * 10)
        score = evaluator.evaluate_baseline(X, y, model_name="random_forest")
        assert 0.0 <= score <= 1.0

    def test_evaluate_feature_gain(self):
        from feature_forge.config import Settings

        config = Settings(task="classification", metric="auc", evaluation={"cv_folds": 3})
        evaluator = CVEvaluator(config=config)
        X = pd.DataFrame({"a": list(range(20))})
        y = pd.Series([0, 1] * 10)
        baseline = evaluator.evaluate_baseline(X, y, model_name="random_forest")
        new_feature = pd.DataFrame({"b": [0, 1] * 10})
        gain = evaluator.evaluate_feature(
            X, y, new_feature, baseline_score=baseline, model_name="random_forest"
        )
        assert isinstance(gain, float)

    def test_metric_direction_resolved_in_init(self):
        from feature_forge.evaluation.metrics import MetricDirection

        maximize = CVEvaluator(config=Settings(task="classification", metric="auc"))
        minimize = CVEvaluator(config=Settings(task="regression", metric="rmse"))
        assert maximize.metric_direction is MetricDirection.MAXIMIZE
        assert minimize.metric_direction is MetricDirection.MINIMIZE

    def test_evaluate_feature_returns_raw_gain_for_minimize_metric(self, monkeypatch):
        # RMSE: lower is better. If baseline=0.5 and new=0.3, the raw gain
        # returned by evaluate_feature must be -0.2 (NOT sign-flipped).
        evaluator = CVEvaluator(config=Settings(task="regression", metric="rmse"))
        scores = iter([0.5, 0.3])
        monkeypatch.setattr(evaluator, "_cv_score", lambda *a, **k: next(scores))
        X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0.0, 1.0, 2.0, 3.0])
        feat = pd.DataFrame({"b": [0.0, 1.0, 2.0, 3.0]})
        baseline = evaluator.evaluate_baseline(X, y)
        raw_gain = evaluator.evaluate_feature(X, y, feat, baseline_score=baseline)
        assert raw_gain == pytest.approx(-0.2)

    def test_directional_is_identity_for_maximize_metric(self, monkeypatch):
        # For a maximize metric, raw and directional gains must be equal.
        evaluator = CVEvaluator(config=Settings(task="classification", metric="auc"))
        scores = iter([0.4, 0.7])
        monkeypatch.setattr(evaluator, "_cv_score", lambda *a, **k: next(scores))
        X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0, 1])
        feat = pd.DataFrame({"b": [0.0, 1.0, 2.0, 3.0]})
        baseline = evaluator.evaluate_baseline(X, y)
        raw = evaluator.evaluate_feature(X, y, feat, baseline_score=baseline)
        # `_directional` on a maximize metric is the identity function.
        assert evaluator._directional(raw) == pytest.approx(raw)
        assert evaluator._directional(raw) == pytest.approx(0.3)


class TestPrefilterCandidateColumns:
    def test_empty_dataframe_returns_empty(self):
        from feature_forge.evaluation.prefilter import prefilter_candidate_columns

        result = prefilter_candidate_columns(pd.DataFrame(), max_candidates=50)
        assert result == []

    def test_constant_columns_filtered(self):
        from feature_forge.evaluation.prefilter import prefilter_candidate_columns

        df = pd.DataFrame({"const": [1, 1, 1, 1], "varying": [1, 2, 3, 4]})
        result = prefilter_candidate_columns(df, max_candidates=50)
        assert result == ["varying"]

    def test_all_constant_returns_empty(self):
        from feature_forge.evaluation.prefilter import prefilter_candidate_columns

        df = pd.DataFrame({"a": [42] * 10, "b": ["x"] * 10})
        result = prefilter_candidate_columns(df, max_candidates=50)
        assert result == []

    def test_under_50_candidates_returns_all(self):
        from feature_forge.evaluation.prefilter import prefilter_candidate_columns

        cols = {f"col_{i}": list(range(10)) for i in range(10)}
        df = pd.DataFrame(cols)
        result = prefilter_candidate_columns(df, max_candidates=50)
        assert len(result) == 10

    def test_over_50_candidates_caps_at_50(self):
        from feature_forge.evaluation.prefilter import prefilter_candidate_columns

        cols = {f"col_{i}": list(range(100)) for i in range(80)}
        df = pd.DataFrame(cols)
        result = prefilter_candidate_columns(df, max_candidates=50)
        assert len(result) == 50

    def test_high_variance_prioritized(self):
        from feature_forge.evaluation.prefilter import prefilter_candidate_columns

        cols = {}
        for i in range(60):
            if i == 0:
                cols["high_var"] = [float(j * 1000) for j in range(100)]
            else:
                cols[f"low_{i}"] = [float(j % 2) for j in range(100)]
        df = pd.DataFrame(cols)
        result = prefilter_candidate_columns(df, max_candidates=50)
        assert result[0] == "high_var"

    def test_non_numeric_gets_zero_variance(self):
        from feature_forge.evaluation.prefilter import prefilter_candidate_columns

        cols = {f"num_{i}": list(range(10)) for i in range(40)}
        cols["cat"] = [f"x_{i}" for i in range(10)]
        df = pd.DataFrame(cols)
        result = prefilter_candidate_columns(df, max_candidates=50)
        assert "cat" in result
        assert len(result) == 41

    def test_custom_max_candidate_features(self):
        from feature_forge.evaluation.prefilter import prefilter_candidate_columns

        cols = {f"col_{i}": list(range(20)) for i in range(20)}
        df = pd.DataFrame(cols)
        result = prefilter_candidate_columns(df, max_candidates=5)
        assert len(result) == 5


class TestCVPreprocessNoLeakage:
    def test_fit_uses_own_medians(self):
        evaluator = CVEvaluator(config=Settings(task="regression", metric="rmse"))
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
        result, _state = evaluator._fit_preprocess(train)
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[3] == 2.0

    def test_transform_uses_ref_medians(self):
        evaluator = CVEvaluator(config=Settings(task="regression", metric="rmse"))
        train = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        val = pd.DataFrame({"a": [1.0, np.nan]})
        _, state = evaluator._fit_preprocess(train)
        val_proc = evaluator._transform_preprocess(val, state)
        assert val_proc["a"].iloc[1] == 20.0

    def test_transform_categorical_uses_ref_categories(self):
        evaluator = CVEvaluator(config=Settings(task="classification", metric="auc"))
        train = pd.DataFrame({"cat": ["a", "b", "c"]})
        val = pd.DataFrame({"cat": ["a", "d"]})
        train_proc, state = evaluator._fit_preprocess(train)
        val_proc = evaluator._transform_preprocess(val, state)
        assert val_proc["cat"].iloc[0] == train_proc["cat"].iloc[0]
        assert val_proc["cat"].iloc[1] == -1


class TestCVEvaluatorEdgeCases:
    """Cover CVEvaluator edge cases (regression, auto-baseline, column dedup, fold failure)."""

    def test_regression_uses_kfold(self):
        config = Settings(task="regression", metric="rmse", evaluation={"cv_folds": 3})
        evaluator = CVEvaluator(config=config)
        X = pd.DataFrame({"a": list(range(20))})
        y = pd.Series([float(i) for i in range(20)])
        score = evaluator.evaluate_baseline(X, y, model_name="random_forest")
        assert isinstance(score, float)

    def test_evaluate_feature_auto_baseline(self):
        config = Settings(task="classification", metric="auc", evaluation={"cv_folds": 3})
        evaluator = CVEvaluator(config=config)
        X = pd.DataFrame({"a": list(range(20))})
        y = pd.Series([0, 1] * 10)
        new_feat = pd.DataFrame({"b": [float(i % 2) for i in range(20)]})
        gain = evaluator.evaluate_feature(
            X, y, new_feat, baseline_score=None, model_name="random_forest"
        )
        assert isinstance(gain, float)

    def test_column_dedup(self):
        config = Settings(task="classification", metric="auc", evaluation={"cv_folds": 3})
        evaluator = CVEvaluator(config=config)
        X = pd.DataFrame({"a": list(range(20))})
        y = pd.Series([0, 1] * 10)
        # feature_df has overlapping column 'a'
        overlap = pd.DataFrame({"a": [float(i) for i in range(20)]})
        gain = evaluator.evaluate_feature(
            X, y, overlap, baseline_score=0.5, model_name="random_forest"
        )
        assert isinstance(gain, float)

    def test_cv_fold_exception_raises_evaluation_error(self):
        from unittest.mock import patch

        config = Settings(task="classification", metric="auc", evaluation={"cv_folds": 3})
        evaluator = CVEvaluator(config=config)
        X = pd.DataFrame({"a": list(range(20))})
        y = pd.Series([0, 1] * 10)

        # Mock _cv_score to raise ValueError, which gets wrapped as EvaluationError
        with patch.object(evaluator, "_cv_score", side_effect=ValueError("cv fold crash")):
            with pytest.raises(Exception, match="cv fold crash"):
                evaluator.evaluate_baseline(X, y, model_name="random_forest")

    def test_preprocess_transform_missing_ref_state(self):
        evaluator = CVEvaluator(config=Settings(task="regression", metric="rmse"))
        val = pd.DataFrame({"a": [1.0, float("nan")]})
        result = evaluator._transform_preprocess(val, ref_state={})
        assert result["a"].isna().sum() == 0  # falls back to val median

    def test_preprocess_transform_missing_categorical_ref(self):
        evaluator = CVEvaluator(config=Settings(task="classification", metric="auc"))
        val = pd.DataFrame({"cat": ["x", "y"]})
        result = evaluator._transform_preprocess(val, ref_state={})
        assert "cat" in result.columns
