"""Unit tests for evaluation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
        executor = SandboxedExecutor()
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
        gain = evaluator.evaluate_feature(X, y, new_feature, baseline_score=baseline, model_name="random_forest")
        assert isinstance(gain, float)
