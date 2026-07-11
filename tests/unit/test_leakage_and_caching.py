"""Tests for feature leakage filtering and fit_transform caching."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pandas as pd

from feature_forge.llm.base import LLMResponse
from feature_forge.methods.caafe.method import CAAFEMethod
from feature_forge.methods.llmfe.method import LLMFEMethod
from feature_forge.methods.malmus.method import MalmusMethod


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self.content, model="fake")

    async def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return json.loads(self.content)


def test_llmfe_leakage_filtering_and_caching() -> None:
    from feature_forge.config import get_settings

    code = """
import pandas as pd
def generate_features(df):
    res = pd.DataFrame(index=df.index)
    res['feat_good'] = df['a'] * 2
    res['feat_bad'] = df['b'] * 3
    return res
"""
    llm = FakeLLM(code)
    evaluator = MagicMock()
    evaluator.config = get_settings()
    evaluator.evaluate_baseline.return_value = 0.5
    evaluator.evaluate_features_batch.return_value = {"feat_good": 0.1, "feat_bad": 0.0}

    method = LLMFEMethod(
        llm_client=llm,
        mode="iterative",
        n_features=1,
        evaluator=evaluator,
    )
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = pd.Series([0, 1])

    method.fit(X, y)
    assert method._kept_features == ["feat_good"]

    # Transform should only add the kept feature
    X_trans = method.transform(X)
    assert "feat_good" in X_trans.columns
    assert "feat_bad" not in X_trans.columns

    # Check cache exists
    assert "pipeline_result" in method._artifacts
    assert "X_train_enhanced" in method._artifacts["pipeline_result"]
    assert "feat_good" in method._artifacts["pipeline_result"]["X_train_enhanced"].columns
    assert "feat_bad" not in method._artifacts["pipeline_result"]["X_train_enhanced"].columns


def test_caafe_leakage_filtering_and_caching() -> None:
    from feature_forge.config import get_settings

    code = """
import pandas as pd
def generate_features(df):
    res = pd.DataFrame(index=df.index)
    res['feat_good'] = df['a'] * 2
    res['feat_bad'] = df['b'] * 3
    return res
"""
    llm = FakeLLM(code)
    evaluator = MagicMock()
    evaluator.config = get_settings()
    evaluator.evaluate_baseline.return_value = 0.5
    evaluator.evaluate_features_batch.return_value = {"feat_good": 0.1, "feat_bad": 0.0}

    method = CAAFEMethod(
        llm_client=llm,
        iterations=1,
        variant="unified",
        evaluator=evaluator,
    )
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = pd.Series([0, 1])

    method.fit(X, y)
    assert method._kept_features == ["feat_good"]

    X_trans = method.transform(X)
    assert "feat_good" in X_trans.columns
    assert "feat_bad" not in X_trans.columns

    assert "pipeline_result" in method._artifacts
    assert "X_train_enhanced" in method._artifacts["pipeline_result"]


def test_malmus_leakage_filtering_and_caching() -> None:
    from feature_forge.config import get_settings

    response_json = {
        "features": [
            {
                "name": "feat_good",
                "code": "df['a'] * 2",
                "description": "good",
                "libraries": ["pandas"],
            },
            {
                "name": "feat_bad",
                "code": "df['b'] * 3",
                "description": "bad",
                "libraries": ["pandas"],
            },
        ]
    }
    llm = FakeLLM(json.dumps(response_json))
    evaluator = MagicMock()
    evaluator.config = get_settings()
    evaluator.evaluate_baseline.return_value = 0.5
    evaluator.evaluate_features_batch.return_value = {"feat_good": 0.1, "feat_bad": 0.0}

    method = MalmusMethod(
        llm_client=llm,
        mode="iterative",
        n_features=1,
        evaluator=evaluator,
    )
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = pd.Series([0, 1])

    method.fit(X, y)
    assert method._kept_features == ["feat_good"]

    X_trans = method.transform(X)
    assert "feat_good" in X_trans.columns
    assert "feat_bad" not in X_trans.columns

    assert "pipeline_result" in method._artifacts
    assert "X_train_enhanced" in method._artifacts["pipeline_result"]
