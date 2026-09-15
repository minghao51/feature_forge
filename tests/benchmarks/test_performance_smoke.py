"""Performance smoke checks for feature evaluation throughput."""

from __future__ import annotations

import time
from typing import Any, Literal

import pandas as pd
import pytest

from feature_forge.config import EvaluationConfig, Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.pipeline.core import CorePipeline


class _FakeLLM(LLMClient):
    def __init__(self) -> None:
        super().__init__(model="fake", api_key="fake")

    @property
    def provider_name(self) -> str:
        return "fake"

    def _json_mode_kwargs(self) -> dict[str, Any]:
        return {}

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Any:
        del messages, temperature, max_tokens, kwargs
        return None

    def _extract_content(self, raw_response: Any) -> str:
        del raw_response
        return "{}"

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        del raw_response
        return 0, 0, 0


class _FastEvaluator(CVEvaluator):
    def evaluate_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str | None = None,
    ) -> float:
        del X_train, y_train, model_name
        return 0.5

    def evaluate_feature(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_df: pd.DataFrame,
        baseline_score: float | None = None,
        model_name: str | None = None,
    ) -> float:
        del X_train, y_train, baseline_score, model_name
        return float(feature_df.iloc[:, 0].mean()) * 1e-6


def test_feature_eval_smoke_budget() -> None:
    config = Settings(task="classification", metric="auc", evaluation=EvaluationConfig(cv_folds=2))
    evaluator = CVEvaluator(config=config)
    rows = 300
    X = pd.DataFrame(
        {
            "a": [i % 13 for i in range(rows)],
            "b": [i % 7 for i in range(rows)],
            "c": [i % 5 for i in range(rows)],
        }
    )
    y = pd.Series([i % 2 for i in range(rows)])
    baseline = evaluator.evaluate_baseline(X, y, model_name="random_forest")
    feature_df = pd.DataFrame({"candidate": [(i % 13) * (i % 7) for i in range(rows)]})

    start = time.perf_counter()
    gain = evaluator.evaluate_feature(
        X, y, feature_df, baseline_score=baseline, model_name="random_forest"
    )
    elapsed = time.perf_counter() - start

    assert isinstance(gain, float)
    # Smoke budget (lenient for CI variance).
    assert elapsed < 20.0


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_candidate_eval_backend_smoke_budget(backend: Literal["threading", "loky"]) -> None:
    config = Settings(
        task="classification",
        metric="auc",
        evaluation=EvaluationConfig(
            cv_folds=2, feature_eval_backend=backend, max_candidate_features=8
        ),
    )
    pipeline = CorePipeline(config=config, llm_client=_FakeLLM(), evaluator=_FastEvaluator())

    rows = 300
    X_train = pd.DataFrame(
        {
            "a": [i % 17 for i in range(rows)],
            "b": [i % 11 for i in range(rows)],
        }
    )
    y_train = pd.Series([i % 2 for i in range(rows)])
    features = pd.DataFrame(
        {f"f{i}": [((i + 1) * (j % 13)) for j in range(rows)] for i in range(12)}
    )

    start = time.perf_counter()
    result = pipeline._evaluate_and_select(
        features_train=features,
        features_test=pd.DataFrame(index=X_train.index),
        X_train=X_train,
        y_train=y_train,
        X_test=None,
        all_specs=[],
        agents=[],
        code="",
    )
    elapsed = time.perf_counter() - start

    assert result.gains
    assert len(result.gains) <= config.evaluation.max_candidate_features
    assert elapsed < 30.0
