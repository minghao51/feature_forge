"""Performance smoke checks for feature evaluation throughput."""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.metrics import MetricDirection
from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.pipeline.core import CorePipeline


class _FakeLLM(LLMClient):
    def __init__(self) -> None:
        super().__init__(model="fake", api_key="fake")

    @property
    def provider_name(self) -> str:
        return "fake"

    def _json_mode_kwargs(self) -> dict:
        return {}

    async def _call_api(self, messages, temperature, max_tokens, **kwargs):
        del messages, temperature, max_tokens, kwargs
        return None

    def _extract_content(self, raw_response):
        del raw_response
        return "{}"

    def _extract_usage(self, raw_response):
        del raw_response
        return 0, 0, 0


class _FastEvaluator:
    # Mirror the CVEvaluator surface that CorePipeline._evaluate_and_select
    # reads (R18 direction-aware selection).
    metric_direction = MetricDirection.MAXIMIZE

    def evaluate_baseline(self, X_train, y_train):
        del X_train, y_train
        return 0.5

    def evaluate_feature(self, X_train, y_train, feature_df, baseline_score):
        del X_train, y_train, baseline_score
        return float(feature_df.iloc[:, 0].mean()) * 1e-6


def test_pyarrow_parquet_engine_imports() -> None:
    """Guard the sandbox IPC happy path.

    The sandbox transports DataFrames between parent and worker via parquet.
    On some CI runners ``pyarrow._parquet.so`` fails to mmap inside a
    ``multiprocessing.spawn`` child, which silently turns every sandbox-using
    test into a timeout. Importing the engine here surfaces the failure as a
    fast, named test error instead of a 9-minute sandbox-timeout cascade.
    """
    import pyarrow.parquet  # noqa: F401 — import side effect is the test

    # Round-trip a tiny frame end-to-end so a partially-broken wheel (engine
    # imports but cannot actually read/write) is caught too.
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as fh:
        path = fh.name
    try:
        frame.to_parquet(path)
        round_tripped = pd.read_parquet(path)
    finally:
        os.unlink(path)
    pd.testing.assert_frame_equal(round_tripped, frame)


def test_feature_eval_smoke_budget() -> None:
    config = Settings(task="classification", metric="auc", evaluation={"cv_folds": 2})
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


def test_explicit_fold_evidence_parity_and_budget() -> None:
    """Reproducible same-fold characterization for legacy versus evidence output."""
    config = Settings(task="classification", metric="auc", evaluation={"cv_folds": 2})
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
    row_ids = pd.Series([f"r{i}" for i in range(rows)])
    assignments = np.full(rows, -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(evaluator._get_cv_splitter(y).split(X, y)):
        assignments[validation] = fold
    folds = pd.DataFrame({"row_id": row_ids, "fold": assignments})

    started = time.perf_counter()
    legacy_score = evaluator.evaluate_baseline(X, y, model_name="random_forest")
    legacy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    evidence = evaluator.evaluate_on_folds(
        X,
        y,
        row_ids,
        folds,
        arm="baseline",
        model_name="random_forest",
    )
    evidence_seconds = time.perf_counter() - started

    assert evidence.fold_metrics["score"].mean() == pytest.approx(legacy_score, abs=1e-12)
    assert len(evidence.predictions) == rows
    assert len(evidence.fold_metrics) == 2
    assert legacy_seconds < 20.0
    assert evidence_seconds < 20.0


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_candidate_eval_backend_smoke_budget(backend: str) -> None:
    config = Settings(
        task="classification",
        metric="auc",
        evaluation={"cv_folds": 2, "feature_eval_backend": backend, "max_candidate_features": 8},
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
