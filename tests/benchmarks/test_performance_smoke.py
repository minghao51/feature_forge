"""Performance smoke checks for feature evaluation throughput."""

from __future__ import annotations

import time

import pandas as pd

from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator


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
