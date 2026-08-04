"""Tests for case execution and execution backends."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from feature_forge.config import Settings
from feature_forge.experiment.case_executor import (
    CaseComputation,
    CaseComputationInput,
    ExperimentCaseExecutor,
)
from feature_forge.experiment.execution import (
    ExperimentCase,
    ExperimentResult,
    ProcessPoolExecutionAdapter,
    SequentialExecutionAdapter,
)
from feature_forge.experiment.tracker import NoOpTracker
from feature_forge.methods.base import BaseMethod


class DummyMethod(BaseMethod):
    """Simple deterministic method for seam tests."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(name="dummy")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> DummyMethod:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        out["f_dummy"] = 1.0
        return out


class SpyTracker(NoOpTracker):
    """Tracker spy for asserting executor side effects."""

    def __init__(self) -> None:
        super().__init__(project="test")
        self.inits = 0
        self.finishes = 0
        self.metrics_logs = 0

    def init_run(self, run_name: str, config: dict[str, object]) -> None:
        self.inits += 1

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics_logs += 1

    def finish(self) -> None:
        self.finishes += 1


def _sample_dataset() -> dict[str, object]:
    train = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [2, 3, 4, 5, 6, 7],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    return {
        "train": train,
        "test": pd.DataFrame(),
        "target": "target",
        "metadata": {"task": "classification"},
    }


def pool_worker(case: ExperimentCase) -> ExperimentResult:
    """Top-level worker required for process-pool serialization."""
    return ExperimentResult(
        dataset=case.dataset,
        method=case.method,
        model=case.model,
        seed=case.seed,
        cv_score=0.9,
        gain=0.1,
        baseline_score=0.8,
        num_features_generated=1,
        run_id=f"run-{case.seed}",
        state="succeeded",
        manifest_uri=f"04_platinum/runs/run-{case.seed}/manifest.json",
        uncertainty={"lower_bound": 0.01, "upper_bound": 0.19},
        silver_fingerprint="silver",
        gold_fingerprint="gold",
        platinum_fingerprint="platinum",
        directional_gain=0.1,
        selection_profile="recommended",
        metric="auc",
        metric_direction="maximize",
    )


def test_case_executor_success_and_tracker_side_effects(monkeypatch):
    monkeypatch.setattr(
        "feature_forge.experiment.case_executor.DatasetRegistry.load",
        lambda self, name: _sample_dataset(),
    )
    tracker = SpyTracker()
    executor = ExperimentCaseExecutor(
        # The fixture's 6-row frame is too small to host a discovery holdout;
        # this test targets tracker side-effects, not leakage-safe eval.
        settings=Settings(evaluation={"cv_folds": 2, "evaluation_holdout_fraction": 0.0}),
        tracker=tracker,
        extra_methods={"dummy": DummyMethod},
    )
    case = ExperimentCase(dataset="demo", method="dummy", model="random_forest", seed=42)
    result = executor.execute(case)

    assert result.error is None
    assert result.dataset == "demo"
    assert result.method == "dummy"
    assert result.num_candidate_features == 1
    assert result.num_executed_features == 1
    assert result.num_accepted_features == 1
    assert result.num_accepted_output_columns == 1
    assert result.feature_failure_counts == {}
    assert tracker.inits == 1
    assert tracker.finishes == 1
    assert tracker.metrics_logs == 1


def test_case_executor_error_semantics(monkeypatch):
    monkeypatch.setattr(
        "feature_forge.experiment.case_executor.DatasetRegistry.load",
        lambda self, name: {"train": pd.DataFrame({"x": [1, 2]}), "target": None},
    )
    tracker = SpyTracker()
    executor = ExperimentCaseExecutor(
        settings=Settings(evaluation={"cv_folds": 2}),
        tracker=tracker,
        extra_methods={"dummy": DummyMethod},
    )
    case = ExperimentCase(dataset="broken", method="dummy", model="random_forest", seed=7)
    result = executor.execute(case)

    assert result.error is not None
    assert "no target column" in result.error
    assert tracker.inits == 1
    assert tracker.finishes == 1


def test_execution_backend_semantic_parity_seq_vs_pool():
    cases = [
        ExperimentCase(dataset="d1", method="m1", model="xgboost", seed=1),
        ExperimentCase(dataset="d2", method="m2", model="xgboost", seed=2),
    ]
    seq = SequentialExecutionAdapter()
    par = ProcessPoolExecutionAdapter(max_workers=2)

    seq_results = [asdict(r) for r in seq.run(cases, pool_worker, progress=False)]
    par_results = [asdict(r) for r in par.run(cases, pool_worker, progress=False)]

    seq_sorted = sorted(seq_results, key=lambda x: (x["dataset"], x["method"], x["seed"]))
    par_sorted = sorted(par_results, key=lambda x: (x["dataset"], x["method"], x["seed"]))
    assert seq_sorted == par_sorted


def test_case_computation_input_is_serializable_shape():
    payload = CaseComputationInput(
        case=ExperimentCase(dataset="d1", method="m1", model="xgboost", seed=1),
        settings_data=Settings().model_dump(),
    )
    comp = CaseComputation(all_methods={})
    result = comp.compute(payload)
    assert result.error is not None
