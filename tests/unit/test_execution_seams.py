"""Tests for case execution and execution backends."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from feature_forge.config import Settings
from feature_forge.contracts import (
    CaseExecutionPlan,
    Layer,
    ManifestRef,
    StageDisposition,
    StageExecution,
)
from feature_forge.experiment.execution import (
    CaseComputationInput,
    ExperimentCase,
    ExperimentResult,
    ProcessPoolExecutionAdapter,
    SequentialExecutionAdapter,
)
from feature_forge.experiment.hamilton_executor import run_hamilton_case


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
    )


def test_execution_backend_semantic_parity_seq_vs_pool() -> None:
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


def test_case_computation_input_is_serializable_shape() -> None:
    payload = CaseComputationInput(
        case=ExperimentCase(dataset="d1", method="m1", model="xgboost", seed=1),
        settings_data=Settings().model_dump(),
    )
    data = asdict(payload)
    assert data["case"]["dataset"] == "d1"
    assert data["settings_data"]["task"] == "classification"
    assert data["dataset_overrides"] is None
    assert data["method_overrides"] is None


def test_run_hamilton_case_reconstructs_settings_and_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process worker rebuilds settings from the serializable payload map."""
    mock_executor = MagicMock()
    mock_executor.plan_case.return_value = CaseExecutionPlan(
        case_key="casekey",
        attempt_id="attempt-1",
        dataset="demo",
        method="dummy",
        model="xgboost",
        unresolved_layers=list(Layer),
    )
    manifest_ref = ManifestRef(layer=Layer.BRONZE, run_id="attempt-1", sha256="a" * 64)
    mock_executor.execute_case.return_value = ExperimentResult(
        dataset="demo",
        method="dummy",
        model="xgboost",
        seed=42,
        cv_score=0.9,
        gain=0.1,
        baseline_score=0.8,
        num_features_generated=1,
        run_id="attempt-1",
        case_fingerprint="casekey",
        stages=[
            StageExecution(
                layer=Layer.BRONZE,
                disposition=StageDisposition.EXECUTED,
                manifest_ref=manifest_ref,
                reuse_fingerprint="reuse",
                layer_fingerprint="layerfp",
            )
        ],
    )
    monkeypatch.setattr(
        "feature_forge.experiment.hamilton_executor.build_worker_local_hamilton_executor",
        lambda settings, payload: mock_executor,
    )
    payload = CaseComputationInput(
        case=ExperimentCase(
            dataset="demo",
            method="dummy",
            model="xgboost",
            seed=42,
            case_key="casekey",
            attempt_id="attempt-1",
        ),
        settings_data=Settings().model_dump(),
    )
    result = run_hamilton_case(payload)

    mock_executor.execute_case.assert_called_once()
    assert result.run_id == "attempt-1"
    assert result.case_fingerprint == "casekey"
    assert len(result.stages) == 1
    assert result.stages[0].layer is Layer.BRONZE
