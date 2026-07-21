"""Characterization tests for the legacy public experiment result contract."""

from __future__ import annotations

from dataclasses import asdict

from feature_forge.experiment.execution import ExperimentResult


def test_legacy_experiment_result_fields_remain_first_class() -> None:
    result = ExperimentResult(
        dataset="demo",
        method="dummy",
        model="rf",
        seed=42,
        cv_score=0.9,
        gain=0.1,
        baseline_score=0.8,
        num_features_generated=2,
        error=None,
    )

    values = asdict(result)
    assert list(values)[:9] == [
        "dataset",
        "method",
        "model",
        "seed",
        "cv_score",
        "gain",
        "baseline_score",
        "num_features_generated",
        "error",
    ]
    assert values["run_id"] is None
    assert values["manifest_uri"] is None
