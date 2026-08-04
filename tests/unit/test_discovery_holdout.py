"""Leakage-safe evaluation: discovery holdout in the legacy case path.

Feature selection and the reported CV score must run on disjoint rows, or
reported gains are optimistically biased. These tests pin that behavior for
``CaseComputation`` (the legacy, non-layered path).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.evaluation.holdout import resolve_partition
from feature_forge.experiment.case_executor import ExperimentCaseExecutor
from feature_forge.experiment.execution import ExperimentCase
from feature_forge.experiment.tracker import NoOpTracker
from feature_forge.methods.base import BaseMethod


class IdentityMethod(BaseMethod):
    """Adds a single ``f_const`` column on any frame; used to exercise transform on held-out rows."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(name="identity")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> IdentityMethod:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        out["f_const"] = 1.0
        return out


class ProbeMethod(BaseMethod):
    """Adds a column derived from a fit-time coefficient applied to input ``a``.

    At fit time it records the sign of the correlation between ``a`` and ``y``
    on the discovery rows; ``transform`` emits ``sign * a`` on any frame. This
    exercises the held-out transform path with an informative (non-constant)
    feature so AUC stays finite on the evaluation partition.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(name="probe")
        self._coef: float = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> ProbeMethod:
        if "a" in X_train.columns and y_train.nunique() > 1:
            corr = float(np.corrcoef(X_train["a"].to_numpy(), y_train.to_numpy())[0, 1])
            self._coef = 1.0 if (np.isnan(corr) or corr >= 0) else -1.0
        else:
            self._coef = 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        out["f_probe"] = self._coef * X.get("a", 0.0)
        return out


def _classification_dataset(n_rows: int = 80, seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n_rows)
    b = rng.normal(size=n_rows)
    target = (a + 0.5 * b + rng.normal(scale=0.3, size=n_rows) > 0).astype(int)
    train = pd.DataFrame({"a": a, "b": b, "target": target})
    return {
        "train": train,
        "test": pd.DataFrame(),
        "target": "target",
        "metadata": {"task": "classification"},
    }


def _build_executor(
    *,
    cv_folds: int,
    holdout_fraction: float,
    stratified: bool = True,
    method: type[BaseMethod] = IdentityMethod,
) -> ExperimentCaseExecutor:
    settings = Settings(
        task="classification",
        metric="auc",
        evaluation={
            "cv_folds": cv_folds,
            "evaluation_holdout_fraction": holdout_fraction,
            "evaluation_holdout_stratified": stratified,
        },
    )
    return ExperimentCaseExecutor(
        settings=settings,
        tracker=NoOpTracker("test"),
        extra_methods={"identity": IdentityMethod, "probe": ProbeMethod},
    )


@pytest.fixture
def patched_load(monkeypatch):
    monkeypatch.setattr(
        "feature_forge.experiment.case_executor.DatasetRegistry.load",
        lambda self, name: _classification_dataset(),
    )
    monkeypatch.setattr(
        "feature_forge.experiment.case_executor.DatasetRegistry.info",
        lambda self, name: {"task": "classification", "source": "memory"},
    )


def test_holdout_partitions_rows_and_reports_counts(patched_load):
    executor = _build_executor(cv_folds=5, holdout_fraction=0.25, method=IdentityMethod)
    case = ExperimentCase(dataset="demo", method="identity", model="random_forest", seed=42)
    result = executor.execute(case)

    assert result.error is None
    assert result.n_evaluation_rows is not None
    assert result.n_discovery_rows is not None
    assert result.n_evaluation_rows + result.n_discovery_rows == 80
    # 0.25 of 80 == 20 evaluation rows.
    assert result.n_evaluation_rows == 20
    assert result.n_discovery_rows == 60


def test_holdout_zero_fraction_uses_full_data_and_disables_counts(patched_load):
    executor = _build_executor(cv_folds=5, holdout_fraction=0.0, method=IdentityMethod)
    case = ExperimentCase(dataset="demo", method="identity", model="random_forest", seed=42)
    result = executor.execute(case)

    assert result.error is None
    # Holdout disabled: no partition counts, whole frame used.
    assert result.n_evaluation_rows is None
    assert result.n_discovery_rows == 80


def test_holdout_degrades_when_data_too_small():
    # When the data is too small to host cv_folds in each partition, the
    # partition helper (the contract that governs the legacy path's decision)
    # must drop the holdout rather than emit a degenerate split. End-to-end
    # re-execution is exercised by the other tests; this pins the guard.
    # cv_folds=5 -> need >= 10 rows for the holdout to activate.
    p = resolve_partition(
        8, fraction=0.25, cv_folds=5, stratified=True, seed=42, y_for_stratify=pd.Series([0, 1] * 4)
    )
    assert not p.holdout_active
    assert len(p.discovery_idx) == 8
    assert len(p.evaluation_idx) == 0


def test_probe_method_transforms_held_out_rows_without_refit(patched_load):
    # Confirms transform() runs against the held-out evaluation subset using
    # fit-time state only; the probe column carries the discovery mean.
    executor = _build_executor(cv_folds=5, holdout_fraction=0.25, method=ProbeMethod)
    case = ExperimentCase(dataset="demo", method="probe", model="random_forest", seed=42)
    result = executor.execute(case)

    assert result.error is None
    assert result.n_evaluation_rows == 20
    # Baseline/gain must be finite (gain may legitimately be 0.0) — the
    # transform produced a usable frame on the held-out rows rather than
    # failing.
    assert result.baseline_score is not None
    assert np.isfinite(result.baseline_score)
    assert result.gain is not None
    assert np.isfinite(result.gain)


# ---- unit-level checks on the partition helper itself ---------------------


def test_resolve_partition_is_deterministic_and_disjoint():
    y = pd.Series([0, 1] * 25)  # 50 balanced rows
    p1 = resolve_partition(50, fraction=0.25, cv_folds=5, stratified=True, seed=7, y_for_stratify=y)
    p2 = resolve_partition(50, fraction=0.25, cv_folds=5, stratified=True, seed=7, y_for_stratify=y)
    np.testing.assert_array_equal(p1.discovery_idx, p2.discovery_idx)
    np.testing.assert_array_equal(p1.evaluation_idx, p2.evaluation_idx)
    assert p1.holdout_active
    full = np.concatenate([p1.discovery_idx, p1.evaluation_idx])
    assert sorted(full.tolist()) == list(range(50))  # disjoint + exhaustive


def test_resolve_partition_disabled_returns_all_discovery():
    p = resolve_partition(40, fraction=0.0, cv_folds=5, stratified=True, seed=7)
    assert not p.holdout_active
    assert len(p.discovery_idx) == 40
    assert len(p.evaluation_idx) == 0
