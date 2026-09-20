"""Regression tests encoding findings 2-7 from plan 23 §1 (evaluation integrity).

Mirrors ``tests/unit/test_plan23_sandbox_hardening.py``: desired post-fix
behavior was encoded as xfail-strict tests that demonstrate each confirmed
audit finding on the then-current code.

Findings 2-6 were remediated by PRs 2-3 (ADR 0018): typed protocol settings,
discovery-only method fitting, fail-closed small data, fold-local
preprocessing with the documented unknown-category sentinel, partition-aware
candidate/reporting evidence, greedy selection on discovery folds with every
selection-policy field enforced, and PlatinumRequest partition/profile
validation. Their tests now pass unmarked. The generated-transformation
row-local scope contract (plan 23 §4.2/§7.4) is covered in
``tests/unit/test_scope_contract.py``; fold-local preprocessing units live in
``tests/unit/test_fold_preprocessing.py``; Student-t interval units live in
``tests/unit/test_platinum_uncertainty.py``.

Finding 7 (directional Student-t intervals) was remediated by PR 4 (ADR 0018
decision 7): paired directional fold deltas, the Student-t critical value
from the configured confidence level and ``pair_count - 1`` degrees of
freedom, and a SciPy direct dependency. These tests now pass unmarked.

Helper style (``_packages``/``_environment``) is copied from
``tests/unit/test_platinum_dataflow.py``; it is deliberately duplicated here so
these findings stay reviewable independently of that file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from feature_forge.config import EvaluationConfig, Settings
from feature_forge.contracts import (
    BronzeRecord,
    DatasetComputationRequest,
    EnvironmentSnapshot,
    EvaluationPolicy,
    EvaluationProtocol,
    FeatureCandidate,
    FeatureDecision,
    FeatureDecisionState,
    FeatureProvenance,
    GoldFeatureCounts,
    GoldPackage,
    GoldRequest,
    Layer,
    ManifestRef,
    PlatinumRequest,
    RunManifest,
    RunRequest,
    RunState,
    SelectionPolicy,
    SilverPackage,
    UncertaintyPolicy,
)
from feature_forge.contracts.identity import gold_input_fingerprint, platinum_input_fingerprint
from feature_forge.dataflows import platinum
from feature_forge.dataflows.platinum import (
    aggregate_metrics,
    baseline_fold_evidence,
    build_platinum_request,
    candidate_fold_evidence,
    selection_decisions,
    uncertainty_summary,
)
from feature_forge.dataflows.silver import dataset_fingerprint_value, fold_assignments
from feature_forge.evaluation.holdout import resolve_partition
from feature_forge.evaluation.metrics import MetricDirection
from feature_forge.evaluation.model_factory import ModelRegistry
from feature_forge.exceptions import DatasetError
from feature_forge.experiment.hamilton_executor import (
    _dataset_request,
    _fit_method_on_discovery,
)
from feature_forge.methods import BaseMethod

_FINDING_7_T_REASON = (
    "plan 23 §1 finding 7: uncertainty_summary hard-codes 1.96 instead of the "
    "Student-t critical value for the configured confidence and pair count"
)


class _SpyEstimator:
    """Minimal estimator recording the frames passed to fit/predict."""

    def __init__(self, calls: list[tuple[str, pd.DataFrame]]) -> None:
        self._calls = calls

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _SpyEstimator:
        self._calls.append(("fit", X.copy()))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._calls.append(("predict", X.copy()))
        return np.zeros(len(X))


def _register_spy(name: str) -> list[tuple[str, pd.DataFrame]]:
    """Register a recording estimator under ``name`` and return its call log."""
    calls: list[tuple[str, pd.DataFrame]] = []

    def factory(task: str, random_state: int = 0) -> _SpyEstimator:
        return _SpyEstimator(calls)

    ModelRegistry.register(name, factory)
    return calls


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.12",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
    )


def _packages(
    *,
    features: pd.DataFrame | None = None,
    target: Sequence[float] = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    folds: Sequence[int] = (0, 1, 2, 0, 1, 2),
    partitions: Sequence[str] | None = None,
    candidates: Mapping[str, Sequence[float]] | None = None,
    task: str = "classification",
) -> tuple[SilverPackage, GoldPackage, ManifestRef]:
    """Build a verified Silver/Gold fixture pair (style of test_platinum_dataflow)."""
    candidate_features = dict(candidates or {})
    now = datetime.now(UTC)
    silver_ref = ManifestRef(layer=Layer.SILVER, run_id="silver-run", sha256="a" * 64)
    gold_ref = ManifestRef(layer=Layer.GOLD, run_id="gold-run", sha256="b" * 64)
    silver_manifest = RunManifest(
        layer=Layer.SILVER,
        package_kind="silver",
        layer_fingerprint="silver-fingerprint",
        run_id="silver-run",
        case_fingerprint="case",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="fixture", method="silver", model="none", seed=42),
        environment=_environment(),
        created_at=now,
        completed_at=now,
    )
    n_rows = len(target)
    row_ids = [f"row-{index:09d}" for index in range(n_rows)]
    feature_frame = (
        pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        if features is None
        else features.reset_index(drop=True)
    )
    fold_frame = pd.DataFrame({"row_id": row_ids, "fold": list(folds)})
    if partitions is not None:
        fold_frame["partition"] = list(partitions)
    silver = SilverPackage(
        manifest=silver_manifest,
        bronze=BronzeRecord(
            source="fixture",
            source_checksum="source",
            target="target",
            task=task,
            snapshot_mode="snapshot",
            row_count=n_rows,
            column_count=feature_frame.shape[1] + 1,
        ),
        canonical_features=feature_frame,
        canonical_target=pd.DataFrame({"row_id": row_ids, "target": list(target)}),
        row_ids=pd.DataFrame({"row_id": row_ids}),
        fold_assignments=fold_frame,
        profile={},
        checks=[],
    )
    gold_input = gold_input_fingerprint(
        silver_fingerprint_value="silver-fingerprint",
        method_name="fixture",
        method_version="1",
        method_config={},
        prompt_bundle_fingerprint="fixture",
        generated_contract_version="1",
        selection_policy={"policy": "validation"},
    )
    gold_request = GoldRequest(
        run_id="gold-run",
        case_fingerprint="case",
        silver_manifest=silver_ref,
        silver_fingerprint="silver-fingerprint",
        gold_input_fingerprint=gold_input,
        method_name="fixture",
        method_version="1",
        prompt_bundle_fingerprint="fixture",
    )
    names = list(candidate_features)
    candidate_models = [
        FeatureCandidate(
            candidate_id=f"candidate-{index}",
            name=name,
            batch_index=index,
            code_path=f"generated_code/batch_{index:04d}.py",
            specification={},
        )
        for index, name in enumerate(names)
    ]
    provenance = [
        FeatureProvenance(
            candidate_id=f"candidate-{index}",
            method="fixture",
            method_version="1",
            prompt_bundle_fingerprint="fixture",
            code_sha256="c" * 64,
        )
        for index in range(len(names))
    ]
    decisions = [
        FeatureDecision(
            candidate_id=f"candidate-{index}",
            state=FeatureDecisionState.ACCEPTED,
            reason_code="accepted",
            reason="fixture",
        )
        for index in range(len(names))
    ]
    accepted_features = pd.DataFrame(
        {"row_id": row_ids, **{name: list(values) for name, values in candidate_features.items()}}
    )
    counts = GoldFeatureCounts(
        candidates_proposed=len(names),
        candidates_executed=len(names),
        candidates_accepted=len(names),
        accepted_output_columns=len(names),
    )
    gold_manifest = RunManifest(
        layer=Layer.GOLD,
        package_kind="gold",
        layer_fingerprint="gold-fingerprint",
        run_id="gold-run",
        case_fingerprint="case",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="silver", method="fixture", model="none", seed=42),
        environment=_environment(),
        upstream_manifests=[silver_ref],
        created_at=now,
        completed_at=now,
    )
    gold = GoldPackage(
        manifest=gold_manifest,
        request=gold_request,
        candidates=candidate_models,
        provenance=provenance,
        decisions=decisions,
        accepted_features=accepted_features,
        checks=[],
        code={},
        dependencies={},
        counts=counts,
    )
    return silver, gold, gold_ref


def _complementary_packages() -> tuple[SilverPackage, GoldPackage, ManifestRef]:
    """Silver/Gold pair where x1 and x2 each improve alone and improve jointly.

    Rows are three balanced folds; ``target == x1 + x2`` with ``x1`` and ``x2``
    mirrored so both single-candidate arms yield identical mean gains. The
    random-forest regressor interpolates per-group means, so single-candidate
    gains are strictly positive and the combined set is strictly better.
    """
    x1 = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    x2 = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    target = [float(a + b) for a, b in zip(x1, x2, strict=True)]
    return _packages(
        features=pd.DataFrame({"a": [5.0] * 9}),
        target=target,
        # Partitions and folds are pinned so both scopes host two folds whose
        # validation rows are non-degenerate for R^2 (>= 2 rows, non-constant
        # target) — same design as test_platinum_evidence_v2 after ADR 0018
        # decision 3/6 made partition-less Silver fail closed.
        folds=(0, 0, 1, 0, 0, 1, 1, 1, 1),
        partitions=(
            "discovery",
            "discovery",
            "discovery",
            "evaluation",
            "evaluation",
            "discovery",
            "discovery",
            "evaluation",
            "evaluation",
        ),
        candidates={"x1": x1, "x2": x2},
        task="regression",
    )


def _request(
    silver: SilverPackage,
    gold: GoldPackage,
    gold_ref: ManifestRef,
    *,
    metric: str = "acc",
    model_name: str = "random_forest",
    seed: int = 42,
    evaluation_policy: EvaluationPolicy | None = None,
    uncertainty_policy: UncertaintyPolicy | None = None,
    selection_policy: SelectionPolicy | None = None,
) -> PlatinumRequest:
    """Build a fingerprint-consistent Platinum request (mirrors _io helpers)."""
    return build_platinum_request(
        run_id="plan23-run",
        case_fingerprint="case",
        silver=silver,
        gold=gold,
        gold_ref=gold_ref,
        model_name=model_name,
        metric=metric,
        seed=seed,
        evaluation_policy=evaluation_policy,
        uncertainty_policy=uncertainty_policy,
        selection_policy=selection_policy,
    )


def _hand_built_aggregate(fold_deltas: list[float]) -> dict[str, Any]:
    """Aggregate dict with the keys platinum.py's downstream nodes consume."""
    mean = float(np.mean(fold_deltas))
    return {
        "baseline_score": 0.5,
        "enhanced_score": 0.5 + mean,
        "legacy_gain": mean,
        "directional_gain": mean,
        "fold_deltas": fold_deltas,
        "candidates": {"signal": {"legacy_gain": mean, "directional_gain": mean}},
        "best": "signal",
    }


# ─────────────────────────────────────────────────────────────
# Finding 2 (High) — remediated by PR 2 (partition-aware discovery, ADR 0018):
# typed protocol/fraction settings, executor threading, fail-closed small data,
# discovery-only method fitting, and protocol-aware reuse fingerprints.
# ─────────────────────────────────────────────────────────────


def test_settings_expose_evaluation_protocol() -> None:
    evaluation = Settings().evaluation
    assert evaluation.protocol == "holdout"
    assert evaluation.evaluation_holdout_fraction == 0.25


def test_dataset_request_rejects_incoherent_protocol_combinations() -> None:
    base: dict[str, Any] = {
        "name": "fixture",
        "target": "target",
        "task": "regression",
        "split_seed": 0,
        "cv_folds": 2,
    }
    with pytest.raises(ValidationError, match="requires"):
        DatasetComputationRequest(
            **base, evaluation_protocol="holdout", evaluation_holdout_fraction=0.0
        )
    with pytest.raises(ValidationError, match=r"must be 0\.0"):
        DatasetComputationRequest(
            **base, evaluation_protocol="compatibility", evaluation_holdout_fraction=0.25
        )


def test_executor_threads_configured_evaluation_protocol() -> None:
    """Behavioral replacement for the PR-1 source pin (plan 23 §6 PR 2)."""
    holdout = _dataset_request(
        Settings(evaluation=EvaluationConfig(protocol="holdout", evaluation_holdout_fraction=0.3)),
        dataset="fixture",
        target="target",
        task="classification",
        run_id="run",
        case_fingerprint="case",
        split_seed=42,
        cv_folds=3,
    )
    assert holdout.evaluation_protocol == "holdout"
    assert holdout.evaluation_holdout_fraction == 0.3
    compatibility = _dataset_request(
        Settings(evaluation=EvaluationConfig(protocol="compatibility")),
        dataset="fixture",
        target="target",
        task="classification",
        run_id="run",
        case_fingerprint="case",
        split_seed=42,
        cv_folds=3,
    )
    assert compatibility.evaluation_protocol == "compatibility"
    assert compatibility.evaluation_holdout_fraction == 0.0


def test_holdout_fails_closed_for_small_datasets() -> None:
    with pytest.raises(DatasetError):
        resolve_partition(n_rows=7, fraction=0.25, cv_folds=5, stratified=False, seed=0)


def _silver_folds(
    n_rows: int,
    *,
    evaluation_protocol: EvaluationProtocol,
    evaluation_holdout_fraction: float,
    cv_folds: int = 5,
) -> pd.DataFrame:
    features = pd.DataFrame({"a": [float(value) for value in range(n_rows)]})
    target = pd.Series([float(value % 2) for value in range(n_rows)])
    row_ids = [f"row-{index:09d}" for index in range(n_rows)]
    request = DatasetComputationRequest(
        name="fixture",
        target="target",
        task="regression",
        split_seed=0,
        cv_folds=cv_folds,
        evaluation_protocol=evaluation_protocol,
        evaluation_holdout_fraction=evaluation_holdout_fraction,
    )
    return fold_assignments(features, target, row_ids, request)


def test_small_data_fails_closed_under_holdout_and_succeeds_in_compatibility() -> None:
    """Plan 23 §7.5: holdout never degrades silently; compatibility is explicit."""
    with pytest.raises(DatasetError):
        _silver_folds(7, evaluation_protocol="holdout", evaluation_holdout_fraction=0.25)
    folds = _silver_folds(7, evaluation_protocol="compatibility", evaluation_holdout_fraction=0.0)
    assert (folds["partition"] == "discovery").all()
    assert folds["fold"].nunique() == 5


class _SpyMethod(BaseMethod):
    """Method whose fit records exactly the frames and targets it receives."""

    def __init__(self) -> None:
        super().__init__(name="plan23_spy")
        self.fit_calls: list[tuple[pd.DataFrame, pd.Series]] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> _SpyMethod:
        self.fit_calls.append((X_train.copy(), y_train.copy()))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()


def test_method_fit_receives_discovery_rows_and_targets_only() -> None:
    """Plan 23 §7.1: evaluation rows and targets never reach method fitting."""
    silver, _, _ = _partitioned_packages()
    spy = _SpyMethod()
    _fit_method_on_discovery(spy, silver)
    assert len(spy.fit_calls) == 1
    features, target = spy.fit_calls[0]
    # Exactly the 8 discovery rows in Silver order; the 4 evaluation rows
    # (a == 8..11) and their targets never reach the method (ADR 0018 §3).
    assert features["a"].tolist() == [float(value) for value in range(8)]
    assert target.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_partition_provenance_invalidates_reuse_fingerprints() -> None:
    """Plan 23 PR 2: Silver/Gold/Platinum identities distinguish protocols."""
    bronze = BronzeRecord(
        source="fixture",
        source_checksum="source",
        target="target",
        task="regression",
        snapshot_mode="snapshot",
        row_count=12,
        column_count=2,
    )
    common: dict[str, Any] = {
        "name": "fixture",
        "target": "target",
        "task": "regression",
        "split_seed": 0,
        "cv_folds": 2,
    }
    holdout = DatasetComputationRequest(
        **common, evaluation_protocol="holdout", evaluation_holdout_fraction=0.25
    )
    compatibility = DatasetComputationRequest(
        **common, evaluation_protocol="compatibility", evaluation_holdout_fraction=0.0
    )
    assert dataset_fingerprint_value(bronze, holdout) != dataset_fingerprint_value(
        bronze, compatibility
    )
    gold_common: dict[str, Any] = {
        "silver_fingerprint_value": "silver-fingerprint",
        "method_name": "fixture",
        "method_version": "1",
        "method_config": {},
        "prompt_bundle_fingerprint": "fixture",
        "generated_contract_version": "1",
        "selection_policy": {"policy": "validation"},
    }
    assert gold_input_fingerprint(
        **gold_common, evaluation_protocol="holdout"
    ) != gold_input_fingerprint(**gold_common, evaluation_protocol="compatibility")
    platinum_common: dict[str, Any] = {
        "gold_fingerprint_value": "gold-fingerprint",
        "model_name": "random_forest",
        "model_version": "1",
        "model_config": {},
        "metric": "acc",
        "fold_fingerprint": "folds",
        "uncertainty_policy": UncertaintyPolicy().model_dump(mode="json"),
    }
    assert platinum_input_fingerprint(
        **platinum_common,
        evaluation_policy=EvaluationPolicy(evaluation_protocol="holdout").model_dump(mode="json"),
    ) != platinum_input_fingerprint(
        **platinum_common,
        evaluation_policy=EvaluationPolicy(evaluation_protocol="compatibility").model_dump(
            mode="json"
        ),
    )


# ─────────────────────────────────────────────────────────────
# Finding 3 (High) — remediated by PR 3 (fold-local preprocessing,
# ADR 0018 decision 4): per fold, numeric imputation and categorical
# encoding are fitted on training rows only; validation rows are
# transformed with train-only statistics and unseen categories map to the
# documented stable sentinel -1.
# ─────────────────────────────────────────────────────────────


def test_preprocessing_uses_training_fold_statistics() -> None:
    calls = _register_spy("plan23_spy_numeric")
    features = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, float("nan")]})
    silver, gold, gold_ref = _packages(
        features=features,
        target=(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0),
        folds=(1, 1, 1, 1, 0, 0, 0),  # fold 0 validates rows 4-6
        # Row 3 is the discovery partition; both folds stay in the evaluation
        # partition so the fold-local preprocessing pins keep their row-index
        # semantics (only row 3 leaves the evaluated frame).
        partitions=(
            "evaluation",
            "evaluation",
            "evaluation",
            "discovery",
            "evaluation",
            "evaluation",
            "evaluation",
        ),
    )
    request = _request(silver, gold, gold_ref, metric="acc", model_name="plan23_spy_numeric")
    baseline_fold_evidence(request, silver)
    fits = [frame for kind, frame in calls if kind == "fit"]
    predicts = [frame for kind, frame in calls if kind == "predict"]
    assert [frame.index.tolist() for frame in fits] == [[0, 1, 2], [4, 5, 6]]
    assert [frame.index.tolist() for frame in predicts] == [[4, 5, 6], [0, 1, 2]]
    # Fold 0's training rows are [0, 1, 2] (median 2.0); its validation NaN
    # must be transformed with those train-only statistics, not the whole-frame
    # median 3.5 that leaks today.
    assert predicts[0].loc[6, "a"] == 2.0
    # Fold 1 fits on rows 4-6 (containing the NaN); its own train-only median
    # (150) must fill it, not the whole-frame median 3.5.
    assert fits[1].loc[6, "a"] == 150.0


def test_unseen_validation_category_follows_documented_policy() -> None:
    calls = _register_spy("plan23_spy_categorical")
    features = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "cat": ["a", "b", "a", "b", "a", "zzz", "zzz"],
        }
    )
    silver, gold, gold_ref = _packages(
        features=features,
        target=(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0),
        folds=(1, 1, 1, 1, 0, 0, 0),  # fold 0 validates rows 4-6
        partitions=(
            "evaluation",
            "evaluation",
            "evaluation",
            "discovery",
            "evaluation",
            "evaluation",
            "evaluation",
        ),
    )
    request = _request(silver, gold, gold_ref, metric="acc", model_name="plan23_spy_categorical")
    baseline_fold_evidence(request, silver)
    predicts = [frame for kind, frame in calls if kind == "predict"]
    assert predicts[0].index.tolist() == [4, 5, 6]
    # 'zzz' never appears in fold 0's training rows; the documented policy maps
    # unknown categories to the stable sentinel -1, not a whole-frame code (2).
    assert predicts[0].loc[5, "cat"] == -1


# ─────────────────────────────────────────────────────────────
# Finding 4 (High) — remediated by PR 3 (partition-aware evidence,
# ADR 0018 decisions 3/6): candidate/selection evidence covers discovery
# folds only; the reported arms are evaluated on evaluation folds only.
# ─────────────────────────────────────────────────────────────


def _partitioned_packages() -> tuple[SilverPackage, GoldPackage, ManifestRef]:
    return _packages(
        features=pd.DataFrame({"a": [float(value) for value in range(12)]}),
        target=(0.0, 1.0) * 6,
        folds=(0, 1) * 6,
        partitions=("discovery",) * 8 + ("evaluation",) * 4,
        candidates={"signal": [float(value) for value in range(12)]},
    )


def test_candidate_evidence_excludes_evaluation_partition() -> None:
    silver, gold, gold_ref = _partitioned_packages()
    request = _request(silver, gold, gold_ref, metric="acc")
    candidate_evidence = candidate_fold_evidence(request, silver, gold)
    metrics = candidate_evidence["signal"]["metrics"]
    # Selection evidence must cover the 8 discovery rows only (both partitions
    # host 2 folds; today all 12 rows are scored).
    assert int(metrics["n_validation"].sum()) == 8


def test_reported_enhanced_uses_evaluation_partition_only() -> None:
    silver, gold, gold_ref = _partitioned_packages()
    request = _request(silver, gold, gold_ref, metric="acc")
    baseline = baseline_fold_evidence(request, silver)
    candidates = candidate_fold_evidence(request, silver, gold)
    aggregate = aggregate_metrics(request, baseline, candidates)
    # The reported arms must come from the 4 evaluation rows (2 folds), not all
    # 12 rows / 4 fold scores as today.
    assert int(baseline["metrics"]["n_validation"].sum()) == 4
    assert len(aggregate["fold_deltas"]) == 2


# ─────────────────────────────────────────────────────────────
# Finding 5 (High) — remediated by PR 3 (ADR 0018 decision 6): the reported
# enhanced arm is the frozen selected set on evaluation folds; a
# policy-rejected winner mirrors baseline with zero gain.
# ─────────────────────────────────────────────────────────────


def test_rejected_winner_mirrors_baseline() -> None:
    silver, gold, gold_ref = _packages(
        features=pd.DataFrame({"a": [1.0] * 6}),  # constant: baseline cannot learn
        target=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        candidates={"signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},  # perfectly separates
        # Interleaved folds x partitions so each scope hosts two folds (rows
        # 0,1,3,4 discovery / 2,5 evaluation, both with folds 0 and 1).
        folds=(0, 1, 0, 1, 0, 1),
        partitions=("discovery", "discovery", "evaluation", "discovery", "discovery", "evaluation"),
    )
    request = _request(
        silver,
        gold,
        gold_ref,
        metric="acc",
        selection_policy=SelectionPolicy(minimum_practical_gain=1e9),
    )
    baseline = baseline_fold_evidence(request, silver)
    candidates = candidate_fold_evidence(request, silver, gold)
    aggregate = aggregate_metrics(request, baseline, candidates)
    uncertainty = uncertainty_summary(request, aggregate)
    decisions = selection_decisions(request, gold, aggregate, uncertainty)
    # The candidate is numerically best but policy-rejected, so the reported
    # enhanced arm must mirror baseline and the decision must be a rejection.
    assert aggregate["enhanced_score"] == aggregate["baseline_score"]
    assert aggregate["directional_gain"] == 0.0
    assert all(not decision.selected for decision in decisions)


# ─────────────────────────────────────────────────────────────
# Finding 6 (Medium) — remediated by PR 3 (ADR 0018 decision 5):
# selection_partition/profile validation at request construction, greedy
# forward selection on discovery folds enforcing minimum_practical_gain,
# require_positive_lower_bound, and max_selected_features with a stable
# feature-name tie-break.
# ─────────────────────────────────────────────────────────────


def test_selection_partition_validation() -> None:
    silver, gold, gold_ref = _packages()
    with pytest.raises(ValidationError):
        _request(
            silver,
            gold,
            gold_ref,
            evaluation_policy=EvaluationPolicy(selection_partition="evaluation"),
        )


def test_require_positive_lower_bound_enforced() -> None:
    silver, gold, gold_ref = _packages(
        candidates={"signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
    )
    request = _request(
        silver,
        gold,
        gold_ref,
        selection_policy=SelectionPolicy(require_positive_lower_bound=True),
    )
    aggregate = _hand_built_aggregate([0.5, 0.0])  # mean gain 0.25, high variance
    uncertainty = uncertainty_summary(request, aggregate)
    # Guard: the 2-fold delta interval genuinely dips below zero.
    assert uncertainty.lower_bound < 0.0
    decisions = selection_decisions(request, gold, aggregate, uncertainty)
    # Positive mean gain with a negative lower bound must be rejected.
    assert all(not decision.selected for decision in decisions)


def test_greedy_selection_combines_complementary_features() -> None:
    silver, gold, gold_ref = _complementary_packages()
    request = _request(silver, gold, gold_ref, metric="r2")
    baseline = baseline_fold_evidence(request, silver)
    candidates = candidate_fold_evidence(request, silver, gold)
    aggregate = aggregate_metrics(request, baseline, candidates)
    uncertainty = uncertainty_summary(request, aggregate)
    decisions = selection_decisions(request, gold, aggregate, uncertainty)
    # Guard: each candidate improves alone under the default registered model.
    assert aggregate["candidates"]["x1"]["directional_gain"] > 0.0
    assert aggregate["candidates"]["x2"]["directional_gain"] > 0.0
    # Greedy forward selection must keep both complementary features.
    assert {item.feature_name for item in decisions if item.selected} == {"x1", "x2"}


def test_max_selected_features_caps_selection() -> None:
    """Greedy selection stops at ``max_selected_features`` (plan 23 §4.4).

    Extended from the PR-1 coincidence pin (cap=1 with single-best
    selection): the cap now bounds real greedy forward selection — cap=1
    keeps exactly the first greedy pick, the default (no cap) and cap=2
    admit the complementary pair, and repeated runs are deterministic.
    """
    silver, gold, gold_ref = _complementary_packages()

    def _selected(cap: int | None) -> list[str]:
        request = _request(
            silver,
            gold,
            gold_ref,
            metric="r2",
            selection_policy=SelectionPolicy(max_selected_features=cap),
        )
        baseline = baseline_fold_evidence(request, silver)
        candidates = candidate_fold_evidence(request, silver, gold)
        aggregate = aggregate_metrics(request, baseline, candidates)
        uncertainty = uncertainty_summary(request, aggregate)
        decisions = selection_decisions(request, gold, aggregate, uncertainty)
        return [item.feature_name for item in decisions if item.selected]

    capped = _selected(1)
    assert len(capped) == 1
    assert capped[0] in {"x1", "x2"}
    assert _selected(1) == capped  # deterministic across runs
    assert set(_selected(2)) == {"x1", "x2"}
    assert set(_selected(None)) == {"x1", "x2"}  # uncapped greedy keeps both


def test_selection_tie_break_is_stable_by_feature_name() -> None:
    """Equal directional gains resolve by feature name (plan 23 §4.4/§7.6)."""
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # "z_b" is declared first so Gold column order cannot influence the pick;
    # the identical columns make the greedy gains an exact tie (candidate names
    # deliberately avoid the canonical feature column "a").
    silver, gold, gold_ref = _packages(
        candidates={"z_b": values, "z_a": values},
        folds=(0, 1, 0, 1, 0, 1),
        partitions=("discovery", "discovery", "evaluation", "discovery", "discovery", "evaluation"),
    )
    request = _request(
        silver,
        gold,
        gold_ref,
        metric="acc",
        selection_policy=SelectionPolicy(max_selected_features=1),
    )
    baseline = baseline_fold_evidence(request, silver)
    candidates = candidate_fold_evidence(request, silver, gold)
    aggregate = aggregate_metrics(request, baseline, candidates)
    uncertainty = uncertainty_summary(request, aggregate)
    decisions = selection_decisions(request, gold, aggregate, uncertainty)
    selected = [item.feature_name for item in decisions if item.selected]
    assert selected == ["z_a"]  # lexicographically smallest wins the tie


def test_evaluation_scores_unreadable_until_selection_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan 23 §7.2: evaluation-fold scores are unavailable until freeze.

    Spies on the fold-evaluation engine and the greedy selection routine:
    every evaluation made while selection is running must be scoped to the
    discovery partition (reading an evaluation-fold score during selection
    trips the spy), and evaluation-partition scores may only be read after
    the selection freezes.
    """
    silver, gold, gold_ref = _partitioned_packages()
    request = _request(silver, gold, gold_ref, metric="acc")
    events: list[tuple[str, str]] = []
    selecting = {"active": False}
    real_evaluate = platinum._evaluate_frame
    real_greedy = platinum._greedy_forward_selection

    def spy_evaluate(
        request: PlatinumRequest,
        frame: pd.DataFrame,
        silver: SilverPackage,
        arm: str,
        *,
        partition: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        assert not (selecting["active"] and partition != "discovery"), (
            f"selection read {partition}-fold scores before freezing"
        )
        events.append(("evaluate", partition))
        return real_evaluate(request, frame, silver, arm, partition=partition)  # type: ignore[arg-type]

    def spy_greedy(*args: Any, **kwargs: Any) -> Any:
        selecting["active"] = True
        try:
            return real_greedy(*args, **kwargs)
        finally:
            selecting["active"] = False
            events.append(("freeze", "-"))

    monkeypatch.setattr(platinum, "_evaluate_frame", spy_evaluate)
    monkeypatch.setattr(platinum, "_greedy_forward_selection", spy_greedy)
    evidence = candidate_fold_evidence(request, silver, gold)

    freeze_at = next(index for index, event in enumerate(events) if event[0] == "freeze")
    before = [partition for kind, partition in events[:freeze_at] if kind == "evaluate"]
    after = [partition for kind, partition in events[freeze_at:] if kind == "evaluate"]
    # Selection-phase evidence is discovery-scoped only (per-candidate arms,
    # the discovery baseline, and every greedy step), and the frozen combined
    # arm is the only evaluation-partition read — strictly after the freeze.
    assert before
    assert set(before) == {"discovery"}
    assert after == ["evaluation"]
    assert evidence.selection.selected  # a non-empty set actually froze
    assert evidence.selected_arm is not None


# ─────────────────────────────────────────────────────────────
# Finding 7 (Medium) — remediated by PR 4 (ADR 0018 decision 7): paired
# directional deltas, Student-t critical value from the configured confidence
# level and pair_count - 1 degrees of freedom (scipy direct dependency).
# ─────────────────────────────────────────────────────────────


def test_interval_uses_student_t() -> None:
    silver, gold, gold_ref = _packages()
    request = _request(silver, gold, gold_ref)  # default confidence 0.95
    aggregate = _hand_built_aggregate([3.0, 1.0, 2.0])
    uncertainty = uncertainty_summary(request, aggregate)
    t_star = 4.302652729911275  # scipy.stats.t.ppf(0.975, df=2)
    ratio = (uncertainty.upper_bound - uncertainty.lower_bound) / (2 * uncertainty.standard_error)
    assert abs(ratio - t_star) < 1e-6


def test_interval_responds_to_confidence_level() -> None:
    silver, gold, gold_ref = _packages()
    aggregate = _hand_built_aggregate([3.0, 1.0, 2.0])
    loose = uncertainty_summary(
        _request(
            silver, gold, gold_ref, uncertainty_policy=UncertaintyPolicy(confidence_level=0.8)
        ),
        aggregate,
    )
    tight = uncertainty_summary(
        _request(
            silver, gold, gold_ref, uncertainty_policy=UncertaintyPolicy(confidence_level=0.99)
        ),
        aggregate,
    )
    loose_margin = loose.upper_bound - loose.lower_bound
    tight_margin = tight.upper_bound - tight.lower_bound
    assert loose_margin != tight_margin


def test_interval_is_directional_for_minimizing_metrics() -> None:
    silver, gold, gold_ref = _packages()
    request = _request(silver, gold, gold_ref, metric="rmse")
    assert request.metric_direction is MetricDirection.MINIMIZE
    # Raw deltas: enhanced improves on every fold (negative RMSE deltas).
    # Dispersion is kept small because the PR-4 Student-t margin at df=2
    # (t* = 4.302653) is much wider than the 1.96 normal approximation the
    # PR-1 fixture assumed; the original [-3.0, -1.0, -2.0] deltas leave the
    # lower bound below zero even though the directional mean is positive.
    aggregate = _hand_built_aggregate([-3.0, -2.5, -2.0])
    uncertainty = uncertainty_summary(request, aggregate)
    assert uncertainty.mean_directional_gain == pytest.approx(2.5)
    # Bounds must be built on the directional (positive-mean) deltas, so the
    # whole interval sits above zero instead of describing raw RMSE deltas.
    assert uncertainty.lower_bound > 0.0
    assert uncertainty.upper_bound > 0.0
