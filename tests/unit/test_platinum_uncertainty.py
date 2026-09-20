"""Unit tests for the Platinum directional Student-t uncertainty (plan 23 PR 4).

Covers ADR 0018 decision 7: paired directional fold deltas with a Student-t
critical value derived from the configured confidence level and
``pair_count - 1`` degrees of freedom (replacing the pre-PR-4 1.96
normal-approximation margin).

Helper style (``_packages``/``_environment``/``_hand_built_aggregate``) is
copied from ``tests/unit/test_platinum_dataflow.py`` and
``tests/unit/test_plan23_evaluation_integrity.py``; it is deliberately
duplicated here so these units stay reviewable independently of those files.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from feature_forge.contracts import (
    BronzeRecord,
    EnvironmentSnapshot,
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
    SilverPackage,
    UncertaintyPolicy,
)
from feature_forge.contracts.identity import gold_input_fingerprint
from feature_forge.dataflows.platinum import (
    _directional_lower_bound,
    _t_critical,
    build_platinum_request,
    uncertainty_summary,
)
from feature_forge.evaluation.metrics import MetricDirection
from feature_forge.exceptions import DatasetError

# scipy.stats.t.ppf(0.975, df=2), the pinned two-sided 95% critical value.
T_STAR_95_DF2 = 4.302652729911275
# scipy.stats.t.ppf(0.975, df=1), the two-sided 95% critical value at df=1.
T_STAR_95_DF1 = 12.706204736174694


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.12",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
    )


def _packages() -> tuple[SilverPackage, GoldPackage, ManifestRef]:
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
    row_ids = [f"row-{index:09d}" for index in range(6)]
    silver = SilverPackage(
        manifest=silver_manifest,
        bronze=BronzeRecord(
            source="fixture",
            source_checksum="source",
            target="target",
            task="classification",
            snapshot_mode="snapshot",
            row_count=6,
            column_count=2,
        ),
        canonical_features=pd.DataFrame({"a": [0, 1, 2, 3, 4, 5]}),
        canonical_target=pd.DataFrame({"row_id": row_ids, "target": [0, 0, 0, 1, 1, 1]}),
        row_ids=pd.DataFrame({"row_id": row_ids}),
        fold_assignments=pd.DataFrame({"row_id": row_ids, "fold": [0, 1, 2, 0, 1, 2]}),
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
    candidate = FeatureCandidate(
        candidate_id="candidate-0",
        name="double_a",
        batch_index=0,
        code_path="generated_code/batch_0000.py",
        specification={},
    )
    provenance = [
        FeatureProvenance(
            candidate_id="candidate-0",
            method="fixture",
            method_version="1",
            prompt_bundle_fingerprint="fixture",
            code_sha256="c" * 64,
        )
    ]
    decisions = [
        FeatureDecision(
            candidate_id="candidate-0",
            state=FeatureDecisionState.ACCEPTED,
            reason_code="accepted",
            reason="fixture",
        )
    ]
    accepted_features = pd.DataFrame({"row_id": row_ids, "double_a": [0, 2, 4, 6, 8, 10]})
    counts = GoldFeatureCounts(
        candidates_proposed=1,
        candidates_executed=1,
        candidates_accepted=1,
        accepted_output_columns=1,
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
        candidates=[candidate],
        provenance=provenance,
        decisions=decisions,
        accepted_features=accepted_features,
        checks=[],
        code={},
        dependencies={},
        counts=counts,
    )
    return silver, gold, gold_ref


def _request(
    silver: SilverPackage,
    gold: GoldPackage,
    gold_ref: ManifestRef,
    *,
    metric: str = "acc",
    uncertainty_policy: UncertaintyPolicy | None = None,
) -> PlatinumRequest:
    """Build a fingerprint-consistent Platinum request (mirrors _io helpers)."""
    return build_platinum_request(
        run_id="platinum-uncertainty-run",
        case_fingerprint="case",
        silver=silver,
        gold=gold,
        gold_ref=gold_ref,
        model_name="random_forest",
        metric=metric,
        seed=42,
        uncertainty_policy=uncertainty_policy,
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


def test_t_critical_matches_pinned_student_t_quantile() -> None:
    assert _t_critical(0.95, 2) == pytest.approx(T_STAR_95_DF2, abs=1e-9)


def test_t_critical_monotone_in_confidence_and_converges_to_normal() -> None:
    for degrees_of_freedom in (2, 5, 30):
        assert _t_critical(0.80, degrees_of_freedom) < _t_critical(0.95, degrees_of_freedom)
        assert _t_critical(0.95, degrees_of_freedom) < _t_critical(0.99, degrees_of_freedom)
    # Many degrees of freedom: the t quantile approaches the normal 1.96.
    assert _t_critical(0.95, 2) > _t_critical(0.95, 10**6)
    assert _t_critical(0.95, 10**6) == pytest.approx(1.959963984540054, abs=1e-4)


def test_uncertainty_summary_is_directional_for_minimizing_metrics() -> None:
    silver, gold, gold_ref = _packages()
    request = _request(silver, gold, gold_ref, metric="rmse")
    assert request.metric_direction is MetricDirection.MINIMIZE
    aggregate = _hand_built_aggregate([-3.0, -1.0, -2.0])  # raw deltas: enhanced improves
    uncertainty = uncertainty_summary(request, aggregate)
    # Directional deltas [3.0, 1.0, 2.0]: every statistic is centered on the
    # directional mean, not the raw negative one.
    assert uncertainty.pair_count == 3
    assert uncertainty.confidence_level == 0.95
    assert uncertainty.mean_directional_gain == pytest.approx(2.0)
    expected_se = 1.0 / np.sqrt(3)
    assert uncertainty.standard_deviation == pytest.approx(1.0)
    assert uncertainty.standard_error == pytest.approx(expected_se)
    assert uncertainty.lower_bound == pytest.approx(2.0 - T_STAR_95_DF2 * expected_se)
    assert uncertainty.upper_bound == pytest.approx(2.0 + T_STAR_95_DF2 * expected_se)
    # With lower-variance gains the whole interval sits above zero on the
    # directional (gain) scale: directional [3.0, 2.5, 2.0], se = 0.5/sqrt(3).
    strong = uncertainty_summary(request, _hand_built_aggregate([-3.0, -2.5, -2.0]))
    assert strong.mean_directional_gain == pytest.approx(2.5)
    assert strong.lower_bound == pytest.approx(2.5 - T_STAR_95_DF2 * (0.5 / np.sqrt(3)))
    assert strong.lower_bound > 0.0
    assert strong.upper_bound > 0.0


def test_uncertainty_summary_responds_to_confidence_level() -> None:
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
    assert tight_margin > loose_margin
    assert loose.confidence_level == 0.8
    assert tight.confidence_level == 0.99


def test_directional_lower_bound_fails_closed_and_applies_t_margin() -> None:
    assert _directional_lower_bound(np.asarray([], dtype=float), 0.95) == float("-inf")
    # A paired t bound needs at least two folds; a single delta fails closed.
    assert _directional_lower_bound(np.asarray([2.5]), 0.95) == float("-inf")
    # df=1: mean 0.25, se exactly 0.25 (std 0.35355... / sqrt(2)); the bound is
    # 0.25 - t(0.975, 1) * 0.25.
    bound = _directional_lower_bound(np.asarray([0.5, 0.0]), 0.95)
    assert bound == pytest.approx(0.25 - T_STAR_95_DF1 * 0.25)
    assert bound == pytest.approx(-2.9265511840436735)


def test_uncertainty_summary_enforces_minimum_successful_folds() -> None:
    silver, gold, gold_ref = _packages()
    request = _request(silver, gold, gold_ref)  # minimum_successful_folds defaults to 2
    with pytest.raises(DatasetError, match="minimum successful-fold"):
        uncertainty_summary(request, _hand_built_aggregate([1.0]))
