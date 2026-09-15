"""Offline Platinum fold, aggregation, and selection coverage."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from feature_forge.contracts import (
    AggregateMetric,
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
    PlatinumMaterialization,
    PlatinumRequest,
    RunManifest,
    RunRequest,
    RunState,
    SilverPackage,
)
from feature_forge.contracts.identity import gold_input_fingerprint
from feature_forge.dataflows.platinum import (
    aggregate_metrics,
    baseline_fold_evidence,
    build_platinum_request,
    candidate_fold_evidence,
    load_platinum_package,
    platinum_checks,
    platinum_manifest,
    platinum_materialization,
    platinum_request,
    selection_decisions,
    uncertainty_summary,
)
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.storage.local import LocalArtifactStore


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.12",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
    )


def _packages(*, with_candidate: bool = True) -> tuple[SilverPackage, GoldPackage, ManifestRef]:
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
    features = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
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
        canonical_features=features,
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
    if with_candidate:
        candidates = [candidate]
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
    else:
        candidates, provenance, decisions = [], [], []
        accepted_features = pd.DataFrame({"row_id": row_ids})
        counts = GoldFeatureCounts(
            candidates_proposed=0,
            candidates_executed=0,
            candidates_accepted=0,
            accepted_output_columns=0,
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
        candidates=candidates,
        provenance=provenance,
        decisions=decisions,
        accepted_features=accepted_features,
        checks=[],
        code={},
        dependencies={},
        counts=counts,
    )
    return silver, gold, gold_ref


def _node_tags(function: object) -> dict[str, str]:
    tags: dict[str, str] = {}
    for decorator in getattr(function, "decorate_nodes", []):
        tags.update(getattr(decorator, "tags", {}))
        tags.update(getattr(decorator, "cache_tags", {}))
    return tags


def test_per_attempt_manifest_node_is_marked_recompute() -> None:
    assert _node_tags(platinum_manifest)["persistence"] == "recompute"
    assert _node_tags(platinum_manifest).get("cache.behavior") == "recompute"


def _materialized_package(
    tmp_path: Any, *, with_candidate: bool
) -> tuple[LocalArtifactStore, PlatinumMaterialization, dict[str, Any], PlatinumRequest]:
    """Run the full Platinum node chain and persist the package under ``tmp_path``."""
    silver, gold, gold_ref = _packages(with_candidate=with_candidate)
    request = build_platinum_request(
        run_id="platinum-run",
        case_fingerprint="case",
        silver=silver,
        gold=gold,
        gold_ref=gold_ref,
        model_name="random_forest",
        metric="acc",
        seed=42,
    )
    request = platinum_request(request)
    baseline = baseline_fold_evidence(request, silver)
    candidates = candidate_fold_evidence(request, silver, gold)
    aggregate = aggregate_metrics(request, baseline, candidates)
    uncertainty = uncertainty_summary(request, aggregate)
    decisions = selection_decisions(request, gold, aggregate, uncertainty)
    checks = platinum_checks(request, silver, baseline, aggregate)
    manifest = platinum_manifest(request, _environment(), checks, aggregate, uncertainty, decisions)
    store = LocalArtifactStore(tmp_path / "lake")
    materialization = platinum_materialization(
        manifest,
        request,
        silver,
        baseline,
        candidates,
        aggregate,
        uncertainty,
        decisions,
        checks,
        ExecutionProfile.DEVELOPMENT,
        store,
    )
    return store, materialization, aggregate, request


def test_platinum_persisted_evidence_reconstructs_candidate_run(tmp_path: Path) -> None:
    store, materialization, aggregate, request = _materialized_package(
        tmp_path, with_candidate=True
    )

    assert materialization.persisted is True
    assert materialization.manifest_ref is not None
    package = load_platinum_package(store, materialization.manifest_ref)
    assert set(package.fold_metrics["arm"].unique()) == {"baseline", "enhanced"}
    enhanced = package.fold_metrics.loc[package.fold_metrics["arm"] == "enhanced"]
    assert math.isclose(
        float(enhanced["score"].mean()),
        package.aggregate.enhanced_score,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert package.aggregate == AggregateMetric(
        metric=request.metric,
        metric_direction=request.metric_direction,
        aggregation="unweighted_mean_of_fold_metrics",
        baseline_score=aggregate["baseline_score"],
        enhanced_score=aggregate["enhanced_score"],
        legacy_gain=aggregate["legacy_gain"],
        directional_gain=aggregate["directional_gain"],
        successful_folds=len(aggregate["fold_deltas"]),
    )
    assert [item.feature_name for item in package.decisions if item.selected] == ["double_a"]


def test_platinum_persisted_evidence_reconstructs_no_candidate_run(tmp_path: Path) -> None:
    store, materialization, aggregate, request = _materialized_package(
        tmp_path, with_candidate=False
    )

    assert materialization.persisted is True
    assert materialization.manifest_ref is not None
    package = load_platinum_package(store, materialization.manifest_ref)
    assert set(package.fold_metrics["arm"].unique()) == {"baseline", "enhanced"}
    metrics = package.fold_metrics.sort_values(["arm", "fold"])
    baseline_scores = metrics.loc[metrics["arm"] == "baseline", "score"].to_list()
    enhanced_scores = metrics.loc[metrics["arm"] == "enhanced", "score"].to_list()
    # Without candidates the persisted enhanced arm mirrors the baseline evidence.
    assert enhanced_scores == baseline_scores
    assert package.aggregate.baseline_score == package.aggregate.enhanced_score
    assert package.aggregate == AggregateMetric(
        metric=request.metric,
        metric_direction=request.metric_direction,
        aggregation="unweighted_mean_of_fold_metrics",
        baseline_score=aggregate["baseline_score"],
        enhanced_score=aggregate["enhanced_score"],
        legacy_gain=aggregate["legacy_gain"],
        directional_gain=aggregate["directional_gain"],
        successful_folds=len(aggregate["fold_deltas"]),
    )
    assert package.decisions == []


def test_platinum_reconstructs_fold_evidence_offline() -> None:
    silver, gold, gold_ref = _packages()
    request = build_platinum_request(
        run_id="platinum-run",
        case_fingerprint="case",
        silver=silver,
        gold=gold,
        gold_ref=gold_ref,
        model_name="random_forest",
        metric="acc",
        seed=42,
    )
    request = platinum_request(request)
    baseline = baseline_fold_evidence(request, silver)
    candidates = candidate_fold_evidence(request, silver, gold)
    aggregate = aggregate_metrics(request, baseline, candidates)
    uncertainty = uncertainty_summary(request, aggregate)
    decisions = selection_decisions(request, gold, aggregate, uncertainty)
    checks = platinum_checks(request, silver, baseline, aggregate)
    manifest = platinum_manifest(request, _environment(), checks, aggregate, uncertainty, decisions)
    materialization = platinum_materialization(
        manifest,
        request,
        silver,
        baseline,
        candidates,
        aggregate,
        uncertainty,
        decisions,
        checks,
        ExecutionProfile.CI,
        None,
    )

    assert materialization.persisted is False
    assert len(baseline["metrics"]) == 3
    assert len(candidates["double_a"]["metrics"]) == 3
    assert aggregate["enhanced_score"] >= 0.0
    assert all(check.passed for check in checks)
