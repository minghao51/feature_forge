"""Platinum v2 evidence-set persistence and offline reconstruction coverage.

Covers plan 23 PR 4 slice B (ADR 0018 decision 8, plan 23 §4.5/§6 PR 4/§7.9):
the durable v2 evidence set (``discovery_fold_metrics.parquet``,
``selection_steps.json``, ``preprocessing.json``, ``evidence.json``) written
by ``platinum_materialization``, v1 read compatibility, byte-tamper
detection, and the semantic reconstruction ``load_platinum_package``
performs on v2 packages from persisted evidence alone.

Fixture style (``_packages``/``_materialized_package``) is copied from
``tests/unit/test_platinum_dataflow.py`` (full node chain with
``ExecutionProfile.DEVELOPMENT`` + ``LocalArtifactStore``) and the
complementary two-candidate fixture follows
``tests/unit/test_plan23_evaluation_integrity.py``; it is deliberately
duplicated here so these units stay reviewable independently of those files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from feature_forge.contracts import (
    AggregateMetric,
    ArtifactNamespace,
    ArtifactRef,
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
    PlatinumEvidenceIndex,
    PlatinumMaterialization,
    PlatinumRequest,
    PlatinumSelectionDecision,
    RunManifest,
    RunRequest,
    RunState,
    SilverPackage,
    UncertaintySummary,
)
from feature_forge.contracts.identity import gold_input_fingerprint
from feature_forge.dataflows.platinum import (
    PLATINUM_REQUIRED_ARTIFACTS,
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
from feature_forge.exceptions import DatasetError
from feature_forge.storage.local import LocalArtifactStore


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
    target: Sequence[float] = (0, 0, 0, 1, 1, 1),
    folds: Sequence[int] = (0, 1, 2, 0, 1, 2),
    candidates: Mapping[str, Sequence[float]] | None = None,
    task: str = "classification",
    partitions: Sequence[str] | None = None,
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
        pd.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
        if features is None
        else features.reset_index(drop=True)
    )
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
        fold_assignments=pd.DataFrame(
            {
                "row_id": row_ids,
                "fold": list(folds),
                # Post-ADR-0018 Silver always labels partitions; holdout
                # platinum scopes fail closed on a missing column. Default:
                # first half discovery, remainder evaluation.
                "partition": list(partitions)
                if partitions is not None
                else ["discovery"] * (n_rows // 2) + ["evaluation"] * (n_rows - n_rows // 2),
            }
        ),
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

    Style of ``tests/unit/test_plan23_evaluation_integrity.py``: mirrored
    candidates so greedy forward selection needs both steps. Partitions and
    folds are pinned so both scopes host two folds whose validation rows are
    non-degenerate for R^2 (each fold validates on >= 2 rows with a
    non-constant target).
    """
    x1 = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    x2 = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    target = [float(a + b) for a, b in zip(x1, x2, strict=True)]
    return _packages(
        features=pd.DataFrame({"a": [5.0] * 9}),
        target=target,
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


def _run_full_chain(
    tmp_path: Path,
    silver: SilverPackage,
    gold: GoldPackage,
    gold_ref: ManifestRef,
    *,
    metric: str = "acc",
) -> tuple[LocalArtifactStore, PlatinumMaterialization, dict[str, Any], PlatinumRequest]:
    """Run the full Platinum node chain and persist the package under ``tmp_path``."""
    request = build_platinum_request(
        run_id="platinum-run",
        case_fingerprint="case",
        silver=silver,
        gold=gold,
        gold_ref=gold_ref,
        model_name="random_forest",
        metric=metric,
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


def _materialized_package(
    tmp_path: Path, *, candidates: Mapping[str, Sequence[float]] | None
) -> tuple[LocalArtifactStore, PlatinumMaterialization, dict[str, Any], PlatinumRequest]:
    """Materialized package over the default single-feature fixture."""
    silver, gold, gold_ref = _packages(candidates=candidates)
    return _run_full_chain(tmp_path, silver, gold, gold_ref)


def _hand_staged_ref(
    store: LocalArtifactStore,
    manifest: RunManifest,
    request: PlatinumRequest,
    silver: SilverPackage,
    baseline: dict[str, Any],
    candidates: Any,
    aggregate: dict[str, Any],
    uncertainty: UncertaintySummary,
    decisions: list[PlatinumSelectionDecision],
    checks: list[Any],
    *,
    with_v2_artifacts: bool = True,
    uncertainty_override: UncertaintySummary | None = None,
    evidence_override: PlatinumEvidenceIndex | None = None,
    decisions_override: list[PlatinumSelectionDecision] | None = None,
) -> ManifestRef:
    """Stage and commit a package by hand so payloads can be tampered with."""
    staging = store.begin(ArtifactNamespace(layer=Layer.PLATINUM, run_id=request.run_id))
    staged_uncertainty = uncertainty_override if uncertainty_override is not None else uncertainty
    staged_decisions = decisions_override if decisions_override is not None else decisions
    selected_arm = getattr(candidates, "selected_arm", None)
    enhanced = selected_arm if selected_arm is not None else baseline
    metrics = pd.concat(
        [baseline["metrics"], enhanced["metrics"].assign(arm="enhanced")],
        ignore_index=True,
    )
    predictions = pd.concat(
        [baseline["predictions"], enhanced["predictions"].assign(arm="enhanced")],
        ignore_index=True,
    )
    aggregate_model = AggregateMetric(
        metric=request.metric,
        metric_direction=request.metric_direction,
        aggregation="unweighted_mean_of_fold_metrics",
        baseline_score=aggregate["baseline_score"],
        enhanced_score=aggregate["enhanced_score"],
        legacy_gain=aggregate["legacy_gain"],
        directional_gain=aggregate["directional_gain"],
        successful_folds=len(aggregate["fold_deltas"]),
    )
    staging.write_json("request.json", request.model_dump(mode="json"))
    staging.write_dataframe("fold_assignments.parquet", silver.fold_assignments)
    staging.write_dataframe("fold_metrics.parquet", metrics)
    staging.write_dataframe("predictions.parquet", predictions)
    staging.write_json("aggregate_metrics.json", aggregate_model.model_dump(mode="json"))
    staging.write_json("uncertainty.json", staged_uncertainty.model_dump(mode="json"))
    staging.write_json(
        "selection_decisions.json", [item.model_dump(mode="json") for item in staged_decisions]
    )
    staging.write_json("checks.json", [item.model_dump(mode="json") for item in checks])
    if with_v2_artifacts:
        discovery_arms = candidates.discovery_arms
        staging.write_dataframe(
            "discovery_fold_metrics.parquet",
            pd.concat(
                [discovery_arms, *(item["metrics"] for item in candidates.values())],
                ignore_index=True,
            ),
        )
        staging.write_json("selection_steps.json", list(candidates.selection.steps))
        staging.write_json("preprocessing.json", aggregate["preprocessing"])
        staged_evidence = (
            evidence_override
            if evidence_override is not None
            else PlatinumEvidenceIndex(
                selected=list(candidates.selection.selected),
                selection_partition=request.evaluation_policy.selection_partition,
                evaluation_protocol=request.evaluation_policy.evaluation_protocol,
                confidence_level=request.uncertainty_policy.confidence_level,
            )
        )
        staging.write_json("evidence.json", staged_evidence.model_dump(mode="json"))
    staging.set_manifest(manifest.model_copy(update={"artifacts": staging.artifacts}))
    return store.commit(staging)


def _full_chain_inputs(
    silver: SilverPackage, gold: GoldPackage, gold_ref: ManifestRef
) -> dict[str, Any]:
    """Node outputs of the full Platinum chain, for hand staging."""
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
    return {
        "manifest": manifest,
        "request": request,
        "silver": silver,
        "baseline": baseline,
        "candidates": candidates,
        "aggregate": aggregate,
        "uncertainty": uncertainty,
        "decisions": decisions,
        "checks": checks,
    }


def test_full_chain_persists_v2_evidence_set_and_reconstructs(tmp_path: Path) -> None:
    store, materialization, _aggregate, request = _materialized_package(
        tmp_path, candidates={"double_a": [0, 2, 4, 6, 8, 10]}
    )

    assert materialization.persisted is True
    ref = materialization.manifest_ref
    assert ref is not None
    manifest = store.load_manifest(ref)
    paths = {item.relative_path for item in manifest.artifacts}
    assert paths == set(PLATINUM_REQUIRED_ARTIFACTS)
    for name in (
        "discovery_fold_metrics.parquet",
        "selection_steps.json",
        "preprocessing.json",
        "evidence.json",
    ):
        assert name in paths
    package = load_platinum_package(store, ref)
    assert set(package.fold_metrics["arm"].unique()) == {"baseline", "enhanced"}
    discovery = pd.read_parquet(
        store.resolve(
            ArtifactRef(
                layer=Layer.PLATINUM,
                run_id=ref.run_id,
                relative_path="discovery_fold_metrics.parquet",
            )
        )
    )
    discovery_arms = set(discovery["arm"])
    assert "candidate:double_a" in discovery_arms
    assert "baseline" in discovery_arms
    assert set(discovery["partition"]) == {"discovery"}
    evidence = PlatinumEvidenceIndex.model_validate(store.read_json_artifact(ref, "evidence.json"))
    assert evidence.evidence_schema_version == "2"
    assert evidence.selected == ["double_a"]
    assert evidence.selection_partition == request.evaluation_policy.selection_partition
    assert evidence.confidence_level == request.uncertainty_policy.confidence_level


def test_no_candidate_v2_package_reconstructs_with_empty_selection(tmp_path: Path) -> None:
    store, materialization, _aggregate, _request = _materialized_package(tmp_path, candidates=None)

    assert materialization.persisted is True
    ref = materialization.manifest_ref
    assert ref is not None
    evidence = PlatinumEvidenceIndex.model_validate(store.read_json_artifact(ref, "evidence.json"))
    assert evidence.selected == []
    assert store.read_json_artifact(ref, "selection_steps.json") == []
    package = load_platinum_package(store, ref)
    assert package.decisions == []


@pytest.mark.parametrize(
    ("artifact", "offset"),
    [("discovery_fold_metrics.parquet", 40), ("evidence.json", 4)],
)
def test_byte_tampering_fails_package_load(tmp_path: Path, artifact: str, offset: int) -> None:
    store, materialization, _aggregate, _request = _materialized_package(
        tmp_path, candidates={"double_a": [0, 2, 4, 6, 8, 10]}
    )
    assert materialization.manifest_ref is not None
    package_path = store.package_path(
        ArtifactNamespace(layer=Layer.PLATINUM, run_id="platinum-run")
    )
    target = package_path / artifact
    payload = bytearray(target.read_bytes())
    payload[offset] ^= 0xFF
    target.write_bytes(bytes(payload))

    with pytest.raises(DatasetError):
        load_platinum_package(store, materialization.manifest_ref)


def test_semantic_tamper_of_uncertainty_bounds_is_loader_caught(tmp_path: Path) -> None:
    silver, gold, gold_ref = _packages(candidates={"double_a": [0, 2, 4, 6, 8, 10]})
    inputs = _full_chain_inputs(silver, gold, gold_ref)
    store = LocalArtifactStore(tmp_path / "lake")
    tampered = inputs["uncertainty"].model_copy(
        update={"lower_bound": inputs["uncertainty"].lower_bound + 1.0}
    )
    ref = _hand_staged_ref(
        store,
        inputs["manifest"],
        inputs["request"],
        inputs["silver"],
        inputs["baseline"],
        inputs["candidates"],
        inputs["aggregate"],
        inputs["uncertainty"],
        inputs["decisions"],
        inputs["checks"],
        uncertainty_override=tampered,
    )

    with pytest.raises(DatasetError, match="lower_bound does not reconstruct"):
        load_platinum_package(store, ref)


def test_semantic_tamper_of_selected_set_is_loader_caught(tmp_path: Path) -> None:
    silver, gold, gold_ref = _packages(candidates={"double_a": [0, 2, 4, 6, 8, 10]})
    inputs = _full_chain_inputs(silver, gold, gold_ref)
    store = LocalArtifactStore(tmp_path / "lake")
    tampered = PlatinumEvidenceIndex(
        selected=["double_a", "bogus"],
        selection_partition=inputs["request"].evaluation_policy.selection_partition,
        evaluation_protocol=inputs["request"].evaluation_policy.evaluation_protocol,
        confidence_level=inputs["request"].uncertainty_policy.confidence_level,
    )
    ref = _hand_staged_ref(
        store,
        inputs["manifest"],
        inputs["request"],
        inputs["silver"],
        inputs["baseline"],
        inputs["candidates"],
        inputs["aggregate"],
        inputs["uncertainty"],
        inputs["decisions"],
        inputs["checks"],
        evidence_override=tampered,
    )

    with pytest.raises(DatasetError, match="selected set disagrees"):
        load_platinum_package(store, ref)


def test_semantic_tamper_of_decision_bounds_is_loader_caught(tmp_path: Path) -> None:
    """Decision records carry the evaluation-aggregate bound; drift is caught."""
    silver, gold, gold_ref = _packages(candidates={"double_a": [0, 2, 4, 6, 8, 10]})
    inputs = _full_chain_inputs(silver, gold, gold_ref)
    store = LocalArtifactStore(tmp_path / "lake")
    tampered = [
        item.model_copy(update={"lower_bound": item.lower_bound + 1.0})
        for item in inputs["decisions"]
    ]
    ref = _hand_staged_ref(
        store,
        inputs["manifest"],
        inputs["request"],
        inputs["silver"],
        inputs["baseline"],
        inputs["candidates"],
        inputs["aggregate"],
        inputs["uncertainty"],
        inputs["decisions"],
        inputs["checks"],
        decisions_override=tampered,
    )

    with pytest.raises(DatasetError, match="decision bounds disagree"):
        load_platinum_package(store, ref)


def test_v1_package_without_evidence_marker_loads_with_v1_semantics(tmp_path: Path) -> None:
    silver, gold, gold_ref = _packages(candidates={"double_a": [0, 2, 4, 6, 8, 10]})
    inputs = _full_chain_inputs(silver, gold, gold_ref)
    store = LocalArtifactStore(tmp_path / "lake")
    ref = _hand_staged_ref(
        store,
        inputs["manifest"],
        inputs["request"],
        inputs["silver"],
        inputs["baseline"],
        inputs["candidates"],
        inputs["aggregate"],
        inputs["uncertainty"],
        inputs["decisions"],
        inputs["checks"],
        with_v2_artifacts=False,
    )
    manifest = store.load_manifest(ref)
    assert "evidence.json" not in {item.relative_path for item in manifest.artifacts}

    package = load_platinum_package(store, ref)
    assert {item.feature_name for item in package.decisions if item.selected} == {"double_a"}
    assert set(package.fold_metrics["arm"].unique()) == {"baseline", "enhanced"}


def test_complementary_two_step_greedy_persists_greedy_arms_and_reconstructs(
    tmp_path: Path,
) -> None:
    silver, gold, gold_ref = _complementary_packages()
    store, materialization, _aggregate, _request = _run_full_chain(
        tmp_path, silver, gold, gold_ref, metric="r2"
    )

    assert materialization.persisted is True
    ref = materialization.manifest_ref
    assert ref is not None
    steps = store.read_json_artifact(ref, "selection_steps.json")
    assert len(steps) == 2
    assert all(step["chosen"] is not None for step in steps)
    evidence = PlatinumEvidenceIndex.model_validate(store.read_json_artifact(ref, "evidence.json"))
    assert evidence.selected == [step["chosen"] for step in steps]
    discovery = pd.read_parquet(
        store.resolve(
            ArtifactRef(
                layer=Layer.PLATINUM,
                run_id=ref.run_id,
                relative_path="discovery_fold_metrics.parquet",
            )
        )
    )
    arms = set(discovery["arm"])
    assert "candidate:x1" in arms
    assert "candidate:x2" in arms
    # Step 2 must have evaluated a greedy-labeled combined arm.
    assert any(arm.startswith("greedy:") for arm in arms)
    # Reconstruction validates both greedy steps offline.
    package = load_platinum_package(store, ref)
    assert {item.feature_name for item in package.decisions if item.selected} == {"x1", "x2"}
