"""Gold generation, atomic materialization, verified loading, and replay."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from feature_forge.contracts import (
    ArtifactNamespace,
    ArtifactRef,
    CheckResult,
    EnvironmentSnapshot,
    FeatureCandidate,
    FeatureDecision,
    FeatureDecisionState,
    FeatureProvenance,
    GoldExecutionResult,
    GoldFeatureCounts,
    GoldMaterialization,
    GoldPackage,
    GoldRequest,
    Layer,
    ManifestRef,
    RunManifest,
    RunRequest,
    RunState,
    SilverPackage,
    StageResult,
    gold_materialization_fingerprint,
)
from feature_forge.dataflows.hamilton_compat import tag
from feature_forge.dataflows.profile import ExecutionProfile, get_profile_policy
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import DatasetError
from feature_forge.llm.replay import replay_provider_guard
from feature_forge.methods.adapter import (
    GoldGenerationPlan,
    GoldMethodEvidence,
    execute_method_plan,
    prepare_method_generation,
    select_method_evidence,
    verify_method_evidence,
)
from feature_forge.methods.base import MethodProtocol
from feature_forge.storage.hashing import sha256_bytes
from feature_forge.storage.local import LocalArtifactStore

GOLD_REQUIRED_ARTIFACTS = {
    "request.json": "application/json",
    "candidates.json": "application/json",
    "provenance.json": "application/json",
    "decisions.json": "application/json",
    "accepted_features.parquet": "application/vnd.apache.parquet",
    "checks.json": "application/json",
    "dependencies.json": "application/json",
}
GOLD_OPTIONAL_ARTIFACTS = {
    "candidate_features.parquet": "application/vnd.apache.parquet",
    "generated_code/*.py": "text/x-python",
}


def _validated_request(request: GoldRequest) -> GoldRequest:
    """Revalidate model-copy inputs and bind all Gold identity fields."""
    return GoldRequest.model_validate(request.model_dump(mode="python"))


def gold_feature_counts(evidence: GoldMethodEvidence) -> GoldFeatureCounts:
    """Derive exact, non-overloaded feature accounting from Gold evidence."""
    failure_counts: dict[str, int] = {}
    for decision in evidence.decisions:
        if decision.state is not FeatureDecisionState.ACCEPTED:
            failure_counts[decision.reason_code] = failure_counts.get(decision.reason_code, 0) + 1
    return GoldFeatureCounts(
        candidates_proposed=len(evidence.candidates),
        candidates_executed=sum(
            decision.state is not FeatureDecisionState.ERROR for decision in evidence.decisions
        ),
        candidates_accepted=sum(
            decision.state is FeatureDecisionState.ACCEPTED for decision in evidence.decisions
        ),
        accepted_output_columns=max(len(evidence.accepted_features.columns) - 1, 0),
        failure_counts=failure_counts,
    )


@tag(
    layer="gold",
    cost="expensive",
    persistence="none",
    sensitivity="dataset-derived",
    owner="methods",
)
def method_generation_request(gold_request: GoldRequest) -> GoldRequest:
    """Freeze the secret-free request before any method/provider work."""
    return _validated_request(gold_request)


@tag(
    layer="gold",
    cost="expensive",
    persistence="none",
    sensitivity="dataset-derived",
    owner="methods",
)
def candidate_feature_specs(
    method: MethodProtocol,
    silver_package: SilverPackage,
    method_generation_request: GoldRequest,
) -> GoldGenerationPlan:
    return prepare_method_generation(method, silver_package, method_generation_request)


@tag(
    layer="gold",
    cost="expensive",
    persistence="none",
    sensitivity="dataset-derived",
    owner="methods",
)
def candidate_execution_batches(
    candidate_feature_specs: GoldGenerationPlan,
    sandbox: SandboxedExecutor,
) -> GoldMethodEvidence:
    """Execute ordered generated-code batches in the sandbox."""
    return execute_method_plan(candidate_feature_specs, sandbox)


@tag(layer="gold", cost="cheap", persistence="none", sensitivity="metadata", owner="verification")
def candidate_verification(
    candidate_execution_batches: GoldMethodEvidence,
) -> GoldMethodEvidence:
    """Validate evidence cardinality and structural checks."""
    return verify_method_evidence(candidate_execution_batches)


@tag(layer="gold", cost="cheap", persistence="none", sensitivity="metadata", owner="methods")
def candidate_selection(candidate_verification: GoldMethodEvidence) -> GoldMethodEvidence:
    """Return validation-accepted evidence; metric selection is Platinum work."""
    return select_method_evidence(candidate_verification)


def _manifest(
    request: GoldRequest,
    evidence: GoldMethodEvidence,
    environment: EnvironmentSnapshot,
) -> RunManifest:
    now = datetime.now(UTC)
    fingerprint = gold_materialization_fingerprint(
        gold_input_fingerprint_value=request.gold_input_fingerprint,
        candidates=[item.model_dump(mode="json") for item in evidence.candidates],
        provenance=[item.model_dump(mode="json") for item in evidence.provenance],
        decisions=[item.model_dump(mode="json") for item in evidence.decisions],
    )
    return RunManifest(
        layer=Layer.GOLD,
        package_kind="gold",
        layer_fingerprint=fingerprint,
        run_id=request.run_id,
        case_fingerprint=request.case_fingerprint,
        state=RunState.SUCCEEDED,
        request=RunRequest(
            dataset="silver",
            method=request.method_name,
            model="none",
            seed=environment.random_seeds.get("method", 0),
            options={
                "gold_input_fingerprint": request.gold_input_fingerprint,
                "persist_candidates": request.persist_candidates,
                "selection_semantics": "validation-accepted",
            },
        ),
        environment=environment,
        upstream_manifests=[request.silver_manifest],
        stages=[
            StageResult(
                stage="gold",
                state=RunState.SUCCEEDED,
                started_at=now,
                finished_at=now,
                checks=evidence.checks,
            )
        ],
        created_at=now,
        completed_at=now,
    )


@tag(layer="gold", cost="cheap", persistence="none", sensitivity="metadata", owner="storage")
def gold_manifest(
    candidate_selection: GoldMethodEvidence,
    method_generation_request: GoldRequest,
    environment_snapshot: EnvironmentSnapshot,
) -> RunManifest:
    """Build the content-bound Gold manifest before atomic publication."""
    method_generation_request = _validated_request(method_generation_request)
    return _manifest(method_generation_request, candidate_selection, environment_snapshot)


@tag(
    layer="gold", cost="io", persistence="boundary", sensitivity="dataset-derived", owner="storage"
)
def gold_materialization(
    candidate_selection: GoldMethodEvidence,
    method_generation_request: GoldRequest,
    environment_snapshot: EnvironmentSnapshot,
    execution_profile: ExecutionProfile,
    artifact_store: LocalArtifactStore | None,
    gold_manifest: RunManifest,
) -> GoldMaterialization:
    method_generation_request = _validated_request(method_generation_request)
    evidence = select_method_evidence(verify_method_evidence(candidate_selection))
    gold_request = method_generation_request
    manifest = gold_manifest
    expected_fingerprint = _manifest(
        gold_request,
        evidence,
        environment_snapshot,
    ).layer_fingerprint
    if manifest.layer_fingerprint != expected_fingerprint:
        raise DatasetError("Gold manifest does not match selected evidence")
    policy = get_profile_policy(execution_profile)
    counts = gold_feature_counts(evidence)
    if not policy.persistence_enabled or artifact_store is None:
        return GoldMaterialization(manifest=manifest, persisted=False, counts=counts)
    namespace = ArtifactNamespace(layer=Layer.GOLD, run_id=gold_request.run_id)
    try:
        existing = artifact_store.get_manifest_ref(namespace)
    except ValueError as exc:
        raise DatasetError(f"Existing Gold package failed verification: {exc}") from exc
    if existing is not None:
        loaded = load_gold_package(artifact_store, existing).manifest
        if loaded.layer_fingerprint != manifest.layer_fingerprint:
            raise DatasetError("Gold run ID already contains different evidence")
        return GoldMaterialization(
            manifest=loaded,
            manifest_ref=existing,
            persisted=True,
            artifact_paths=[item.relative_path for item in loaded.artifacts],
            counts=counts,
        )
    staging = artifact_store.begin(namespace)
    staging.write_json("request.json", gold_request.model_dump(mode="json"))
    staging.write_json("candidates.json", [x.model_dump(mode="json") for x in evidence.candidates])
    staging.write_json("provenance.json", [x.model_dump(mode="json") for x in evidence.provenance])
    staging.write_json("decisions.json", [x.model_dump(mode="json") for x in evidence.decisions])
    if gold_request.persist_candidates:
        staging.write_dataframe("candidate_features.parquet", evidence.candidate_features)
    staging.write_dataframe("accepted_features.parquet", evidence.accepted_features)
    staging.write_json("checks.json", [x.model_dump(mode="json") for x in evidence.checks])
    staging.write_json("dependencies.json", evidence.dependencies)
    for path, code in evidence.code.items():
        staging.write_text(path, code, media_type="text/x-python")
    staging.set_manifest(manifest)
    ref = artifact_store.commit(staging)
    committed = cast(RunManifest, staging.manifest)
    return GoldMaterialization(
        manifest=committed,
        manifest_ref=ref,
        persisted=True,
        artifact_paths=[item.relative_path for item in committed.artifacts],
        counts=counts,
    )


def _json_artifact(store: LocalArtifactStore, ref: ManifestRef, path: str) -> Any:
    resolved = store.resolve(ArtifactRef(layer=Layer.GOLD, run_id=ref.run_id, relative_path=path))
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_gold_package(store: LocalArtifactStore, ref: ManifestRef) -> GoldPackage:
    if ref.layer is not Layer.GOLD:
        raise DatasetError("Gold loading requires a Gold manifest reference")
    try:
        manifest = store.load_manifest(ref)
        if manifest.layer is not Layer.GOLD or manifest.package_kind != "gold":
            raise ValueError("manifest is not a Gold package")
        by_path = {item.relative_path: item for item in manifest.artifacts}
        for path, media_type in GOLD_REQUIRED_ARTIFACTS.items():
            item = by_path.get(path)
            if (
                item is None
                or not item.required
                or item.schema_version != "1"
                or item.media_type != media_type
                or item.layer is not Layer.GOLD
            ):
                raise ValueError(f"invalid required Gold artifact: {path}")
        request = GoldRequest.model_validate(_json_artifact(store, ref, "request.json"))
        expected = set(GOLD_REQUIRED_ARTIFACTS)
        if request.persist_candidates:
            expected.add("candidate_features.parquet")
        code_paths = {
            candidate.code_path
            for candidate in [
                FeatureCandidate.model_validate(x)
                for x in _json_artifact(store, ref, "candidates.json")
            ]
        }
        expected.update(code_paths)
        if set(by_path) != expected:
            raise ValueError("Gold package artifact set does not match its contract")
        if request.persist_candidates:
            candidate_descriptor = by_path["candidate_features.parquet"]
            if (
                candidate_descriptor.media_type != "application/vnd.apache.parquet"
                or candidate_descriptor.layer is not Layer.GOLD
                or candidate_descriptor.schema_version != "1"
            ):
                raise ValueError("invalid candidate matrix descriptor")
        if any(
            by_path[path].media_type != "text/x-python"
            or by_path[path].layer is not Layer.GOLD
            or by_path[path].schema_version != "1"
            for path in code_paths
        ):
            raise ValueError("invalid generated code descriptor")
        candidates = [
            FeatureCandidate.model_validate(x)
            for x in _json_artifact(store, ref, "candidates.json")
        ]
        provenance = [
            FeatureProvenance.model_validate(x)
            for x in _json_artifact(store, ref, "provenance.json")
        ]
        decisions = [
            FeatureDecision.model_validate(x) for x in _json_artifact(store, ref, "decisions.json")
        ]
        checks = [CheckResult.model_validate(x) for x in _json_artifact(store, ref, "checks.json")]
        accepted = pd.read_parquet(
            store.resolve(
                ArtifactRef(
                    layer=Layer.GOLD, run_id=ref.run_id, relative_path="accepted_features.parquet"
                )
            )
        )
        candidate_frame = None
        if request.persist_candidates:
            candidate_frame = pd.read_parquet(
                store.resolve(
                    ArtifactRef(
                        layer=Layer.GOLD,
                        run_id=ref.run_id,
                        relative_path="candidate_features.parquet",
                    )
                )
            )
        code = {
            path: store.resolve(
                ArtifactRef(layer=Layer.GOLD, run_id=ref.run_id, relative_path=path)
            ).read_text(encoding="utf-8")
            for path in code_paths
        }
        dependencies = cast(dict[str, Any], _json_artifact(store, ref, "dependencies.json"))
    except Exception as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"Unable to load verified Gold package: {exc}") from exc
    ids = [item.candidate_id for item in candidates]
    provenance_ids = [item.candidate_id for item in provenance]
    decision_ids = [item.candidate_id for item in decisions]
    candidate_names = [item.name for item in candidates]
    check_ids = [item.check_id for item in checks]
    if (
        len(ids) != len(set(ids))
        or len(provenance_ids) != len(ids)
        or len(decision_ids) != len(ids)
        or len(provenance_ids) != len(set(provenance_ids))
        or len(decision_ids) != len(set(decision_ids))
        or set(provenance_ids) != set(ids)
        or set(decision_ids) != set(ids)
        or len(candidate_names) != len(set(candidate_names))
        or len(check_ids) != len(set(check_ids))
        or any(check.required and not check.passed for check in checks)
        or manifest.upstream_manifests != [request.silver_manifest]
        or manifest.run_id != ref.run_id
        or request.run_id != ref.run_id
        or manifest.request.method != request.method_name
        or manifest.request.options.get("gold_input_fingerprint") != request.gold_input_fingerprint
    ):
        raise DatasetError("Verified Gold package contains inconsistent evidence")
    recomputed = gold_materialization_fingerprint(
        gold_input_fingerprint_value=request.gold_input_fingerprint,
        candidates=[item.model_dump(mode="json") for item in candidates],
        provenance=[item.model_dump(mode="json") for item in provenance],
        decisions=[item.model_dump(mode="json") for item in decisions],
    )
    provenance_by_id = {item.candidate_id: item for item in provenance}
    accepted_ids = {
        item.candidate_id for item in decisions if item.state is FeatureDecisionState.ACCEPTED
    }
    accepted_names = [item.name for item in candidates if item.candidate_id in accepted_ids]
    non_error_ids = {
        item.candidate_id for item in decisions if item.state is not FeatureDecisionState.ERROR
    }
    candidate_output_names = [
        item.name for item in candidates if item.candidate_id in non_error_ids
    ]
    accepted_row_ids = accepted.get("row_id")
    candidate_row_ids = candidate_frame.get("row_id") if candidate_frame is not None else None
    if (
        recomputed != manifest.layer_fingerprint
        or list(accepted.columns) != ["row_id", *accepted_names]
        or accepted_row_ids is None
        or accepted_row_ids.duplicated().any()
        or (
            candidate_frame is not None
            and (
                list(candidate_frame.columns) != ["row_id", *candidate_output_names]
                or candidate_row_ids is None
                or candidate_row_ids.tolist() != accepted_row_ids.tolist()
            )
        )
        or any(
            provenance_by_id[item.candidate_id].code_sha256
            != sha256_bytes(code[item.code_path].encode())
            for item in candidates
        )
    ):
        raise DatasetError("Verified Gold package semantic hashes do not match its manifest")
    return GoldPackage(
        manifest=manifest,
        request=request,
        candidates=candidates,
        provenance=provenance,
        decisions=decisions,
        candidate_features=candidate_frame,
        accepted_features=accepted,
        checks=checks,
        code=code,
        dependencies=dependencies,
        counts=GoldFeatureCounts(
            candidates_proposed=len(candidates),
            candidates_executed=sum(
                item.state is not FeatureDecisionState.ERROR for item in decisions
            ),
            candidates_accepted=len(accepted_ids),
            accepted_output_columns=len(accepted_names),
            failure_counts={
                reason: sum(
                    item.state is not FeatureDecisionState.ACCEPTED and item.reason_code == reason
                    for item in decisions
                )
                for reason in {
                    item.reason_code
                    for item in decisions
                    if item.state is not FeatureDecisionState.ACCEPTED
                }
            },
        ),
    )


def replay_gold_package(
    store: LocalArtifactStore,
    ref: ManifestRef,
    silver: SilverPackage,
    sandbox: SandboxedExecutor,
) -> GoldPackage:
    package = load_gold_package(store, ref)
    from feature_forge.dataflows.silver import load_silver_package

    authoritative_silver = load_silver_package(store, package.request.silver_manifest)
    if (
        package.request.silver_manifest != package.manifest.upstream_manifests[0]
        or package.request.silver_manifest.run_id != silver.manifest.run_id
        or package.request.silver_fingerprint != authoritative_silver.manifest.layer_fingerprint
    ):
        raise DatasetError("Gold package does not reference the supplied Silver package")
    try:
        pd.testing.assert_frame_equal(
            silver.canonical_features,
            authoritative_silver.canonical_features,
            check_dtype=True,
        )
        pd.testing.assert_frame_equal(
            silver.canonical_target,
            authoritative_silver.canonical_target,
            check_dtype=True,
        )
        pd.testing.assert_frame_equal(
            silver.row_ids,
            authoritative_silver.row_ids,
            check_dtype=True,
        )
    except AssertionError as exc:
        raise DatasetError("Supplied Silver package differs from its verified manifest") from exc
    silver = authoritative_silver
    working = silver.canonical_features.copy()
    parts: list[pd.DataFrame] = []
    accepted_ids = {
        item.candidate_id
        for item in package.decisions
        if item.state is FeatureDecisionState.ACCEPTED
    }
    with replay_provider_guard():
        for path in sorted(package.code):
            batch_candidates = [item for item in package.candidates if item.code_path == path]
            batch_decisions = {
                item.candidate_id: item
                for item in package.decisions
                if item.candidate_id in {x.candidate_id for x in batch_candidates}
            }
            try:
                output = sandbox.execute(package.code[path], working, source="gold_replay")
            except Exception as exc:
                if batch_candidates and all(
                    batch_decisions[item.candidate_id].state is FeatureDecisionState.ERROR
                    and batch_decisions[item.candidate_id].reason_code == type(exc).__name__
                    for item in batch_candidates
                ):
                    continue
                raise DatasetError(f"Gold replay failure outcome mismatch for {path}") from exc
            names = [item.name for item in batch_candidates]
            if any(name not in output for name in names):
                raise DatasetError(f"Gold replay output schema mismatch for {path}")
            batch = output[names].reset_index(drop=True)
            for candidate_item in batch_candidates:
                values = batch[candidate_item.name]
                finite = not (
                    pd.api.types.is_numeric_dtype(values)
                    and bool(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any())
                )
                replayed_state = (
                    FeatureDecisionState.ACCEPTED if finite else FeatureDecisionState.REJECTED
                )
                if batch_decisions[candidate_item.candidate_id].state is not replayed_state:
                    raise DatasetError(
                        f"Gold replay decision outcome mismatch for {candidate_item.candidate_id}"
                    )
            parts.append(batch)
            accepted_names = [
                item.name for item in batch_candidates if item.candidate_id in accepted_ids
            ]
            working = pd.concat([working.reset_index(drop=True), batch[accepted_names]], axis=1)
    candidate = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=working.index)
    row_ids = silver.row_ids["row_id"].reset_index(drop=True)
    replayed = pd.concat([row_ids.rename("row_id"), candidate], axis=1)
    if package.candidate_features is not None:
        pd.testing.assert_frame_equal(
            replayed,
            package.candidate_features,
            check_dtype=True,
            rtol=package.request.value_rtol,
            atol=package.request.value_atol,
        )
    accepted_names = [item.name for item in package.candidates if item.candidate_id in accepted_ids]
    accepted = pd.concat([row_ids.rename("row_id"), candidate[accepted_names]], axis=1)
    pd.testing.assert_frame_equal(
        accepted,
        package.accepted_features,
        check_dtype=True,
        rtol=package.request.value_rtol,
        atol=package.request.value_atol,
    )
    return package


def execute_gold(
    *,
    store: LocalArtifactStore,
    request: GoldRequest,
    method: MethodProtocol,
    sandbox: SandboxedExecutor,
    environment: EnvironmentSnapshot,
    profile: ExecutionProfile = ExecutionProfile.PRODUCTION,
) -> GoldExecutionResult:
    """Execute the explicit durable Gold path; legacy platform execution is unchanged."""
    from feature_forge.dataflows.silver import load_silver_package

    request = method_generation_request(request)
    silver = load_silver_package(store, request.silver_manifest)
    plan = candidate_feature_specs(method, silver, request)
    executed = candidate_execution_batches(plan, sandbox)
    verified = candidate_verification(executed)
    selected = candidate_selection(verified)
    manifest = gold_manifest(selected, request, environment)
    materialization = gold_materialization(
        selected,
        request,
        environment,
        profile,
        store,
        manifest,
    )
    if materialization.manifest_ref is None:
        raise DatasetError("Explicit Gold execution requires a persistence-enabled profile")
    package = replay_gold_package(
        store,
        materialization.manifest_ref,
        silver,
        sandbox,
    )
    return GoldExecutionResult(
        materialization=materialization,
        package=package,
        counts=materialization.counts,
    )
