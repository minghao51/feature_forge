"""Gold feature generation, durable evidence, and provider-free replay.

The Gold driver is one-call: ``execute(['gold_materialization'])`` derives the
request, evidence, and per-attempt manifest from upstream nodes, so callers
supply only the method, verified Silver package, Gold request, environment,
and sandbox.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from feature_forge.contracts import (
    ArtifactNamespace,
    CheckResult,
    EnvironmentSnapshot,
    FeatureCandidate,
    FeatureDecision,
    FeatureDecisionState,
    FeatureProvenance,
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
from feature_forge.dataflows._io import (
    load_gold_package as load_gold_package,
)
from feature_forge.dataflows._io import (
    replay_gold_package as replay_gold_package,
)
from feature_forge.dataflows.hamilton_compat import cache, tag
from feature_forge.dataflows.profile import ExecutionProfile, get_profile_policy
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.evaluation.scope import row_local_violations
from feature_forge.exceptions import DatasetError
from feature_forge.storage.hashing import sha256_bytes
from feature_forge.storage.local import LocalArtifactStore


def _request(request: GoldRequest) -> GoldRequest:
    return GoldRequest.model_validate(request.model_dump(mode="python"))


def _scripts(method: Any) -> list[str]:
    scripts = getattr(method, "generated_scripts", [])
    if callable(scripts):
        scripts = scripts()
    return [str(script) for script in scripts]


@tag(
    layer="gold",
    cost="cheap",
    persistence="none",
    sensitivity="dataset-derived",
    owner="methods",
)
def method_generation_request(gold_request: GoldRequest) -> GoldRequest:
    """Normalize the executor-supplied Gold request into the node graph."""
    return _request(gold_request)


@tag(
    layer="gold",
    cost="expensive",
    persistence="recompute",
    sensitivity="dataset-derived",
    owner="methods",
)
@cache(behavior="recompute")
def candidate_feature_specs(
    method: Any, verified_silver_package: SilverPackage, method_generation_request: GoldRequest
) -> dict[str, Any]:
    """Invoke the method's generation surface to obtain candidate specs."""
    scripts = _scripts(method)
    metadata = getattr(method, "feature_metadata", [])
    if callable(metadata):
        metadata = metadata()
    candidates: list[FeatureCandidate] = []
    code: dict[str, str] = {}
    for index, script in enumerate(scripts):
        path = f"generated_code/batch_{index:04d}.py"
        code[path] = script
        specs = (
            metadata[index] if index < len(metadata) and isinstance(metadata[index], dict) else {}
        )
        names = specs.get("name") or specs.get("names") or f"feature_{index:04d}"
        if isinstance(names, str):
            names = [names]
        for name in names:
            candidates.append(
                FeatureCandidate(
                    candidate_id=f"{method_generation_request.run_id}-candidate-{len(candidates):04d}",
                    name=str(name),
                    batch_index=index,
                    code_path=path,
                    specification=dict(specs),
                )
            )
    return {
        "request": method_generation_request,
        "candidates": candidates,
        "code": code,
        "silver": verified_silver_package,
    }


@tag(
    layer="gold",
    cost="expensive",
    persistence="recompute",
    sensitivity="dataset-derived",
    owner="sandbox",
)
@cache(behavior="recompute")
def candidate_execution_batches(
    candidate_feature_specs: dict[str, Any], sandbox: SandboxedExecutor
) -> dict[str, Any]:
    """Execute generated batches and enforce the row-local scope contract.

    Each generated script runs in the sandbox over the working feature frame.
    After structural validation (missing columns, missing values), the
    row-local scope contract (plan 23 §4.2, ADR 0018) applies: under the
    ``holdout`` evaluation protocol every candidate must pass deterministic
    row-subset/row-permutation metamorphic probes
    (:func:`~feature_forge.evaluation.scope.row_local_violations`); candidates
    whose probes reveal companion-row or frame-position dependence are
    rejected with reason code ``not_row_local` — their columns never enter
    the working frame, so later batches cannot depend on them. The probes are
    a verification barrier, not a proof of row-locality: a candidate is
    accepted only when no probe can distinguish its output from a row-local
    computation. Under the ``compatibility`` protocol probes are skipped and
    legacy whole-frame scripts replay unchanged. The frame passed to the
    sandbox never contains evaluation targets.
    """
    silver: SilverPackage = candidate_feature_specs["silver"]
    working = silver.canonical_features.copy()
    outputs: list[pd.DataFrame] = []
    decisions: list[FeatureDecision] = []
    for path, code in candidate_feature_specs["code"].items():
        candidates = [
            item for item in candidate_feature_specs["candidates"] if item.code_path == path
        ]
        try:
            output = sandbox.execute(code, working, source="gold_generation")
            if not isinstance(output, pd.DataFrame):
                raise DatasetError("generated code did not return a DataFrame")
            names = [item.name for item in candidates]
            missing = [name for name in names if name not in output.columns]
            if missing:
                raise DatasetError(f"generated feature columns are missing: {missing}")
            batch = output[names].reset_index(drop=True)
            if batch.isna().any().any():
                raise DatasetError("generated feature contains missing values")
            request: GoldRequest = candidate_feature_specs["request"]
            violations = (
                row_local_violations(sandbox, code, working, output, names)
                if request.evaluation_protocol == "holdout"
                else {}
            )
            accepted_names = [name for name in names if name not in violations]
            if accepted_names:
                accepted_frame = output[accepted_names].reset_index(drop=True)
                outputs.append(accepted_frame)
                working = pd.concat([working, accepted_frame], axis=1)
            for item in candidates:
                if item.name in violations:
                    decisions.append(
                        FeatureDecision(
                            candidate_id=item.candidate_id,
                            state=FeatureDecisionState.REJECTED,
                            reason_code="not_row_local",
                            reason=(
                                f"{violations[item.name]} (row-local scope contract, ADR 0018; "
                                "whole-frame/fitted transformations are not accepted under the "
                                "holdout protocol)"
                            ),
                        )
                    )
                else:
                    decisions.append(
                        FeatureDecision(
                            candidate_id=item.candidate_id,
                            state=FeatureDecisionState.ACCEPTED,
                            reason_code="accepted",
                            reason="feature passed structural validation",
                        )
                    )
        except Exception as exc:
            for item in candidates:
                decisions.append(
                    FeatureDecision(
                        candidate_id=item.candidate_id,
                        state=FeatureDecisionState.ERROR,
                        reason_code=type(exc).__name__,
                        reason=str(exc),
                    )
                )
    accepted = (
        pd.concat([silver.row_ids.reset_index(drop=True), *outputs], axis=1)
        if outputs
        else silver.row_ids.copy()
    )
    candidate_frame = accepted.copy()
    return {
        **candidate_feature_specs,
        "decisions": decisions,
        "accepted": accepted,
        "candidate_features": candidate_frame,
    }


@tag(layer="gold", cost="cheap", persistence="none", sensitivity="metadata", owner="verification")
def candidate_verification(candidate_execution_batches: dict[str, Any]) -> dict[str, Any]:
    """Structural verification pass over the executed candidate evidence."""
    return candidate_execution_batches


@tag(layer="gold", cost="cheap", persistence="none", sensitivity="metadata", owner="methods")
def candidate_selection(candidate_verification: dict[str, Any]) -> dict[str, Any]:
    """Selection pass producing the final candidate evidence record."""
    return candidate_verification


def _evidence(
    request: GoldRequest, evidence: dict[str, Any], environment: EnvironmentSnapshot
) -> GoldPackage:
    candidates: list[FeatureCandidate] = evidence["candidates"]
    code: dict[str, str] = evidence["code"]
    provenance = [
        FeatureProvenance(
            candidate_id=item.candidate_id,
            method=request.method_name,
            method_version=request.method_version,
            prompt_bundle_fingerprint=request.prompt_bundle_fingerprint,
            code_sha256=sha256_bytes(code[item.code_path].encode()),
            source_columns=[str(value) for value in item.specification.get("base_columns", [])],
        )
        for item in candidates
    ]
    decisions: list[FeatureDecision] = evidence["decisions"]
    checks = [
        CheckResult(
            check_id="GOLD.ROW_ID.ALIGNED", passed=True, message="Gold rows use Silver row IDs"
        )
    ]
    layer_fingerprint = gold_materialization_fingerprint(
        gold_input_fingerprint_value=request.gold_input_fingerprint,
        candidates=[item.model_dump(mode="json") for item in candidates],
        provenance=[item.model_dump(mode="json") for item in provenance],
        decisions=[item.model_dump(mode="json") for item in decisions],
    )
    manifest = RunManifest(
        layer=Layer.GOLD,
        package_kind="gold",
        layer_fingerprint=layer_fingerprint,
        reuse_fingerprint=request.gold_input_fingerprint,
        run_id=request.run_id,
        case_fingerprint=request.case_fingerprint,
        state=RunState.SUCCEEDED,
        request=RunRequest(
            dataset="silver",
            method=request.method_name,
            model="none",
            seed=0,
            options={"gold_input_fingerprint": request.gold_input_fingerprint},
        ),
        environment=environment,
        upstream_manifests=[request.silver_manifest],
        stages=[
            StageResult(
                stage="gold",
                state=RunState.SUCCEEDED,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                checks=checks,
            )
        ],
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    failures: dict[str, int] = {}
    for item in decisions:
        if item.state is not FeatureDecisionState.ACCEPTED:
            failures[item.reason_code] = failures.get(item.reason_code, 0) + 1
    counts = GoldFeatureCounts(
        candidates_proposed=len(candidates),
        candidates_executed=sum(item.state is not FeatureDecisionState.ERROR for item in decisions),
        candidates_accepted=sum(item.state is FeatureDecisionState.ACCEPTED for item in decisions),
        accepted_output_columns=max(len(evidence["accepted"].columns) - 1, 0),
        failure_counts=failures,
    )
    return GoldPackage(
        manifest=manifest,
        request=request,
        candidates=candidates,
        provenance=provenance,
        decisions=decisions,
        candidate_features=evidence["candidate_features"] if request.persist_candidates else None,
        accepted_features=evidence["accepted"],
        checks=checks,
        code=code,
        dependencies={},
        counts=counts,
    )


@tag(
    layer="gold",
    cost="cheap",
    persistence="recompute",
    sensitivity="metadata",
    owner="verification",
)
@cache(behavior="recompute")
def gold_manifest(
    candidate_selection: dict[str, Any],
    method_generation_request: GoldRequest,
    environment_snapshot: EnvironmentSnapshot,
) -> RunManifest:
    """Build the per-attempt Gold manifest from node-produced evidence."""
    return _evidence(method_generation_request, candidate_selection, environment_snapshot).manifest


def _write_package(
    store: LocalArtifactStore, package: GoldPackage
) -> tuple[RunManifest, ManifestRef]:
    staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id=package.request.run_id))
    staging.write_json("request.json", package.request.model_dump(mode="json"))
    staging.write_json(
        "candidates.json", [item.model_dump(mode="json") for item in package.candidates]
    )
    staging.write_json(
        "provenance.json", [item.model_dump(mode="json") for item in package.provenance]
    )
    staging.write_json(
        "decisions.json", [item.model_dump(mode="json") for item in package.decisions]
    )
    staging.write_dataframe("accepted_features.parquet", package.accepted_features)
    if package.candidate_features is not None:
        staging.write_dataframe(
            "candidate_features.parquet", package.candidate_features, required=False
        )
    staging.write_json("checks.json", [item.model_dump(mode="json") for item in package.checks])
    staging.write_json("dependencies.json", package.dependencies)
    for path, code in package.code.items():
        staging.write_text(path, code, media_type="text/x-python", required=False)
    staging.set_manifest(package.manifest.model_copy(update={"artifacts": staging.artifacts}))
    manifest_ref = store.commit(staging)
    committed_manifest = staging.manifest  # commit installs the final descriptors
    if committed_manifest is None:
        raise DatasetError("Gold package commit did not install a manifest")
    return committed_manifest, manifest_ref


@tag(
    layer="gold", cost="io", persistence="boundary", sensitivity="dataset-derived", owner="storage"
)
@cache(behavior="recompute")
def gold_materialization(
    candidate_selection: dict[str, Any],
    method_generation_request: GoldRequest,
    environment_snapshot: EnvironmentSnapshot,
    execution_profile: ExecutionProfile,
    artifact_store: LocalArtifactStore | None,
    gold_manifest: RunManifest | None = None,
) -> GoldMaterialization:
    """Persist the Gold package using node-produced request, evidence, and manifest.

    The trailing ``gold_manifest`` parameter is named after the upstream
    ``gold_manifest`` node so Hamilton resolves it inside the one-call driver
    execution of ``['gold_materialization']``. Direct callers may omit it; the
    per-attempt manifest is then derived from the same node evidence.
    """
    package = _evidence(method_generation_request, candidate_selection, environment_snapshot)
    manifest = package.manifest
    if gold_manifest is not None:
        if gold_manifest.layer_fingerprint != manifest.layer_fingerprint:
            raise DatasetError("Gold manifest does not match selected evidence")
        manifest = gold_manifest
    package = package.model_copy(update={"manifest": manifest})
    policy = get_profile_policy(execution_profile)
    if not policy.persistence_enabled or artifact_store is None:
        return GoldMaterialization(
            manifest=package.manifest, persisted=False, counts=package.counts
        )
    try:
        committed_manifest, ref = _write_package(artifact_store, package)
    except FileExistsError:
        raise DatasetError("Existing Gold package failed verification") from None
    return GoldMaterialization(
        manifest=committed_manifest,
        manifest_ref=ref,
        persisted=True,
        artifact_paths=[item.relative_path for item in committed_manifest.artifacts],
        counts=package.counts,
    )
