"""Shared non-node services for the medallion dataflows.

Stage drivers import only the ``bronze``/``silver``/``gold``/``platinum`` node
modules. Everything defined here deliberately lives outside the
Hamilton-loaded modules: package loaders, offline replay, request
construction, and the small helpers shared by node functions. Hamilton never
discovers these functions because they are either private or their
``__module__`` points at this service module instead of a node module.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd

from feature_forge.contracts import (
    AggregateMetric,
    ArtifactNamespace,
    ArtifactRef,
    BronzeRecord,
    CheckResult,
    EvaluationPolicy,
    FeatureCandidate,
    FeatureDecision,
    FeatureDecisionState,
    FeatureProvenance,
    GoldFeatureCounts,
    GoldPackage,
    GoldRequest,
    Layer,
    ManifestRef,
    ModelSpecification,
    PlatinumPackage,
    PlatinumRequest,
    PlatinumSelectionDecision,
    RunManifest,
    SelectionPolicy,
    SilverPackage,
    UncertaintyPolicy,
    UncertaintySummary,
    platinum_input_fingerprint,
)
from feature_forge.evaluation.metrics import get_metric_direction
from feature_forge.evaluation.model_factory import ModelFactory
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import DatasetError
from feature_forge.llm.replay import replay_provider_guard
from feature_forge.storage.hashing import fingerprint
from feature_forge.storage.local import LocalArtifactStore, LocalStagingArea

BRONZE_REFERENCE_REQUIRED_ARTIFACTS = {
    "bronze.json": "application/json",
    "source_reference.json": "application/json",
}
BRONZE_SNAPSHOT_REQUIRED_ARTIFACTS = {
    "bronze.json": "application/json",
    "train.parquet": "application/vnd.apache.parquet",
}
BRONZE_OPTIONAL_ARTIFACTS = {"test.parquet"}
SILVER_REQUIRED_ARTIFACTS = {
    "canonical_features.parquet": "application/vnd.apache.parquet",
    "canonical_target.parquet": "application/vnd.apache.parquet",
    "row_ids.parquet": "application/vnd.apache.parquet",
    "fold_assignments.parquet": "application/vnd.apache.parquet",
    "profile.json": "application/json",
    "checks.json": "application/json",
}


@contextmanager
def package_load_guard(layer_name: str) -> Iterator[None]:
    """Wrap a layer package's deserialization phase.

    Any exception raised inside the ``with`` block is re-raised as a
    :class:`~feature_forge.exceptions.DatasetError` with a uniform message,
    unless it is already a ``DatasetError`` (preserving the original, more
    specific context). Semantic-verification phases that run their own checks
    should stay outside this block.
    """
    try:
        yield
    except DatasetError:
        raise
    except Exception as exc:  # loader surface must normalize any deserialization error
        raise DatasetError(f"Unable to load verified {layer_name} package: {exc}") from exc


def check_result(
    check_id: str,
    passed: bool,
    message: str,
    *,
    observed: Any | None = None,
    expected: Any | None = None,
    affected_artifacts: list[str] | None = None,
    **details: Any,
) -> CheckResult:
    """Build one required boundary check result."""
    return CheckResult(
        check_id=check_id,
        passed=passed,
        required=True,
        message=message,
        observed=observed,
        expected=expected,
        affected_artifacts=affected_artifacts or [],
        details=details,
    )


def attach_staging_artifacts(manifest: RunManifest, staging: LocalStagingArea) -> RunManifest:
    """Return ``manifest`` with the staged artifact descriptors and completion time."""
    artifacts = staging.artifacts
    layer_value = staging.namespace.layer.value
    stages = [
        stage.model_copy(update={"artifacts": artifacts}) if stage.stage == layer_value else stage
        for stage in manifest.stages
    ]
    return manifest.model_copy(
        update={
            "artifacts": artifacts,
            "stages": stages,
            "completed_at": datetime.now(UTC),
        }
    )


def existing_package(
    artifact_store: LocalArtifactStore,
    namespace: ArtifactNamespace,
    expected_manifest: RunManifest,
) -> tuple[RunManifest, ManifestRef] | None:
    """Return the already-committed package for ``namespace`` when compatible.

    A different fingerprint for the same write-once namespace is a hard error;
    callers re-materializing an identical package receive the committed one.
    """
    ref = artifact_store.get_manifest_ref(namespace)
    if ref is None:
        return None
    manifest = artifact_store.load_manifest(ref)
    if (
        manifest.layer is not namespace.layer
        or manifest.layer_fingerprint != expected_manifest.layer_fingerprint
    ):
        raise DatasetError(
            f"Run '{namespace.run_id}' already contains a different {namespace.layer.value} package"
        )
    return manifest, ref


def load_silver_package(
    artifact_store: LocalArtifactStore,
    manifest_ref: ManifestRef,
) -> SilverPackage:
    """Load and validate a committed Silver package without dataset or network access."""
    if manifest_ref.layer is not Layer.SILVER:
        raise DatasetError("Silver replay requires a Silver manifest reference")
    with package_load_guard("Silver"):
        manifest = artifact_store.load_manifest(manifest_ref)
        _validate_artifact_contract(
            manifest,
            layer=Layer.SILVER,
            required=SILVER_REQUIRED_ARTIFACTS,
        )
        upstream_ref = next(ref for ref in manifest.upstream_manifests if ref.layer is Layer.BRONZE)
        bronze_manifest = artifact_store.load_manifest(upstream_ref)
        bronze_required = (
            BRONZE_REFERENCE_REQUIRED_ARTIFACTS
            if bronze_manifest.request.options.get("source_policy") == "reference"
            else BRONZE_SNAPSHOT_REQUIRED_ARTIFACTS
        )
        _validate_artifact_contract(
            bronze_manifest,
            layer=Layer.BRONZE,
            required=bronze_required,
            allowed_optional=BRONZE_OPTIONAL_ARTIFACTS,
        )
        bronze_path = artifact_store.resolve(
            ArtifactRef(
                layer=Layer.BRONZE,
                run_id=upstream_ref.run_id,
                relative_path="bronze.json",
            )
        )
        bronze = BronzeRecord.model_validate_json(bronze_path.read_text(encoding="utf-8"))

        def silver_path(relative_path: str) -> ArtifactRef:
            return ArtifactRef(
                layer=Layer.SILVER,
                run_id=manifest_ref.run_id,
                relative_path=relative_path,
            )

        def resolved_silver_path(relative_path: str) -> str:
            return str(artifact_store.resolve(silver_path(relative_path)))

        canonical_features = pd.read_parquet(resolved_silver_path("canonical_features.parquet"))
        canonical_target = pd.read_parquet(resolved_silver_path("canonical_target.parquet"))
        row_ids_frame = pd.read_parquet(resolved_silver_path("row_ids.parquet"))
        folds = pd.read_parquet(resolved_silver_path("fold_assignments.parquet"))
        profile_path = artifact_store.resolve(silver_path("profile.json"))
        checks_path = artifact_store.resolve(silver_path("checks.json"))
        profile = cast(
            dict[str, Any],
            json.loads(profile_path.read_text(encoding="utf-8")),
        )
        check_values = cast(
            list[dict[str, Any]],
            json.loads(checks_path.read_text(encoding="utf-8")),
        )
        checks = [CheckResult.model_validate(value) for value in check_values]

    row_id_values = row_ids_frame.get("row_id")
    target_row_ids = canonical_target.get("row_id")
    fold_row_ids = folds.get("row_id")
    fold_values = folds.get("fold")
    if (
        row_id_values is None
        or target_row_ids is None
        or fold_row_ids is None
        or fold_values is None
        or row_id_values.tolist() != target_row_ids.tolist()
        or row_id_values.tolist() != fold_row_ids.tolist()
        or len(canonical_features) != len(row_ids_frame)
        or list(canonical_target.columns) != ["row_id", bronze.target]
        or len(set(row_id_values.tolist())) != len(row_id_values)
        or fold_values.isna().any()
        or any(check.required and not check.passed for check in checks)
        or bronze_manifest.layer is not Layer.BRONZE
        or manifest.layer is not Layer.SILVER
        or profile.get("dataset_fingerprint") != manifest.request.options.get("dataset_fingerprint")
        or bronze.source_checksum
        != bronze_manifest.request.options.get("source", {}).get("source_checksum")
    ):
        raise DatasetError("Verified Silver package contains misaligned replay artifacts")

    return SilverPackage(
        manifest=manifest,
        bronze=bronze,
        canonical_features=canonical_features,
        canonical_target=canonical_target,
        row_ids=row_ids_frame,
        fold_assignments=folds,
        profile=profile,
        checks=checks,
    )


def _validate_artifact_contract(
    manifest: RunManifest,
    *,
    layer: Layer,
    required: dict[str, str],
    allowed_optional: set[str] | None = None,
) -> None:
    """Enforce the exact persisted file contract for a layer loader."""
    if manifest.layer is not layer or manifest.package_kind != layer.value:
        raise DatasetError(f"Expected a {layer.value} manifest")
    descriptors = {descriptor.relative_path: descriptor for descriptor in manifest.artifacts}
    required_paths = {path for path, descriptor in descriptors.items() if descriptor.required}
    if required_paths != set(required):
        raise DatasetError(
            f"{layer.value.title()} package required artifacts do not match contract"
        )
    unexpected_optional = {
        path for path, descriptor in descriptors.items() if not descriptor.required
    } - (allowed_optional or set())
    if unexpected_optional:
        raise DatasetError(
            f"{layer.value.title()} package contains unsupported optional artifacts: "
            f"{sorted(unexpected_optional)}"
        )
    for path, media_type in required.items():
        descriptor = descriptors[path]
        if (
            descriptor.layer is not layer
            or descriptor.schema_version != "1"
            or descriptor.media_type != media_type
        ):
            raise DatasetError(f"Artifact descriptor does not match contract: {path}")


def _counts(
    candidates: list[FeatureCandidate], decisions: list[FeatureDecision], accepted: pd.DataFrame
) -> GoldFeatureCounts:
    failures: dict[str, int] = {}
    for item in decisions:
        if item.state is not FeatureDecisionState.ACCEPTED:
            failures[item.reason_code] = failures.get(item.reason_code, 0) + 1
    return GoldFeatureCounts(
        candidates_proposed=len(candidates),
        candidates_executed=sum(item.state is not FeatureDecisionState.ERROR for item in decisions),
        candidates_accepted=sum(item.state is FeatureDecisionState.ACCEPTED for item in decisions),
        accepted_output_columns=max(len(accepted.columns) - 1, 0),
        failure_counts=failures,
    )


def load_gold_package(store: LocalArtifactStore, ref: ManifestRef) -> GoldPackage:
    """Load and validate a committed Gold package without executing generated code."""
    try:
        manifest = store.load_manifest(ref)
        if manifest.layer is not Layer.GOLD or not manifest.upstream_manifests:
            raise DatasetError("invalid Gold manifest")
        request = GoldRequest.model_validate(store.read_json_artifact(ref, "request.json"))
        candidates = [
            FeatureCandidate.model_validate(item)
            for item in store.read_json_artifact(ref, "candidates.json")
        ]
        provenance = [
            FeatureProvenance.model_validate(item)
            for item in store.read_json_artifact(ref, "provenance.json")
        ]
        decisions = [
            FeatureDecision.model_validate(item)
            for item in store.read_json_artifact(ref, "decisions.json")
        ]
        checks = [
            CheckResult.model_validate(item)
            for item in store.read_json_artifact(ref, "checks.json")
        ]
        accepted = pd.read_parquet(
            store.resolve(
                ArtifactRef(
                    layer=Layer.GOLD, run_id=ref.run_id, relative_path="accepted_features.parquet"
                )
            )
        )
        candidate_ref = ArtifactRef(
            layer=Layer.GOLD, run_id=ref.run_id, relative_path="candidate_features.parquet"
        )
        candidate = (
            pd.read_parquet(store.resolve(candidate_ref))
            if any(item.relative_path == candidate_ref.relative_path for item in manifest.artifacts)
            else None
        )
        code = {}
        for item in manifest.artifacts:
            if item.relative_path.startswith("generated_code/"):
                code[item.relative_path] = store.resolve(
                    ArtifactRef(
                        layer=Layer.GOLD, run_id=ref.run_id, relative_path=item.relative_path
                    )
                ).read_text()
        return GoldPackage(
            manifest=manifest,
            request=request,
            candidates=candidates,
            provenance=provenance,
            decisions=decisions,
            candidate_features=candidate,
            accepted_features=accepted,
            checks=checks,
            code=code,
            dependencies=store.read_json_artifact(ref, "dependencies.json"),
            counts=_counts(candidates, decisions, accepted),
        )
    except Exception as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"Unable to load verified Gold package: {exc}") from exc


def replay_gold_package(
    store: LocalArtifactStore, ref: ManifestRef, silver: SilverPackage, sandbox: SandboxedExecutor
) -> GoldPackage:
    """Replay committed generated code and compare it with the committed evidence."""
    package = load_gold_package(store, ref)
    if (
        package.request.silver_manifest != package.manifest.upstream_manifests[0]
        or package.request.silver_fingerprint != silver.manifest.layer_fingerprint
    ):
        raise DatasetError("Gold package does not reference the supplied Silver package")
    working = silver.canonical_features.copy()
    with replay_provider_guard():
        for path in sorted(package.code):
            batch = [item for item in package.candidates if item.code_path == path]
            try:
                output = sandbox.execute(package.code[path], working, source="gold_replay")
            except Exception as exc:
                if all(
                    item.state is FeatureDecisionState.ERROR
                    for item in package.decisions
                    if item.candidate_id in {x.candidate_id for x in batch}
                ):
                    continue
                raise DatasetError(f"Gold replay failure outcome mismatch for {path}") from exc
            names = [item.name for item in batch]
            if any(name not in output.columns for name in names):
                raise DatasetError(f"Gold replay output schema mismatch for {path}")
            values = output[names].reset_index(drop=True)
            working = pd.concat([working, values], axis=1)
    if not package.accepted_features.drop(columns=["row_id"], errors="ignore").equals(
        working.drop(columns=["row_id"], errors="ignore").iloc[
            :, -len(package.accepted_features.columns) + 1 :
        ]
        if len(package.accepted_features.columns) > 1
        else package.accepted_features.drop(columns=["row_id"], errors="ignore")
    ):
        # Compare the authoritative accepted values by column, preserving replay order.
        for name in package.accepted_features.columns.drop("row_id"):
            if name not in working or not package.accepted_features[name].reset_index(
                drop=True
            ).equals(working[name].reset_index(drop=True)):
                raise DatasetError("Gold replay values differ from committed evidence")
    return package


def _model_spec(name: str, task: str, seed: int) -> ModelSpecification:
    model = ModelFactory(random_state=seed).get_model(name, task)
    params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
    return ModelSpecification(
        name=name,
        estimator_class=f"{type(model).__module__}.{type(model).__qualname__}",
        distribution="feature-forge",
        distribution_version="1",
        resolved_params=params,
        task=task,
        seed=seed,
    )


def build_platinum_request(
    *,
    run_id: str,
    case_fingerprint: str,
    silver: SilverPackage,
    gold: GoldPackage,
    gold_ref: ManifestRef,
    model_name: str = "random_forest",
    metric: str = "acc",
    seed: int = 42,
    evaluation_policy: EvaluationPolicy | None = None,
    uncertainty_policy: UncertaintyPolicy | None = None,
    selection_policy: SelectionPolicy | None = None,
) -> PlatinumRequest:
    """Build a provider-free Platinum request from verified Silver/Gold evidence."""
    eval_policy = evaluation_policy or EvaluationPolicy()
    unc_policy = uncertainty_policy or UncertaintyPolicy()
    select_policy = selection_policy or SelectionPolicy()
    fold_fp = fingerprint(
        {
            "silver": silver.manifest.layer_fingerprint,
            "folds": silver.fold_assignments.to_dict(orient="records"),
        }
    )
    model = _model_spec(model_name, silver.bronze.task, seed)
    direction = get_metric_direction(metric)
    input_fp = platinum_input_fingerprint(
        gold_fingerprint_value=gold.manifest.layer_fingerprint,
        model_name=model.name,
        model_version=model.distribution_version,
        model_config=model.model_dump(mode="json"),
        metric=metric,
        fold_fingerprint=fold_fp,
        evaluation_policy={
            **eval_policy.model_dump(mode="json"),
            "metric_direction": direction.value,
            "selection_policy": select_policy.model_dump(mode="json"),
        },
        uncertainty_policy=unc_policy.model_dump(mode="json"),
    )
    return PlatinumRequest(
        run_id=run_id,
        case_fingerprint=case_fingerprint,
        silver_manifest=gold.request.silver_manifest,
        gold_manifest=gold_ref,
        gold_fingerprint=gold.manifest.layer_fingerprint,
        fold_fingerprint=fold_fp,
        platinum_input_fingerprint=input_fp,
        model=model,
        metric=metric,
        metric_direction=direction,
        evaluation_policy=eval_policy,
        uncertainty_policy=unc_policy,
        selection_policy=select_policy,
    )


def load_platinum_package(store: LocalArtifactStore, ref: ManifestRef) -> PlatinumPackage:
    """Load and validate persisted Platinum evidence without executing models."""
    if ref.layer is not Layer.PLATINUM:
        raise DatasetError("Platinum loading requires a Platinum manifest reference")
    try:
        manifest = store.load_manifest(ref)
        request = PlatinumRequest.model_validate(store.read_json_artifact(ref, "request.json"))
        folds = pd.read_parquet(
            store.resolve(
                ArtifactRef(
                    layer=Layer.PLATINUM,
                    run_id=ref.run_id,
                    relative_path="fold_assignments.parquet",
                )
            )
        )
        metrics = pd.read_parquet(
            store.resolve(
                ArtifactRef(
                    layer=Layer.PLATINUM, run_id=ref.run_id, relative_path="fold_metrics.parquet"
                )
            )
        )
        predictions = pd.read_parquet(
            store.resolve(
                ArtifactRef(
                    layer=Layer.PLATINUM, run_id=ref.run_id, relative_path="predictions.parquet"
                )
            )
        )
        aggregate = AggregateMetric.model_validate(
            store.read_json_artifact(ref, "aggregate_metrics.json")
        )
        uncertainty = UncertaintySummary.model_validate(
            store.read_json_artifact(ref, "uncertainty.json")
        )
        decisions = [
            PlatinumSelectionDecision.model_validate(item)
            for item in store.read_json_artifact(ref, "selection_decisions.json")
        ]
        checks = [
            CheckResult.model_validate(item)
            for item in store.read_json_artifact(ref, "checks.json")
        ]
        if manifest.upstream_manifests != [request.silver_manifest, request.gold_manifest]:
            raise DatasetError("Platinum manifest lineage does not match its request")
        required_metrics = {"arm", "fold", "metric", "score", "n_train", "n_validation", "status"}
        required_predictions = {"arm", "fold", "row_id", "target_json", "prediction_json"}
        if not required_metrics.issubset(metrics.columns) or not required_predictions.issubset(
            predictions.columns
        ):
            raise DatasetError("Platinum evidence schema is incomplete")
        baseline = metrics[metrics["arm"] == "baseline"].sort_values("fold")
        enhanced = metrics[metrics["arm"] == "enhanced"].sort_values("fold")
        if (
            baseline.empty
            or enhanced.empty
            or baseline["fold"].tolist() != enhanced["fold"].tolist()
        ):
            raise DatasetError("Platinum baseline and enhanced folds are not paired")
        if not math.isclose(
            float(baseline["score"].mean()), aggregate.baseline_score, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise DatasetError("Platinum baseline aggregate does not reconstruct")
        if not math.isclose(
            float(enhanced["score"].mean()), aggregate.enhanced_score, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise DatasetError("Platinum enhanced aggregate does not reconstruct")
        return PlatinumPackage(
            manifest=manifest,
            request=request,
            fold_assignments=folds,
            fold_metrics=metrics,
            predictions=predictions,
            aggregate=aggregate,
            uncertainty=uncertainty,
            decisions=decisions,
            checks=checks,
        )
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Unable to load verified Platinum package: {exc}") from exc
