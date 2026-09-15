"""Hamilton nodes for the Bronze source-ingestion boundary.

The Bronze driver resolves the declared dataset exactly once per case attempt,
records the source identity, and persists a reference or physical snapshot.
Silver consumes the Bronze outputs as external inputs and never reloads the
source.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd

from feature_forge.contracts import (
    ArtifactNamespace,
    BronzeMaterialization,
    BronzeRecord,
    CheckResult,
    DatasetComputationRequest,
    DatasetRequest,
    EnvironmentSnapshot,
    FailureClass,
    Layer,
    ResumePolicy,
    RunManifest,
    RunRequest,
    RunState,
    StageResult,
    bronze_fingerprint,
)
from feature_forge.dataflows._io import (
    attach_staging_artifacts,
    check_result,
    existing_package,
)
from feature_forge.dataflows.hamilton_compat import cache, tag
from feature_forge.dataflows.profile import ExecutionProfile, get_profile_policy
from feature_forge.exceptions import DatasetError
from feature_forge.storage.hashing import fingerprint, normalize_secret_free
from feature_forge.storage.local import LocalArtifactStore


def _frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a deterministic payload that preserves float-level value changes."""
    normalized = frame.reset_index(drop=True)
    try:
        value_hashes = pd.util.hash_pandas_object(
            normalized,
            index=False,
            categorize=False,
        ).astype("uint64")
        values: Any = [int(value) for value in value_hashes]
    except TypeError:
        values = normalized.to_json(
            orient="split",
            date_format="iso",
            double_precision=15,
            default_handler=str,
        )
    return {
        "columns": [str(column) for column in normalized.columns],
        "dtypes": [str(dtype) for dtype in normalized.dtypes],
        "values": values,
    }


@tag(layer="bronze", cost="cheap", persistence="none", sensitivity="metadata", owner="data")
@cache(behavior="recompute")
def dataset_computation_request(dataset_request: DatasetRequest) -> DatasetComputationRequest:
    """Strip attempt fields before values enter pure/cacheable nodes."""
    return DatasetComputationRequest.model_validate(
        dataset_request.model_dump(exclude={"run_id", "case_fingerprint"})
    )


@tag(layer="bronze", cost="io", persistence="boundary", sensitivity="dataset-derived", owner="data")
@cache(behavior="recompute")
def raw_dataset(
    dataset_computation_request: DatasetComputationRequest,
    dataset_registry: Any,
    execution_profile: ExecutionProfile,
) -> dict[str, Any]:
    """Resolve a dataset while enforcing the profile's network policy."""
    raw_info = dataset_registry.info(dataset_computation_request.name)
    if not isinstance(raw_info, dict):
        raise DatasetError(
            f"Dataset '{dataset_computation_request.name}' has invalid registry metadata"
        )
    source = str(raw_info.get("source", "unknown"))
    policy = get_profile_policy(execution_profile)
    if not policy.network_allowed and source not in {"local", "memory", "fixture"}:
        raise DatasetError(
            f"Profile '{execution_profile.value}' forbids network-backed dataset source '{source}'"
        )

    loaded = dataset_registry.load(dataset_computation_request.name)
    if not isinstance(loaded, dict):
        raise DatasetError(
            f"Dataset '{dataset_computation_request.name}' loader returned an invalid payload"
        )
    train = loaded.get("train")
    if not isinstance(train, pd.DataFrame) or train.empty:
        raise DatasetError(
            f"Dataset '{dataset_computation_request.name}' has no non-empty training DataFrame"
        )

    registry_metadata = {
        str(key): value for key, value in raw_info.items() if not str(key).startswith("_")
    }
    loaded_metadata = loaded.get("metadata")
    if isinstance(loaded_metadata, dict):
        registry_metadata.update(loaded_metadata)
    registry_metadata["source"] = source
    return cast(dict[str, Any], {**loaded, "metadata": registry_metadata})


@tag(
    layer="bronze",
    cost="cheap",
    persistence="boundary",
    sensitivity="dataset-derived",
    owner="data",
)
def bronze_snapshot(
    raw_dataset: dict[str, Any],
    dataset_computation_request: DatasetComputationRequest,
) -> BronzeRecord:
    """Record source identity and metadata before canonicalization."""
    train = raw_dataset["train"]
    if not isinstance(train, pd.DataFrame):
        raise DatasetError("Bronze input train value is not a DataFrame")
    metadata = dict(raw_dataset.get("metadata") or {})
    target = (
        dataset_computation_request.target or raw_dataset.get("target") or metadata.get("target")
    )
    if not isinstance(target, str) or not target:
        raise DatasetError(f"Dataset '{dataset_computation_request.name}' has no declared target")
    target_count = list(train.columns).count(target)
    if target_count != 1:
        raise DatasetError(
            f"Target column '{target}' must appear exactly once; found {target_count}"
        )
    source = str(metadata.get("source", "unknown"))
    source_checksum = fingerprint(
        {
            "train": _frame_payload(train),
            "test": _frame_payload(raw_dataset["test"])
            if isinstance(raw_dataset.get("test"), pd.DataFrame)
            else None,
            "target": target,
            "task": dataset_computation_request.task,
        }
    )
    return BronzeRecord(
        source=source,
        source_checksum=source_checksum,
        target=target,
        task=dataset_computation_request.task,
        snapshot_mode=dataset_computation_request.source_policy,
        row_count=len(train),
        column_count=len(train.columns),
        metadata=metadata,
    )


@tag(layer="bronze", cost="cheap", persistence="none", sensitivity="metadata", owner="data")
def source_metadata(bronze_snapshot: BronzeRecord) -> dict[str, Any]:
    """Expose stable source metadata as a small queryable Hamilton output."""
    return {
        "source": bronze_snapshot.source,
        "source_checksum": bronze_snapshot.source_checksum,
        "snapshot_mode": bronze_snapshot.snapshot_mode,
        "row_count": bronze_snapshot.row_count,
        "column_count": bronze_snapshot.column_count,
    }


@tag(layer="bronze", cost="cheap", persistence="none", sensitivity="metadata", owner="verification")
def bronze_checks(bronze_snapshot: BronzeRecord) -> list[CheckResult]:
    """Run the required checks for a reproducible Bronze boundary."""
    checksum_valid = len(bronze_snapshot.source_checksum) == 64 and all(
        character in "0123456789abcdef" for character in bronze_snapshot.source_checksum
    )
    return [
        check_result(
            "BRONZE.SOURCE.READABLE",
            bronze_snapshot.row_count > 0,
            "Source contains readable training rows",
            observed=bronze_snapshot.row_count,
            expected="> 0",
        ),
        check_result(
            "BRONZE.SOURCE.CHECKSUM",
            checksum_valid,
            "Source checksum is a SHA-256 digest",
            observed=bronze_snapshot.source_checksum,
            expected="64 lowercase hexadecimal characters",
        ),
        check_result(
            "BRONZE.TARGET.DECLARED",
            bool(bronze_snapshot.target),
            "Target metadata is declared",
            observed=bronze_snapshot.target,
            expected="non-empty target column",
        ),
    ]


@tag(layer="bronze", cost="cheap", persistence="none", sensitivity="metadata", owner="verification")
@cache(behavior="recompute")
def bronze_manifest(
    dataset_request: DatasetRequest,
    bronze_snapshot: BronzeRecord,
    source_metadata: dict[str, Any],
    bronze_checks: list[CheckResult],
    environment_snapshot: EnvironmentSnapshot,
) -> RunManifest:
    """Build the per-attempt Bronze manifest before materialization."""
    checks_passed = all(check.passed for check in bronze_checks if check.required)
    state = RunState.SUCCEEDED if checks_passed else RunState.FAILED
    now = datetime.now(UTC)
    return RunManifest(
        schema_version="1",
        layer=Layer.BRONZE,
        package_kind="bronze",
        layer_fingerprint=bronze_fingerprint(
            source_checksum=bronze_snapshot.source_checksum,
            source_contract_version=bronze_snapshot.schema_version,
        ),
        reuse_fingerprint=bronze_fingerprint(
            source_checksum=bronze_snapshot.source_checksum,
            source_contract_version=bronze_snapshot.schema_version,
        ),
        run_id=dataset_request.run_id,
        case_fingerprint=dataset_request.case_fingerprint,
        state=state,
        request=RunRequest(
            dataset=dataset_request.name,
            method="bronze",
            model="none",
            seed=dataset_request.split_seed,
            options={
                "source": normalize_secret_free(source_metadata),
                "source_policy": dataset_request.source_policy,
            },
        ),
        environment=environment_snapshot,
        stages=[
            StageResult(
                stage="bronze",
                state=state,
                started_at=now,
                finished_at=now,
                checks=bronze_checks,
                failure_class=None if checks_passed else FailureClass.DETERMINISTIC,
                error_message=None if checks_passed else "Required Bronze checks failed",
            )
        ],
        created_at=now,
        completed_at=now if checks_passed else None,
    )


@tag(
    layer="bronze",
    cost="io",
    persistence="boundary",
    sensitivity="dataset-derived",
    owner="storage",
)
@cache(behavior="recompute")
def bronze_materialization(
    bronze_manifest: RunManifest,
    bronze_snapshot: BronzeRecord,
    bronze_checks: list[CheckResult],
    raw_dataset: dict[str, Any],
    dataset_request: DatasetRequest,
    execution_profile: ExecutionProfile,
    artifact_store: LocalArtifactStore | None,
    resume_policy: ResumePolicy | None = None,
) -> BronzeMaterialization:
    """Persist a Bronze reference or physical source snapshot."""
    failed = [check.check_id for check in bronze_checks if check.required and not check.passed]
    if failed:
        raise DatasetError(f"Bronze checks failed: {', '.join(failed)}")

    policy = get_profile_policy(execution_profile)
    if not policy.persistence_enabled or artifact_store is None:
        return BronzeMaterialization(manifest=bronze_manifest, persisted=False)

    namespace = ArtifactNamespace(layer=Layer.BRONZE, run_id=dataset_request.run_id)
    if resume_policy is not None and resume_policy.enabled:
        reusable = artifact_store.find_reusable(
            bronze_manifest.reuse_fingerprint or bronze_manifest.layer_fingerprint,
            Layer.BRONZE,
            preferred_run_id=dataset_request.run_id,
            allow_cross_run=resume_policy.allow_cross_run_reuse,
        )
        if reusable is not None:
            manifest = artifact_store.load_manifest(reusable)
            return BronzeMaterialization(
                manifest=manifest,
                manifest_ref=reusable,
                persisted=True,
                artifact_paths=[artifact.relative_path for artifact in manifest.artifacts],
            )
    existing = existing_package(artifact_store, namespace, bronze_manifest)
    if existing is not None:
        manifest, manifest_ref = existing
        return BronzeMaterialization(
            manifest=manifest,
            manifest_ref=manifest_ref,
            persisted=True,
            artifact_paths=[artifact.relative_path for artifact in manifest.artifacts],
        )

    staging = artifact_store.begin(namespace)
    staging.write_json("bronze.json", bronze_snapshot.model_dump(mode="json"))
    if dataset_request.source_policy == "snapshot":
        staging.write_dataframe("train.parquet", cast(pd.DataFrame, raw_dataset["train"]))
        test = raw_dataset.get("test")
        if isinstance(test, pd.DataFrame) and not test.empty:
            staging.write_dataframe("test.parquet", test, required=False)
    else:
        staging.write_json(
            "source_reference.json",
            {
                "source": bronze_snapshot.source,
                "source_checksum": bronze_snapshot.source_checksum,
                "metadata": bronze_snapshot.metadata,
            },
        )
    staging.set_manifest(attach_staging_artifacts(bronze_manifest, staging))
    manifest_ref = artifact_store.commit(staging)
    return BronzeMaterialization(
        manifest=cast(RunManifest, staging.manifest),
        manifest_ref=manifest_ref,
        persisted=True,
        artifact_paths=[descriptor.relative_path for descriptor in staging.artifacts],
    )
