"""Hamilton nodes for deterministic Bronze and Silver preparation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from feature_forge.contracts import (
    ArtifactNamespace,
    ArtifactRef,
    BronzeMaterialization,
    BronzeRecord,
    CheckResult,
    DatasetRequest,
    EnvironmentSnapshot,
    FailureClass,
    Layer,
    ManifestRef,
    RunManifest,
    RunRequest,
    RunState,
    SilverMaterialization,
    SilverPackage,
    StageResult,
    bronze_fingerprint,
)
from feature_forge.dataflows._io import package_load_guard
from feature_forge.dataflows.hamilton_compat import tag
from feature_forge.dataflows.profile import ExecutionProfile, get_profile_policy
from feature_forge.evaluation.holdout import assign_row_partitions
from feature_forge.exceptions import DatasetError
from feature_forge.storage.hashing import (
    dataset_fingerprint,
    fingerprint,
    normalize_secret_free,
)
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


def _check(
    check_id: str,
    passed: bool,
    message: str,
    *,
    observed: Any | None = None,
    expected: Any | None = None,
    affected_artifacts: list[str] | None = None,
    **details: Any,
) -> CheckResult:
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


def _with_artifacts(manifest: RunManifest, staging: LocalStagingArea) -> RunManifest:
    artifacts = staging.artifacts
    stages = [
        stage.model_copy(update={"artifacts": artifacts})
        if stage.stage in {"bronze", "silver"}
        else stage
        for stage in manifest.stages
    ]
    return manifest.model_copy(
        update={
            "artifacts": artifacts,
            "stages": stages,
            "completed_at": datetime.now(UTC),
        }
    )


def _existing_package(
    artifact_store: LocalArtifactStore,
    namespace: ArtifactNamespace,
    expected_manifest: RunManifest,
) -> tuple[RunManifest, ManifestRef] | None:
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


@tag(layer="bronze", cost="io", persistence="boundary", sensitivity="dataset-derived", owner="data")
def raw_dataset(
    dataset_request: DatasetRequest,
    dataset_registry: Any,
    execution_profile: ExecutionProfile,
) -> dict[str, Any]:
    """Resolve a dataset while enforcing the profile's network policy."""
    raw_info = dataset_registry.info(dataset_request.name)
    if not isinstance(raw_info, dict):
        raise DatasetError(f"Dataset '{dataset_request.name}' has invalid registry metadata")
    source = str(raw_info.get("source", "unknown"))
    policy = get_profile_policy(execution_profile)
    if not policy.network_allowed and source not in {"local", "memory", "fixture"}:
        raise DatasetError(
            f"Profile '{execution_profile.value}' forbids network-backed dataset source '{source}'"
        )

    loaded = dataset_registry.load(dataset_request.name)
    if not isinstance(loaded, dict):
        raise DatasetError(f"Dataset '{dataset_request.name}' loader returned an invalid payload")
    train = loaded.get("train")
    if not isinstance(train, pd.DataFrame) or train.empty:
        raise DatasetError(f"Dataset '{dataset_request.name}' has no non-empty training DataFrame")

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
    dataset_request: DatasetRequest,
) -> BronzeRecord:
    """Record source identity and metadata before canonicalization."""
    train = raw_dataset["train"]
    if not isinstance(train, pd.DataFrame):
        raise DatasetError("Bronze input train value is not a DataFrame")
    metadata = dict(raw_dataset.get("metadata") or {})
    target = dataset_request.target or raw_dataset.get("target") or metadata.get("target")
    if not isinstance(target, str) or not target:
        raise DatasetError(f"Dataset '{dataset_request.name}' has no declared target")
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
            "task": dataset_request.task,
        }
    )
    return BronzeRecord(
        source=source,
        source_checksum=source_checksum,
        target=target,
        task=dataset_request.task,
        snapshot_mode=dataset_request.source_policy,
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
        _check(
            "BRONZE.SOURCE.READABLE",
            bronze_snapshot.row_count > 0,
            "Source contains readable training rows",
            observed=bronze_snapshot.row_count,
            expected="> 0",
        ),
        _check(
            "BRONZE.SOURCE.CHECKSUM",
            checksum_valid,
            "Source checksum is a SHA-256 digest",
            observed=bronze_snapshot.source_checksum,
            expected="64 lowercase hexadecimal characters",
        ),
        _check(
            "BRONZE.TARGET.DECLARED",
            bool(bronze_snapshot.target),
            "Target metadata is declared",
            observed=bronze_snapshot.target,
            expected="non-empty target column",
        ),
    ]


@tag(layer="bronze", cost="cheap", persistence="none", sensitivity="metadata", owner="verification")
def bronze_manifest(
    dataset_request: DatasetRequest,
    bronze_snapshot: BronzeRecord,
    source_metadata: dict[str, Any],
    bronze_checks: list[CheckResult],
    environment_snapshot: EnvironmentSnapshot,
) -> RunManifest:
    """Build the Bronze manifest before reference or snapshot materialization."""
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
def bronze_materialization(
    bronze_manifest: RunManifest,
    bronze_snapshot: BronzeRecord,
    bronze_checks: list[CheckResult],
    raw_dataset: dict[str, Any],
    dataset_request: DatasetRequest,
    execution_profile: ExecutionProfile,
    artifact_store: LocalArtifactStore | None,
) -> BronzeMaterialization:
    """Persist a Bronze reference or physical source snapshot."""
    failed = [check.check_id for check in bronze_checks if check.required and not check.passed]
    if failed:
        raise DatasetError(f"Bronze checks failed: {', '.join(failed)}")

    policy = get_profile_policy(execution_profile)
    if not policy.persistence_enabled or artifact_store is None:
        return BronzeMaterialization(manifest=bronze_manifest, persisted=False)

    namespace = ArtifactNamespace(layer=Layer.BRONZE, run_id=dataset_request.run_id)
    existing = _existing_package(artifact_store, namespace, bronze_manifest)
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
    staging.set_manifest(_with_artifacts(bronze_manifest, staging))
    manifest_ref = artifact_store.commit(staging)
    return BronzeMaterialization(
        manifest=cast(RunManifest, staging.manifest),
        manifest_ref=manifest_ref,
        persisted=True,
        artifact_paths=[descriptor.relative_path for descriptor in staging.artifacts],
    )


@tag(
    layer="silver",
    cost="cheap",
    persistence="boundary",
    sensitivity="dataset-derived",
    owner="data",
)
def canonical_features(
    raw_dataset: dict[str, Any],
    bronze_snapshot: BronzeRecord,
) -> pd.DataFrame:
    """Remove the declared target and preserve the source row order."""
    train = raw_dataset["train"]
    if not isinstance(train, pd.DataFrame):
        raise DatasetError("Silver input train value is not a DataFrame")
    features = train.drop(columns=[bronze_snapshot.target]).reset_index(drop=True)
    return features


@tag(layer="silver", cost="cheap", persistence="boundary", sensitivity="target", owner="data")
def canonical_target(
    raw_dataset: dict[str, Any],
    bronze_snapshot: BronzeRecord,
) -> pd.Series:
    """Return the target aligned to canonical feature row order."""
    train = raw_dataset["train"]
    if not isinstance(train, pd.DataFrame):
        raise DatasetError("Silver input train value is not a DataFrame")
    return train[bronze_snapshot.target].reset_index(drop=True)


@tag(
    layer="silver",
    cost="cheap",
    persistence="boundary",
    sensitivity="dataset-derived",
    owner="data",
)
def row_ids(canonical_features: pd.DataFrame) -> list[str]:
    """Create stable, non-null row IDs from canonical row position."""
    return [f"row-{position:09d}" for position in range(len(canonical_features))]


@tag(
    layer="silver",
    cost="cheap",
    persistence="boundary",
    sensitivity="dataset-derived",
    owner="data",
)
def fold_assignments(
    canonical_features: pd.DataFrame,
    canonical_target: pd.Series,
    row_ids: list[str],
    dataset_request: DatasetRequest,
) -> pd.DataFrame:
    """Generate exhaustive validation-fold assignments per Silver row.

    Rows are first split into a ``discovery`` partition (seen by feature
    selection) and an ``evaluation`` partition (used only for the reported CV
    score) when ``dataset_request.evaluation_holdout_fraction > 0``. Folds are
    then generated *within each partition*, so each partition independently
    covers folds ``0..cv_folds-1``. When the data is too small to host the
    holdout, every row lands in the ``discovery`` partition (and selection runs
    on all rows, matching the pre-holdout behavior).
    """
    if len(canonical_features) != len(canonical_target) or len(row_ids) != len(canonical_features):
        raise DatasetError("Silver features, target, and row IDs are misaligned")
    if canonical_target.isna().any():
        raise DatasetError("Silver target contains missing values")

    n_rows = len(row_ids)
    partition_labels = assign_row_partitions(
        n_rows,
        fraction=dataset_request.evaluation_holdout_fraction,
        cv_folds=dataset_request.cv_folds,
        stratified=dataset_request.task == "classification",
        seed=dataset_request.split_seed,
        y_for_stratify=canonical_target if dataset_request.task == "classification" else None,
    )

    row_ids_arr = np.asarray(row_ids, dtype=object)
    folds = np.full(n_rows, -1, dtype=np.int64)
    # Generate folds independently within each partition so candidate evidence
    # (discovery) and the reported aggregate (evaluation) never share a fold.
    for partition_name in ("discovery", "evaluation"):
        mask = partition_labels == partition_name
        if not mask.any():
            continue
        partition_idx = np.flatnonzero(mask)
        if dataset_request.task == "classification":
            splitter = StratifiedKFold(
                n_splits=dataset_request.cv_folds,
                shuffle=True,
                random_state=dataset_request.split_seed,
            )
            split_iterator = splitter.split(
                canonical_features.iloc[partition_idx],
                canonical_target.iloc[partition_idx],
            )
        else:
            splitter = KFold(
                n_splits=dataset_request.cv_folds,
                shuffle=True,
                random_state=dataset_request.split_seed,
            )
            split_iterator = splitter.split(canonical_features.iloc[partition_idx])
        try:
            for fold, (_, validation_indices) in enumerate(split_iterator):
                folds[partition_idx[validation_indices]] = fold
        except ValueError as exc:
            raise DatasetError(
                f"Unable to create {dataset_request.cv_folds} folds in {partition_name}: {exc}"
            ) from exc

    if (folds < 0).any():
        raise DatasetError("Fold generation did not assign every Silver row")
    return pd.DataFrame({"row_id": row_ids_arr, "fold": folds, "partition": partition_labels})


@tag(layer="silver", cost="cheap", persistence="none", sensitivity="metadata", owner="data")
def dataset_fingerprint_value(
    bronze_snapshot: BronzeRecord,
    dataset_request: DatasetRequest,
) -> str:
    """Compute the stable identity of the canonical dataset and split policy."""
    split_strategy = "stratified_kfold" if dataset_request.task == "classification" else "kfold"
    return dataset_fingerprint(
        source_checksum=bronze_snapshot.source_checksum,
        canonicalization_config={
            "config": dataset_request.canonicalization_config,
        },
        schema_version=bronze_snapshot.schema_version,
        target_name=bronze_snapshot.target,
        task=dataset_request.task,
        split_policy={
            "strategy": split_strategy,
            "folds": dataset_request.cv_folds,
            "evaluation_holdout_fraction": dataset_request.evaluation_holdout_fraction,
        },
        split_seed=dataset_request.split_seed,
    )


@tag(
    layer="silver",
    cost="cheap",
    persistence="boundary",
    sensitivity="dataset-derived",
    owner="data",
)
def dataset_profile(
    canonical_features: pd.DataFrame,
    canonical_target: pd.Series,
    bronze_snapshot: BronzeRecord,
    dataset_fingerprint_value: str,
) -> dict[str, Any]:
    """Create a deterministic, secret-free profile for the canonical dataset."""
    return {
        "dataset_fingerprint": dataset_fingerprint_value,
        "row_count": len(canonical_features),
        "feature_count": len(canonical_features.columns),
        "target": bronze_snapshot.target,
        "task": bronze_snapshot.task,
        "target_missing": int(canonical_target.isna().sum()),
        "missing_by_column": {
            str(column): int(canonical_features[column].isna().sum())
            for column in canonical_features.columns
        },
        "cardinality_by_column": {
            str(column): int(canonical_features[column].nunique(dropna=False))
            for column in canonical_features.columns
        },
        "schema": {str(column): str(dtype) for column, dtype in canonical_features.dtypes.items()},
    }


@tag(
    layer="silver",
    cost="cheap",
    persistence="none",
    sensitivity="dataset-derived",
    owner="verification",
)
def silver_checks(
    canonical_features: pd.DataFrame,
    canonical_target: pd.Series,
    row_ids: list[str],
    fold_assignments: pd.DataFrame,
    bronze_snapshot: BronzeRecord,
    dataset_request: DatasetRequest,
) -> list[CheckResult]:
    """Run the required Silver boundary checks."""
    target_is_excluded = bronze_snapshot.target not in canonical_features.columns
    target_proxy_columns = [
        str(column)
        for column in canonical_features.columns
        if canonical_features[column].equals(canonical_target)
    ]
    rows_are_aligned = (
        len(row_ids) == len(canonical_features) == len(canonical_target)
        and len(set(row_ids)) == len(row_ids)
        and all(isinstance(row_id, str) and bool(row_id) for row_id in row_ids)
    )
    folds_are_valid = (
        {"row_id", "fold", "partition"}.issubset(fold_assignments.columns)
        and len(fold_assignments) == len(row_ids)
        and fold_assignments["row_id"].is_unique
        and fold_assignments["row_id"].tolist() == row_ids
        and fold_assignments["fold"].nunique() == dataset_request.cv_folds
        and not (fold_assignments["fold"] < 0).any()
    )
    # Discovery/evaluation partition: every row must carry a partition label,
    # and when the holdout is enabled both partitions must be non-empty and
    # each must independently cover all cv_folds so candidate selection
    # (discovery) and the reported aggregate (evaluation) never share a fold.
    holdout_enabled = dataset_request.evaluation_holdout_fraction > 0.0
    partition_col = fold_assignments.get("partition")
    if isinstance(partition_col, pd.Series):
        partitions_present = set(partition_col.unique().tolist())
    else:
        partitions_present = set()
    partition_is_valid = partitions_present <= {"discovery", "evaluation"} and (
        partitions_present == {"discovery"}
        if not holdout_enabled
        else partitions_present == {"discovery", "evaluation"}
    )
    if holdout_enabled and partition_is_valid:
        per_partition_folds_ok = all(
            fold_assignments.loc[fold_assignments["partition"] == part, "fold"].nunique()
            == dataset_request.cv_folds
            for part in ("discovery", "evaluation")
        )
    else:
        per_partition_folds_ok = fold_assignments["fold"].nunique() == dataset_request.cv_folds
    folds_are_valid = folds_are_valid and partition_is_valid and per_partition_folds_ok
    string_columns = [str(column) for column in canonical_features.columns]
    schema_is_valid = (
        bool(len(canonical_features.columns))
        and all(isinstance(column, str) for column in canonical_features.columns)
        and len(set(string_columns)) == len(string_columns)
        and not canonical_features.columns.duplicated().any()
    )
    target_is_complete = not canonical_target.isna().any()
    return [
        _check(
            "SILVER.ROW_ID.UNIQUE",
            rows_are_aligned,
            "Row IDs are unique and aligned",
            observed=len(set(row_ids)),
            expected=len(canonical_features),
            affected_artifacts=["row_ids.parquet"],
        ),
        _check(
            "SILVER.TARGET.EXCLUDED",
            target_is_excluded,
            "Target is absent from canonical features",
            observed=list(map(str, canonical_features.columns)),
            expected=f"no '{bronze_snapshot.target}' column",
            affected_artifacts=["canonical_features.parquet"],
        ),
        _check(
            "SILVER.TARGET.COMPLETE",
            target_is_complete,
            "Canonical target has no missing values",
            observed=int(canonical_target.isna().sum()),
            expected=0,
            affected_artifacts=["canonical_target.parquet"],
        ),
        _check(
            "SILVER.SPLIT.NON_OVERLAP",
            folds_are_valid,
            "Fold assignments are exhaustive, ordered, and unique",
            affected_artifacts=["fold_assignments.parquet"],
        ),
        _check(
            "SILVER.SCHEMA.VALID",
            schema_is_valid,
            "Canonical feature schema is non-empty with unique columns",
            affected_artifacts=["canonical_features.parquet"],
        ),
        _check(
            "SILVER.LEAKAGE.NO_TARGET_PROXY",
            not target_proxy_columns,
            "No feature column exactly duplicates the target",
            observed=target_proxy_columns,
            expected=[],
            affected_artifacts=["canonical_features.parquet", "canonical_target.parquet"],
            proxy_columns=target_proxy_columns,
        ),
    ]


@tag(layer="silver", cost="cheap", persistence="none", sensitivity="metadata", owner="verification")
def silver_manifest(
    dataset_request: DatasetRequest,
    bronze_snapshot: BronzeRecord,
    bronze_materialization: BronzeMaterialization,
    source_metadata: dict[str, Any],
    dataset_fingerprint_value: str,
    silver_checks: list[CheckResult],
    environment_snapshot: EnvironmentSnapshot,
) -> RunManifest:
    """Build the Silver manifest metadata before any package is committed."""
    checks_passed = all(check.passed for check in silver_checks if check.required)
    state = RunState.SUCCEEDED if checks_passed else RunState.FAILED
    now = datetime.now(UTC)
    upstream = (
        [bronze_materialization.manifest_ref]
        if bronze_materialization.manifest_ref is not None
        else []
    )
    return RunManifest(
        schema_version="1",
        layer=Layer.SILVER,
        package_kind="silver",
        layer_fingerprint=dataset_fingerprint_value,
        run_id=dataset_request.run_id,
        case_fingerprint=dataset_request.case_fingerprint,
        state=state,
        request=RunRequest(
            dataset=dataset_request.name,
            method="silver",
            model="none",
            seed=dataset_request.split_seed,
            cv_folds=dataset_request.cv_folds,
            options={
                "dataset_fingerprint": dataset_fingerprint_value,
                "source": normalize_secret_free(source_metadata),
                "bronze": normalize_secret_free(bronze_snapshot.model_dump(mode="json")),
            },
        ),
        environment=environment_snapshot,
        upstream_manifests=upstream,
        stages=[
            StageResult(
                stage="silver",
                state=state,
                started_at=now,
                finished_at=now,
                checks=silver_checks,
                failure_class=None if checks_passed else FailureClass.DETERMINISTIC,
                error_message=None if checks_passed else "Required Silver checks failed",
            )
        ],
        created_at=now,
        completed_at=now if checks_passed else None,
    )


@tag(
    layer="silver",
    cost="io",
    persistence="boundary",
    sensitivity="dataset-derived",
    owner="storage",
)
def silver_materialization(
    silver_manifest: RunManifest,
    bronze_snapshot: BronzeRecord,
    canonical_features: pd.DataFrame,
    canonical_target: pd.Series,
    row_ids: list[str],
    fold_assignments: pd.DataFrame,
    dataset_profile: dict[str, Any],
    silver_checks: list[CheckResult],
    dataset_request: DatasetRequest,
    execution_profile: ExecutionProfile,
    artifact_store: LocalArtifactStore | None,
) -> SilverMaterialization:
    """Persist a verified Silver package or return a side-effect-free dry-run."""
    failed = [check.check_id for check in silver_checks if check.required and not check.passed]
    if failed:
        raise DatasetError(f"Silver checks failed: {', '.join(failed)}")

    policy = get_profile_policy(execution_profile)
    if not policy.persistence_enabled or artifact_store is None:
        return SilverMaterialization(manifest=silver_manifest, persisted=False)

    namespace = ArtifactNamespace(layer=Layer.SILVER, run_id=dataset_request.run_id)
    existing = _existing_package(artifact_store, namespace, silver_manifest)
    if existing is not None:
        manifest, manifest_ref = existing
        return SilverMaterialization(
            manifest=manifest,
            manifest_ref=manifest_ref,
            persisted=True,
            artifact_paths=[artifact.relative_path for artifact in manifest.artifacts],
        )

    staging = artifact_store.begin(namespace)
    staging.write_dataframe("canonical_features.parquet", canonical_features)
    staging.write_dataframe(
        "canonical_target.parquet",
        pd.DataFrame({"row_id": row_ids, bronze_snapshot.target: canonical_target.to_numpy()}),
    )
    staging.write_dataframe("row_ids.parquet", pd.DataFrame({"row_id": row_ids}))
    staging.write_dataframe("fold_assignments.parquet", fold_assignments)
    staging.write_json("profile.json", dataset_profile)
    staging.write_json("checks.json", [check.model_dump(mode="json") for check in silver_checks])
    staging.set_manifest(_with_artifacts(silver_manifest, staging))
    manifest_ref = artifact_store.commit(staging)
    return SilverMaterialization(
        manifest=cast(RunManifest, staging.manifest),
        manifest_ref=manifest_ref,
        persisted=True,
        artifact_paths=[descriptor.relative_path for descriptor in staging.artifacts],
    )


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
