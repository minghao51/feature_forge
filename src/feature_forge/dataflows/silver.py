"""Hamilton nodes for deterministic Silver canonicalization and fold policy.

Silver consumes the Bronze driver's outputs (``raw_dataset``,
``bronze_snapshot``, ``source_metadata``, ``bronze_materialization``) as
external inputs and never reloads the dataset source.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

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
    RunManifest,
    RunRequest,
    RunState,
    SilverMaterialization,
    StageResult,
)
from feature_forge.dataflows._io import (
    attach_staging_artifacts,
    check_result,
    existing_package,
)
from feature_forge.dataflows._io import (
    load_silver_package as load_silver_package,
)
from feature_forge.dataflows.hamilton_compat import cache, tag
from feature_forge.dataflows.profile import ExecutionProfile, get_profile_policy
from feature_forge.evaluation.holdout import assign_row_partitions
from feature_forge.exceptions import DatasetError
from feature_forge.storage.hashing import dataset_fingerprint, normalize_secret_free
from feature_forge.storage.local import LocalArtifactStore


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
    dataset_computation_request: DatasetComputationRequest,
) -> pd.DataFrame:
    """Generate exhaustive validation-fold assignments per Silver row.

    Rows are first split into a ``discovery`` partition (seen by feature
    selection) and an ``evaluation`` partition (used only for the reported CV
    score) when ``dataset_computation_request.evaluation_holdout_fraction > 0``. Folds are
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
        fraction=dataset_computation_request.evaluation_holdout_fraction,
        cv_folds=dataset_computation_request.cv_folds,
        stratified=dataset_computation_request.task == "classification",
        seed=dataset_computation_request.split_seed,
        y_for_stratify=(
            canonical_target if dataset_computation_request.task == "classification" else None
        ),
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
        if dataset_computation_request.task == "classification":
            splitter = StratifiedKFold(
                n_splits=dataset_computation_request.cv_folds,
                shuffle=True,
                random_state=dataset_computation_request.split_seed,
            )
            split_iterator = splitter.split(
                canonical_features.iloc[partition_idx],
                canonical_target.iloc[partition_idx],
            )
        else:
            splitter = KFold(
                n_splits=dataset_computation_request.cv_folds,
                shuffle=True,
                random_state=dataset_computation_request.split_seed,
            )
            split_iterator = splitter.split(canonical_features.iloc[partition_idx])
        try:
            for fold, (_, validation_indices) in enumerate(split_iterator):
                folds[partition_idx[validation_indices]] = fold
        except ValueError as exc:
            raise DatasetError(
                "Unable to create "
                f"{dataset_computation_request.cv_folds} folds in {partition_name}: {exc}"
            ) from exc

    if (folds < 0).any():
        raise DatasetError("Fold generation did not assign every Silver row")
    return pd.DataFrame({"row_id": row_ids_arr, "fold": folds, "partition": partition_labels})


@tag(layer="silver", cost="cheap", persistence="none", sensitivity="metadata", owner="data")
def dataset_fingerprint_value(
    bronze_snapshot: BronzeRecord,
    dataset_computation_request: DatasetComputationRequest,
) -> str:
    """Compute the stable identity of the canonical dataset and split policy."""
    split_strategy = (
        "stratified_kfold" if dataset_computation_request.task == "classification" else "kfold"
    )
    return dataset_fingerprint(
        source_checksum=bronze_snapshot.source_checksum,
        canonicalization_config={
            "config": dataset_computation_request.canonicalization_config,
        },
        schema_version=bronze_snapshot.schema_version,
        target_name=bronze_snapshot.target,
        task=dataset_computation_request.task,
        split_policy={
            "strategy": split_strategy,
            "folds": dataset_computation_request.cv_folds,
            "evaluation_holdout_fraction": (
                dataset_computation_request.evaluation_holdout_fraction
            ),
        },
        split_seed=dataset_computation_request.split_seed,
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
    dataset_computation_request: DatasetComputationRequest,
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
        and fold_assignments["fold"].nunique() == dataset_computation_request.cv_folds
        and not (fold_assignments["fold"] < 0).any()
    )
    # Discovery/evaluation partition: every row must carry a partition label,
    # and when the holdout is enabled both partitions must be non-empty and
    # each must independently cover all cv_folds so candidate selection
    # (discovery) and the reported aggregate (evaluation) never share a fold.
    holdout_enabled = dataset_computation_request.evaluation_holdout_fraction > 0.0
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
            == dataset_computation_request.cv_folds
            for part in ("discovery", "evaluation")
        )
    else:
        per_partition_folds_ok = (
            fold_assignments["fold"].nunique() == dataset_computation_request.cv_folds
        )
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
        check_result(
            "SILVER.ROW_ID.UNIQUE",
            rows_are_aligned,
            "Row IDs are unique and aligned",
            observed=len(set(row_ids)),
            expected=len(canonical_features),
            affected_artifacts=["row_ids.parquet"],
        ),
        check_result(
            "SILVER.TARGET.EXCLUDED",
            target_is_excluded,
            "Target is absent from canonical features",
            observed=list(map(str, canonical_features.columns)),
            expected=f"no '{bronze_snapshot.target}' column",
            affected_artifacts=["canonical_features.parquet"],
        ),
        check_result(
            "SILVER.TARGET.COMPLETE",
            target_is_complete,
            "Canonical target has no missing values",
            observed=int(canonical_target.isna().sum()),
            expected=0,
            affected_artifacts=["canonical_target.parquet"],
        ),
        check_result(
            "SILVER.SPLIT.NON_OVERLAP",
            folds_are_valid,
            "Fold assignments are exhaustive, ordered, and unique",
            affected_artifacts=["fold_assignments.parquet"],
        ),
        check_result(
            "SILVER.SCHEMA.VALID",
            schema_is_valid,
            "Canonical feature schema is non-empty with unique columns",
            affected_artifacts=["canonical_features.parquet"],
        ),
        check_result(
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
@cache(behavior="recompute")
def silver_manifest(
    dataset_request: DatasetRequest,
    bronze_snapshot: BronzeRecord,
    bronze_materialization: BronzeMaterialization,
    source_metadata: dict[str, Any],
    dataset_fingerprint_value: str,
    silver_checks: list[CheckResult],
    environment_snapshot: EnvironmentSnapshot,
) -> RunManifest:
    """Build the per-attempt Silver manifest metadata before any commit."""
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
        reuse_fingerprint=dataset_fingerprint_value,
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
@cache(behavior="recompute")
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
    existing = existing_package(artifact_store, namespace, silver_manifest)
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
    staging.set_manifest(attach_staging_artifacts(silver_manifest, staging))
    manifest_ref = artifact_store.commit(staging)
    return SilverMaterialization(
        manifest=cast(RunManifest, staging.manifest),
        manifest_ref=manifest_ref,
        persisted=True,
        artifact_paths=[descriptor.relative_path for descriptor in staging.artifacts],
    )
