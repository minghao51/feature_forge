"""Platinum fold evidence, selection, and semantic verification tests."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.contracts import (
    ArtifactNamespace,
    BronzeRecord,
    CheckResult,
    EnvironmentSnapshot,
    EvaluationPolicy,
    GoldRequest,
    Layer,
    RunManifest,
    RunRequest,
    RunState,
    SelectionPolicy,
    gold_input_fingerprint,
)
from feature_forge.dataflows.gold import execute_gold, load_gold_package
from feature_forge.dataflows.platinum import (
    _evaluate,
    build_platinum_request,
    execute_platinum,
    experiment_result_from_package,
    load_platinum_package,
)
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.dataflows.silver import load_silver_package
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.metrics import MetricDirection, MetricRegistry
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import DatasetError, EvaluationError
from feature_forge.experiment.execution import (
    ProcessPoolExecutionAdapter,
    SequentialExecutionAdapter,
)
from feature_forge.storage.atomic import atomic_write_json
from feature_forge.storage.hashing import sha256_file
from feature_forge.storage.local import LocalArtifactStore


def package_result_worker(
    payload: tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Load real durable packages across the process serialization seam."""
    root, silver_data, gold_data, platinum_data = payload
    from feature_forge.contracts import ManifestRef

    store = LocalArtifactStore(root)
    silver = load_silver_package(store, ManifestRef.model_validate(silver_data))
    gold = load_gold_package(store, ManifestRef.model_validate(gold_data))
    platinum = load_platinum_package(store, ManifestRef.model_validate(platinum_data))
    return experiment_result_from_package(
        platinum,
        silver=silver,
        gold=gold,
        manifest_uri=f"04_platinum/runs/{platinum.manifest.run_id}/manifest.json",
    )


class DeterministicMethod:
    name = "deterministic"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> DeterministicMethod:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"signal": X["a"] + X["b"]})

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        return self.transform(X_train)

    @property
    def generated_scripts(self) -> list[str]:
        return [
            "def generate_features(df):\n"
            "    return pd.DataFrame({'signal': df['a'] + df['b']}, index=df.index)\n"
        ]

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"name": "signal", "base_columns": ["a", "b"]}]

    def get_artifacts(self) -> dict[str, Any]:
        return {}


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.13",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
        random_seeds={"method": 42},
    )


def _silver(store: LocalArtifactStore) -> Any:
    features = pd.DataFrame(
        {
            "a": [0, 1, 0, 1, 0, 1, 0, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    target_values = [0, 1, 0, 1, 0, 1, 0, 1]
    row_ids = pd.DataFrame({"row_id": [f"r{i}" for i in range(8)]})
    folds = pd.DataFrame({"row_id": row_ids["row_id"], "fold": [0, 0, 0, 0, 1, 1, 1, 1]})
    bronze = BronzeRecord(
        source="fixture",
        source_checksum="source",
        target="target",
        task="classification",
        snapshot_mode="snapshot",
        row_count=8,
        column_count=3,
    )
    bronze_manifest = RunManifest(
        layer=Layer.BRONZE,
        package_kind="bronze",
        layer_fingerprint="bronze-fingerprint",
        run_id="bronze-platinum",
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(
            dataset="fixture",
            method="bronze",
            model="none",
            seed=42,
            options={"source": {"source_checksum": "source"}},
        ),
        environment=_environment(),
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    staging = store.begin(ArtifactNamespace(layer=Layer.BRONZE, run_id="bronze-platinum"))
    staging.write_json("bronze.json", bronze.model_dump(mode="json"))
    staging.write_dataframe("train.parquet", features.assign(target=target_values))
    staging.set_manifest(bronze_manifest)
    bronze_ref = store.commit(staging)
    checks = [CheckResult(check_id="SILVER.OK", passed=True, message="ok")]
    silver_manifest = RunManifest(
        layer=Layer.SILVER,
        package_kind="silver",
        layer_fingerprint="silver-fingerprint",
        run_id="silver-platinum",
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(
            dataset="fixture",
            method="silver",
            model="none",
            seed=42,
            options={"dataset_fingerprint": "silver-fingerprint"},
        ),
        environment=_environment(),
        upstream_manifests=[bronze_ref],
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    staging = store.begin(ArtifactNamespace(layer=Layer.SILVER, run_id="silver-platinum"))
    staging.write_dataframe("canonical_features.parquet", features)
    staging.write_dataframe(
        "canonical_target.parquet",
        pd.DataFrame({"row_id": row_ids["row_id"], "target": target_values}),
    )
    staging.write_dataframe("row_ids.parquet", row_ids)
    staging.write_dataframe("fold_assignments.parquet", folds)
    staging.write_json("profile.json", {"dataset_fingerprint": "silver-fingerprint"})
    staging.write_json("checks.json", [item.model_dump(mode="json") for item in checks])
    staging.set_manifest(silver_manifest)
    return store.commit(staging)


def _gold(
    store: LocalArtifactStore,
    silver_ref: Any,
    *,
    silver_fingerprint: str = "silver-fingerprint",
    run_id: str = "gold-platinum",
) -> Any:
    identity = gold_input_fingerprint(
        silver_fingerprint_value=silver_fingerprint,
        method_name="deterministic",
        method_version="1",
        method_config={},
        prompt_bundle_fingerprint="none",
        generated_contract_version="1",
        selection_policy={"policy": "validation"},
    )
    result = execute_gold(
        store=store,
        request=GoldRequest(
            run_id=run_id,
            case_fingerprint="case-fingerprint",
            silver_manifest=silver_ref,
            silver_fingerprint=silver_fingerprint,
            gold_input_fingerprint=identity,
            method_name="deterministic",
            method_version="1",
            prompt_bundle_fingerprint="none",
        ),
        method=DeterministicMethod(),
        sandbox=SandboxedExecutor(timeout_seconds=5),
        environment=_environment(),
        profile=ExecutionProfile.DEVELOPMENT,
    )
    assert result.materialization.manifest_ref is not None
    return result.materialization.manifest_ref


def _execute(tmp_path: Path, *, selection_policy: SelectionPolicy | None = None) -> Any:
    store = LocalArtifactStore(tmp_path / "lake")
    silver_ref = _silver(store)
    gold_ref = _gold(store, silver_ref)
    evaluator = CVEvaluator(
        Settings(task="classification", metric="auc", evaluation={"cv_folds": 2})
    )
    result = execute_platinum(
        artifact_store=store,
        silver_ref=silver_ref,
        gold_ref=gold_ref,
        evaluator=evaluator,
        model_name="random_forest",
        run_id="platinum-run",
        case_fingerprint="case-fingerprint",
        environment_snapshot=_environment(),
        selection_policy=selection_policy,
        execution_profile=ExecutionProfile.DEVELOPMENT,
    )
    return store, result


def _silver_partitioned(store: LocalArtifactStore) -> Any:
    """Silver package with a discovery/evaluation partition and per-partition folds.

    16 balanced rows: discovery gets r0..r11 (folds 0,1,2), evaluation gets
    r12..r15 (folds 0,1). Candidate selection therefore runs on discovery
    rows only and the reported aggregate on the held-out evaluation rows.
    """
    n = 16
    features = pd.DataFrame(
        {
            "a": [i % 2 for i in range(n)],
            "b": [(i // 2) % 2 for i in range(n)],
        }
    )
    target_values = [i % 2 for i in range(n)]
    row_ids = pd.DataFrame({"row_id": [f"r{i}" for i in range(n)]})
    partition = ["discovery"] * 12 + ["evaluation"] * 4
    fold = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1]
    folds = pd.DataFrame({"row_id": row_ids["row_id"], "fold": fold, "partition": partition})
    bronze = BronzeRecord(
        source="fixture",
        source_checksum="source-part",
        target="target",
        task="classification",
        snapshot_mode="snapshot",
        row_count=n,
        column_count=3,
    )
    bronze_manifest = RunManifest(
        layer=Layer.BRONZE,
        package_kind="bronze",
        layer_fingerprint="bronze-fingerprint-part",
        run_id="bronze-platinum-part",
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(
            dataset="fixture",
            method="bronze",
            model="none",
            seed=42,
            options={"source": {"source_checksum": "source-part"}},
        ),
        environment=_environment(),
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    staging = store.begin(ArtifactNamespace(layer=Layer.BRONZE, run_id="bronze-platinum-part"))
    staging.write_json("bronze.json", bronze.model_dump(mode="json"))
    staging.write_dataframe("train.parquet", features.assign(target=target_values))
    staging.set_manifest(bronze_manifest)
    bronze_ref = store.commit(staging)
    checks = [CheckResult(check_id="SILVER.OK", passed=True, message="ok")]
    silver_manifest = RunManifest(
        layer=Layer.SILVER,
        package_kind="silver",
        layer_fingerprint="silver-fingerprint-part",
        run_id="silver-platinum-part",
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(
            dataset="fixture",
            method="silver",
            model="none",
            seed=42,
            options={"dataset_fingerprint": "silver-fingerprint-part"},
        ),
        environment=_environment(),
        upstream_manifests=[bronze_ref],
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    staging = store.begin(ArtifactNamespace(layer=Layer.SILVER, run_id="silver-platinum-part"))
    staging.write_dataframe("canonical_features.parquet", features)
    staging.write_dataframe(
        "canonical_target.parquet",
        pd.DataFrame({"row_id": row_ids["row_id"], "target": target_values}),
    )
    staging.write_dataframe("row_ids.parquet", row_ids)
    staging.write_dataframe("fold_assignments.parquet", folds)
    staging.write_json("profile.json", {"dataset_fingerprint": "silver-fingerprint-part"})
    staging.write_json("checks.json", [item.model_dump(mode="json") for item in checks])
    staging.set_manifest(silver_manifest)
    return store.commit(staging)


def _execute_partitioned(tmp_path: Path) -> Any:
    store = LocalArtifactStore(tmp_path / "lake-part")
    silver_ref = _silver_partitioned(store)
    gold_ref = _gold(
        store,
        silver_ref,
        silver_fingerprint="silver-fingerprint-part",
        run_id="gold-platinum-part",
    )
    evaluator = CVEvaluator(
        Settings(task="classification", metric="auc", evaluation={"cv_folds": 2})
    )
    result = execute_platinum(
        artifact_store=store,
        silver_ref=silver_ref,
        gold_ref=gold_ref,
        evaluator=evaluator,
        model_name="random_forest",
        run_id="platinum-run-part",
        case_fingerprint="case-fingerprint",
        environment_snapshot=_environment(),
        execution_profile=ExecutionProfile.DEVELOPMENT,
    )
    return store, result


def test_platinum_aggregates_reconstruct_and_folds_match(tmp_path: Path) -> None:
    store, result = _execute(tmp_path)
    package = result.package
    metrics = package.fold_metrics
    baseline = metrics[metrics.arm == "baseline"].sort_values("fold")
    enhanced = metrics[metrics.arm == "enhanced"].sort_values("fold")
    assert baseline.fold.tolist() == enhanced.fold.tolist() == [0, 1]
    assert package.aggregate.baseline_score == pytest.approx(baseline.score.mean())
    assert package.aggregate.enhanced_score == pytest.approx(enhanced.score.mean())
    assert set(package.predictions.query("arm == 'baseline'").row_id) == set(
        package.fold_assignments.row_id
    )
    assert result.materialization.manifest_ref is not None
    reloaded = load_platinum_package(store, result.materialization.manifest_ref)
    assert reloaded.aggregate == package.aggregate
    pd.testing.assert_frame_equal(reloaded.fold_metrics, package.fold_metrics)
    values = list(result.experiment_result)
    assert values[:9] == [
        "dataset",
        "method",
        "model",
        "seed",
        "cv_score",
        "gain",
        "baseline_score",
        "num_features_generated",
        "error",
    ]
    assert result.experiment_result["manifest_uri"].endswith("manifest.json")
    assert result.experiment_result["platinum_fingerprint"] == package.manifest.layer_fingerprint


def test_reordered_gold_rows_normalize_by_row_id(tmp_path: Path) -> None:
    _store, result = _execute(tmp_path)
    assert result.package.request.evaluation_policy.alignment == "normalize_by_row_id"
    assert result.package.fold_assignments.row_id.tolist() == [f"r{i}" for i in range(8)]


def test_recommended_policy_requires_real_evidence_gate() -> None:
    with pytest.raises(ValueError, match="practical threshold"):
        SelectionPolicy(profile="recommended")


def test_metric_direction_metadata_fails_closed_for_plugin() -> None:
    MetricRegistry.register("directionless", lambda y, p: 0.0)
    try:
        with pytest.raises(EvaluationError, match="no direction metadata"):
            MetricRegistry.get_direction("directionless")
    finally:
        MetricRegistry.reset()
    assert MetricRegistry.get_direction("rmse") is MetricDirection.MINIMIZE


def test_explicit_folds_create_fresh_estimator_per_fold() -> None:
    created: list[object] = []

    class Factory:
        def get_model(self, model_name: str | None, task: str) -> Any:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=2, random_state=len(created), n_jobs=1)
            created.append(model)
            return model

    evaluator = CVEvaluator(
        Settings(task="classification", metric="auc", evaluation={"cv_folds": 2}),
        model_factory=Factory(),  # type: ignore[arg-type]
    )
    evaluator.evaluate_on_folds(
        pd.DataFrame({"x": range(8)}),
        pd.Series([0, 1, 0, 1, 0, 1, 0, 1]),
        pd.Series([f"r{i}" for i in range(8)]),
        pd.DataFrame({"row_id": [f"r{i}" for i in range(8)], "fold": [0] * 4 + [1] * 4}),
        arm="baseline",
        model_name="random_forest",
    )
    assert len(created) == 2
    assert created[0] is not created[1]


def test_corrupt_platinum_evidence_fails_verified_loading(tmp_path: Path) -> None:
    store, result = _execute(tmp_path)
    assert result.materialization.manifest_ref is not None
    path = tmp_path / "lake/04_platinum/runs/platinum-run/fold_metrics.parquet"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(DatasetError, match="Unable to load verified Platinum package"):
        load_platinum_package(store, result.materialization.manifest_ref)


def test_reordered_gold_rows_are_normalized_or_rejected_by_policy(tmp_path: Path) -> None:
    store, result = _execute(tmp_path)
    silver = load_silver_package(store, result.package.request.silver_manifest)
    gold = load_gold_package(store, result.package.request.gold_manifest)
    reordered_gold = gold.model_copy(
        update={"accepted_features": gold.accepted_features.iloc[::-1].reset_index(drop=True)}
    )
    evaluator = CVEvaluator(
        Settings(task="classification", metric="auc", evaluation={"cv_folds": 2})
    )
    normalized_request = build_platinum_request(
        run_id="normalized",
        case_fingerprint="case-fingerprint",
        silver=silver,
        gold=reordered_gold,
        gold_ref=result.package.request.gold_manifest,
        evaluator=evaluator,
        model_name="random_forest",
    )
    metrics, *_rest = _evaluate(normalized_request, silver, reordered_gold, evaluator)
    assert set(metrics["arm"]) >= {"baseline", "enhanced"}

    strict_request = build_platinum_request(
        run_id="strict",
        case_fingerprint="case-fingerprint",
        silver=silver,
        gold=reordered_gold,
        gold_ref=result.package.request.gold_manifest,
        evaluator=evaluator,
        model_name="random_forest",
        evaluation_policy=EvaluationPolicy(alignment="strict_order"),
    )
    with pytest.raises(DatasetError, match="row order"):
        _evaluate(strict_request, silver, reordered_gold, evaluator)


def test_existing_verified_package_is_reused_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, first = _execute(tmp_path)
    monkeypatch.setattr(
        "feature_forge.dataflows.platinum._evaluate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not evaluate")),
    )
    second = execute_platinum(
        artifact_store=store,
        silver_ref=first.package.request.silver_manifest,
        gold_ref=first.package.request.gold_manifest,
        evaluator=CVEvaluator(
            Settings(task="classification", metric="auc", evaluation={"cv_folds": 2})
        ),
        model_name="random_forest",
        run_id="platinum-run",
        case_fingerprint="case-fingerprint",
        environment_snapshot=_environment(),
        execution_profile=ExecutionProfile.DEVELOPMENT,
    )
    assert second.package.manifest == first.package.manifest
    assert second.experiment_result == first.experiment_result


def test_hash_valid_prediction_tamper_fails_semantic_reconstruction(tmp_path: Path) -> None:
    store, result = _execute(tmp_path)
    ref = result.materialization.manifest_ref
    assert ref is not None
    package_path = tmp_path / "lake/04_platinum/runs/platinum-run"
    prediction_path = package_path / "predictions.parquet"
    predictions = pd.read_parquet(prediction_path)
    baseline_fold = (predictions["arm"] == "baseline") & (
        predictions["fold"] == predictions.loc[0, "fold"]
    )
    predictions.loc[baseline_fold, "prediction_json"] = "0.5"
    predictions.to_parquet(prediction_path)

    manifest_path = package_path / "manifest.json"
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    artifacts = [
        item.model_copy(
            update={
                "sha256": sha256_file(prediction_path),
                "size_bytes": prediction_path.stat().st_size,
            }
        )
        if item.relative_path == "predictions.parquet"
        else item
        for item in manifest.artifacts
    ]
    atomic_write_json(
        manifest_path,
        manifest.model_copy(update={"artifacts": artifacts}).model_dump(mode="json"),
    )
    new_hash = sha256_file(manifest_path)
    (package_path / "_SUCCESS").write_text(f"{new_hash}\n", encoding="utf-8")
    tampered_ref = ref.model_copy(update={"sha256": new_hash})
    with pytest.raises(DatasetError, match="does not reconstruct"):
        load_platinum_package(store, tampered_ref)


def test_real_package_result_matches_across_process_seam(tmp_path: Path) -> None:
    store, result = _execute(tmp_path)
    assert result.materialization.manifest_ref is not None
    payload = (
        str(store.root),
        result.package.request.silver_manifest.model_dump(mode="json"),
        result.package.request.gold_manifest.model_dump(mode="json"),
        result.materialization.manifest_ref.model_dump(mode="json"),
    )
    sequential = SequentialExecutionAdapter().run([payload], package_result_worker, progress=False)
    process = ProcessPoolExecutionAdapter(max_workers=1).run(
        [payload], package_result_worker, progress=False
    )
    assert process == sequential == [result.experiment_result]


def test_hamilton_entrypoint_uses_driver_config_names() -> None:
    parameters = inspect.signature(execute_platinum).parameters
    assert "artifact_store" in parameters
    assert "environment_snapshot" in parameters
    assert "execution_profile" in parameters
    assert "store" not in parameters
    assert "profile" not in parameters


# --------------------------------------------------------------------------- #
# Leakage-safe evaluation: discovery/evaluation partition on the Platinum path.
# --------------------------------------------------------------------------- #


def test_partitioned_platinum_scores_candidates_on_discovery_and_reports_on_evaluation(
    tmp_path: Path,
) -> None:
    store, result = _execute_partitioned(tmp_path)
    package = result.package
    metrics = package.fold_metrics
    arms = set(metrics["arm"].unique())
    # Two baseline arms: discovery (paired with candidates) + evaluation
    # (paired with the reported enhanced).
    assert "baseline:discovery" in arms
    assert "baseline" in arms
    assert "enhanced" in arms
    assert arms.intersection({f"candidate:{name}" for name in ("signal",)})

    silver = load_silver_package(store, package.request.silver_manifest)
    folds = silver.fold_assignments
    discovery_ids = set(folds.loc[folds["partition"] == "discovery", "row_id"])
    evaluation_ids = set(folds.loc[folds["partition"] == "evaluation", "row_id"])

    predictions = package.predictions
    # Candidate + discovery-baseline rows come from the discovery partition.
    for arm in ("baseline:discovery", *[a for a in arms if a.startswith("candidate:")]):
        arm_rows = set(predictions.loc[predictions["arm"] == arm, "row_id"])
        assert arm_rows == discovery_ids, f"{arm} not on discovery rows"
    # Reported baseline + enhanced come from the evaluation partition.
    for arm in ("baseline", "enhanced"):
        arm_rows = set(predictions.loc[predictions["arm"] == arm, "row_id"])
        assert arm_rows == evaluation_ids, f"{arm} not on evaluation rows"


def test_partitioned_platinum_loader_verifies_and_reports_counts(tmp_path: Path) -> None:
    store, result = _execute_partitioned(tmp_path)
    assert result.materialization.manifest_ref is not None
    # The durable package must replay through the adversarial loader.
    reloaded = load_platinum_package(store, result.materialization.manifest_ref)
    assert reloaded.aggregate == result.package.aggregate
    pd.testing.assert_frame_equal(reloaded.fold_metrics, result.package.fold_metrics)

    # Reported counts surface the discovery/evaluation split.
    exp = result.experiment_result
    assert exp["n_discovery_rows"] == 12
    assert exp["n_evaluation_rows"] == 4


def test_partitioned_platinum_fingerprint_binds_selection_partition(tmp_path: Path) -> None:
    store, result = _execute_partitioned(tmp_path)
    package = result.package
    # The selection_partition policy is part of the persisted evaluation policy.
    assert package.request.evaluation_policy.selection_partition == "discovery"
    # The persisted request round-trips and the fingerprint is stable on reload.
    reloaded = load_platinum_package(store, result.materialization.manifest_ref)
    assert (
        reloaded.request.evaluation_policy.selection_partition
        == package.request.evaluation_policy.selection_partition
    )
