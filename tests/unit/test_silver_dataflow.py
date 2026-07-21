"""Offline Hamilton Bronze/Silver dataflow tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import pytest

from feature_forge.contracts import ArtifactRef, DatasetRequest, EnvironmentSnapshot, Layer
from feature_forge.dataflows import load_silver_package
from feature_forge.dataflows.driver import SILVER_FINAL_VARS, build_driver
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.dataflows.visualization import render_silver_dag
from feature_forge.exceptions import DatasetError
from feature_forge.experiment.lifecycle import LocalRunRepository
from feature_forge.storage.local import LocalArtifactStore

ad_hoc_utils = pytest.importorskip(
    "hamilton.ad_hoc_utils",
    reason="requires the optional pipeline extra",
)


def review_marker() -> str:
    """Small additional Hamilton node used to test module composition."""
    return "ok"


class StubRegistry:
    """In-memory registry double that records every attempted load."""

    def __init__(
        self,
        source: str = "local",
        include_secret: bool = False,
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        self.train = pd.DataFrame(
            {
                "age": [10, 20, 30, 40, 50, 60],
                "fare": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        self.source = source
        self.include_secret = include_secret
        self.metadata_extra = metadata_extra or {}
        self.load_calls = 0

    def info(self, name: str) -> dict[str, Any]:
        metadata = {
            "source": self.source,
            "target": "target",
            "task": "classification",
            "name": name,
            **self.metadata_extra,
        }
        if self.include_secret:
            metadata["api_key"] = "should-not-be-persisted"
        return metadata

    def load(self, name: str) -> dict[str, Any]:
        self.load_calls += 1
        return {
            "train": self.train.copy(),
            "test": pd.DataFrame(),
            "target": "target",
            "metadata": self.info(name),
        }


def _request(
    run_id: str = "silver-run",
    *,
    case_fingerprint: str = "case-fingerprint",
    source_policy: Literal["reference", "snapshot"] = "reference",
    canonicalization_config: dict[str, Any] | None = None,
) -> DatasetRequest:
    return DatasetRequest(
        name="demo",
        target="target",
        task="classification",
        run_id=run_id,
        case_fingerprint=case_fingerprint,
        split_seed=42,
        cv_folds=3,
        source_policy=source_policy,
        canonicalization_config=canonicalization_config or {},
    )


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.13",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
    )


def _execute(
    driver: Any,
    registry: StubRegistry,
    request: DatasetRequest,
    *final_vars: str,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        driver.execute(
            list(final_vars),
            inputs={
                "dataset_request": request,
                "dataset_registry": registry,
                "environment_snapshot": _environment(),
            },
        ),
    )


class TestSilverDataflow:
    @pytest.mark.parametrize("profile", [ExecutionProfile.PRODUCTION, ExecutionProfile.REPLAY])
    def test_durable_profiles_require_an_artifact_store(
        self,
        profile: ExecutionProfile,
    ) -> None:
        with pytest.raises(ValueError, match="requires an artifact store"):
            build_driver(profile=profile)

    def test_additional_modules_extend_instead_of_replace_silver_graph(self) -> None:
        extra_module = ad_hoc_utils.create_temporary_module(review_marker)
        driver = build_driver(
            profile=ExecutionProfile.DOCUMENTATION,
            modules=(extra_module,),
        )

        assert driver.execute(["review_marker"])["review_marker"] == "ok"
        assert any(
            variable.name == "silver_materialization"
            for variable in driver.list_available_variables()
        )

    def test_production_driver_emits_hamilton_events_to_run_journal(self, tmp_path: Path) -> None:
        extra_module = ad_hoc_utils.create_temporary_module(review_marker)
        repository = LocalRunRepository(tmp_path / "control")
        driver = build_driver(
            profile=ExecutionProfile.DOCUMENTATION,
            modules=(extra_module,),
            lifecycle_repository=repository,
            run_id="observed-run",
            case_id="observed-case",
        )

        assert driver.execute(["review_marker"])["review_marker"] == "ok"
        events = repository.load_events("observed-run")
        assert [event.event_type for event in events] == [
            "hamilton_node_running",
            "hamilton_node_succeeded",
        ]
        assert all(event.stage == "review_marker" for event in events)

    def test_ci_profile_is_offline_and_does_not_persist(self) -> None:
        driver = build_driver(profile=ExecutionProfile.CI)
        registry = StubRegistry()
        outputs = _execute(
            driver,
            registry,
            _request(),
            "bronze_materialization",
            "silver_materialization",
        )

        assert registry.load_calls == 1
        assert not outputs["bronze_materialization"].persisted
        assert not outputs["silver_materialization"].persisted
        assert outputs["silver_materialization"].manifest.state.value == "succeeded"
        assert outputs["silver_materialization"].manifest.artifacts == []

    def test_deterministic_identity_folds_and_network_guard(self) -> None:
        driver = build_driver(profile=ExecutionProfile.CI)
        first = _execute(
            driver,
            StubRegistry(metadata_extra={"loaded_at": "first"}),
            _request(canonicalization_config={"max_tokens": 100}),
            "bronze_snapshot",
            "dataset_fingerprint_value",
            "fold_assignments",
        )
        second = _execute(
            driver,
            StubRegistry(metadata_extra={"loaded_at": "second"}),
            _request(canonicalization_config={"max_tokens": 100}),
            "bronze_snapshot",
            "dataset_fingerprint_value",
            "fold_assignments",
        )

        assert first["bronze_snapshot"].source_checksum == second["bronze_snapshot"].source_checksum
        assert first["dataset_fingerprint_value"] == second["dataset_fingerprint_value"]
        pd.testing.assert_frame_equal(first["fold_assignments"], second["fold_assignments"])

        remote_registry = StubRegistry(source="kaggle")
        with pytest.raises(DatasetError, match="forbids network-backed"):
            _execute(driver, remote_registry, _request(), "silver_materialization")
        assert remote_registry.load_calls == 0

    def test_dataset_fingerprint_changes_with_canonicalization_config(self) -> None:
        driver = build_driver(profile=ExecutionProfile.CI)
        first = _execute(
            driver,
            StubRegistry(),
            _request(canonicalization_config={"max_tokens": 100}),
            "dataset_fingerprint_value",
        )
        second = _execute(
            driver,
            StubRegistry(),
            _request(canonicalization_config={"max_tokens": 200}),
            "dataset_fingerprint_value",
        )

        assert first["dataset_fingerprint_value"] != second["dataset_fingerprint_value"]

    def test_physical_source_policy_does_not_change_logical_dataset_identity(self) -> None:
        driver = build_driver(profile=ExecutionProfile.CI)
        referenced = _execute(
            driver,
            StubRegistry(),
            _request(source_policy="reference"),
            "dataset_fingerprint_value",
        )
        snapshotted = _execute(
            driver,
            StubRegistry(),
            _request(source_policy="snapshot"),
            "dataset_fingerprint_value",
        )

        assert referenced["dataset_fingerprint_value"] == snapshotted["dataset_fingerprint_value"]

    def test_development_profile_commits_bronze_and_silver(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        driver = build_driver(
            profile=ExecutionProfile.DEVELOPMENT,
            artifact_store=store,
            cache_dir=tmp_path / "hamilton-cache",
        )
        result = _execute(driver, StubRegistry(), _request(), "silver_materialization")[
            "silver_materialization"
        ]

        assert result.persisted
        assert result.manifest_ref is not None
        assert store.verify(result.manifest_ref).valid
        assert result.manifest_ref.layer is Layer.SILVER
        assert set(result.artifact_paths) == {
            "canonical_features.parquet",
            "canonical_target.parquet",
            "checks.json",
            "fold_assignments.parquet",
            "profile.json",
            "row_ids.parquet",
        }
        assert {artifact.relative_path for artifact in result.manifest.artifacts} == set(
            result.artifact_paths
        )
        assert result.manifest.stages[0].artifacts == result.manifest.artifacts

        bronze_ref = result.manifest.upstream_manifests[0]
        assert bronze_ref.layer is Layer.BRONZE
        assert store.verify(bronze_ref).valid
        bronze_manifest = store.load_manifest(bronze_ref)
        assert {artifact.relative_path for artifact in bronze_manifest.artifacts} == {
            "bronze.json",
            "source_reference.json",
        }

    def test_snapshot_policy_persists_raw_bronze_source(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        driver = build_driver(
            profile=ExecutionProfile.DEVELOPMENT,
            artifact_store=store,
            cache_dir=tmp_path / "hamilton-cache",
        )
        registry = StubRegistry()
        result = _execute(
            driver,
            registry,
            _request("snapshot-run", source_policy="snapshot"),
            "silver_materialization",
        )["silver_materialization"]

        bronze_ref = result.manifest.upstream_manifests[0]
        bronze_manifest = store.load_manifest(bronze_ref)
        assert {artifact.relative_path for artifact in bronze_manifest.artifacts} == {
            "bronze.json",
            "train.parquet",
        }
        snapshot_path = store.resolve(
            ArtifactRef(
                layer=Layer.BRONZE,
                run_id="snapshot-run",
                relative_path="train.parquet",
            )
        )
        pd.testing.assert_frame_equal(pd.read_parquet(snapshot_path), registry.train)

    def test_committed_silver_loads_deterministically_without_registry(
        self, tmp_path: Path
    ) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        driver = build_driver(
            profile=ExecutionProfile.DEVELOPMENT,
            artifact_store=store,
            cache_dir=tmp_path / "hamilton-cache",
        )
        result = _execute(
            driver,
            StubRegistry(),
            _request("replay-run"),
            "silver_materialization",
        )["silver_materialization"]
        assert result.manifest_ref is not None

        first = load_silver_package(store, result.manifest_ref)
        second = load_silver_package(store, result.manifest_ref)

        pd.testing.assert_frame_equal(first.canonical_features, second.canonical_features)
        pd.testing.assert_frame_equal(first.canonical_target, second.canonical_target)
        pd.testing.assert_frame_equal(first.fold_assignments, second.fold_assignments)
        assert first.profile["dataset_fingerprint"] == second.profile["dataset_fingerprint"]
        assert all(check.passed for check in first.checks if check.required)

    def test_reexecuting_same_run_reuses_verified_packages(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        driver = build_driver(
            profile=ExecutionProfile.DEVELOPMENT,
            artifact_store=store,
            cache_dir=tmp_path / "hamilton-cache",
        )
        first = _execute(driver, StubRegistry(), _request("reuse-run"), "silver_materialization")[
            "silver_materialization"
        ]
        second = _execute(driver, StubRegistry(), _request("reuse-run"), "silver_materialization")[
            "silver_materialization"
        ]

        assert first.manifest_ref == second.manifest_ref
        assert first.artifact_paths == second.artifact_paths

    def test_silver_reuse_ignores_aggregate_case_identity(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        driver = build_driver(
            profile=ExecutionProfile.DEVELOPMENT,
            artifact_store=store,
            cache_dir=tmp_path / "hamilton-cache",
        )
        first = _execute(
            driver,
            StubRegistry(),
            _request("layer-reuse", case_fingerprint="model-a-metric-a"),
            "silver_materialization",
        )["silver_materialization"]
        second = _execute(
            driver,
            StubRegistry(),
            _request("layer-reuse", case_fingerprint="model-b-metric-b"),
            "silver_materialization",
        )["silver_materialization"]

        assert first.manifest_ref == second.manifest_ref
        assert first.manifest.layer_fingerprint == second.manifest.layer_fingerprint

    def test_silver_reuse_ignores_physical_source_policy(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        driver = build_driver(
            profile=ExecutionProfile.DEVELOPMENT,
            artifact_store=store,
            cache_dir=tmp_path / "hamilton-cache",
        )
        referenced = _execute(
            driver,
            StubRegistry(),
            _request("source-policy-reuse", source_policy="reference"),
            "silver_materialization",
        )["silver_materialization"]
        snapshotted = _execute(
            driver,
            StubRegistry(),
            _request("source-policy-reuse", source_policy="snapshot"),
            "silver_materialization",
        )["silver_materialization"]

        assert referenced.manifest_ref == snapshotted.manifest_ref
        assert referenced.manifest.layer_fingerprint == snapshotted.manifest.layer_fingerprint

    def test_silver_manifests_and_metadata_redact_source_secrets(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        driver = build_driver(
            profile=ExecutionProfile.DEVELOPMENT,
            artifact_store=store,
            cache_dir=tmp_path / "hamilton-cache",
        )
        result = _execute(
            driver,
            StubRegistry(include_secret=True),
            _request("secret-run"),
            "silver_materialization",
        )["silver_materialization"]

        persisted_json = "\n".join(
            path.read_text(encoding="utf-8") for path in (tmp_path / "lake").rglob("*.json")
        )
        assert result.persisted
        assert "should-not-be-persisted" not in persisted_json
        assert '"configured":true' in persisted_json or '"configured": true' in persisted_json

    def test_documentation_profile_renders_dag_without_execution(self, tmp_path: Path) -> None:
        driver = build_driver(profile=ExecutionProfile.DOCUMENTATION)
        output_path = tmp_path / "silver-dag.png"

        render_silver_dag(driver, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert SILVER_FINAL_VARS == ["silver_materialization"]
