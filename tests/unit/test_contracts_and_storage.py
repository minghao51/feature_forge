"""Contract and atomic-store tests for the medallion-lite foundation."""

from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from feature_forge.contracts import (
    ArtifactDescriptor,
    ArtifactNamespace,
    ArtifactRef,
    EnvironmentSnapshot,
    Layer,
    ManifestRef,
    RunManifest,
    RunRequest,
    RunState,
    bronze_fingerprint,
    gold_input_fingerprint,
    platinum_input_fingerprint,
    silver_fingerprint,
)
from feature_forge.storage import local as local_storage
from feature_forge.storage.hashing import canonical_json_bytes, dataset_fingerprint
from feature_forge.storage.local import LocalArtifactStore


class TestDependencyGroups:
    def test_pipeline_dependencies_are_not_observability_dependencies(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
        extras = project["optional-dependencies"]
        observability = {
            requirement.split("[")[0].split(">=")[0] for requirement in extras["observability"]
        }
        pipeline = {requirement.split("[")[0].split(">=")[0] for requirement in extras["pipeline"]}
        aggregate = {requirement.split("[")[0].split(">=")[0] for requirement in extras["all"]}

        assert {"duckdb", "sf-hamilton"}.isdisjoint(observability)
        assert {"duckdb", "sf-hamilton"} <= pipeline
        assert pipeline <= aggregate


def _manifest(run_id: str, layer: Layer = Layer.GOLD) -> RunManifest:
    return RunManifest(
        schema_version="1",
        layer=layer,
        package_kind=layer.value,
        layer_fingerprint=f"{layer.value}-fingerprint",
        run_id=run_id,
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="demo", method="dummy", model="rf", seed=42),
        environment=EnvironmentSnapshot(
            python_version="3.13",
            operating_system="darwin",
            architecture="arm64",
            feature_forge_version="0+test",
        ),
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


class TestSecretFreeHashing:
    def test_secret_values_do_not_affect_identity_or_serialized_payload(self) -> None:
        first = {"llm": {"api_key": "first-secret"}, "password": "one", "model": "demo"}
        second = {"llm": {"api_key": "second-secret"}, "password": "two", "model": "demo"}

        assert canonical_json_bytes(first) == canonical_json_bytes(second)
        assert b"first-secret" not in canonical_json_bytes(first)
        assert b"second-secret" not in canonical_json_bytes(second)

    def test_dataset_fingerprint_is_stable_for_equivalent_mapping_order(self) -> None:
        first = dataset_fingerprint(
            source_checksum="source",
            canonicalization_config={"b": 2, "a": 1},
            schema_version="1",
            target_name="target",
            task="classification",
            split_policy={"strategy": "kfold", "folds": 3},
            split_seed=42,
        )
        second = dataset_fingerprint(
            source_checksum="source",
            canonicalization_config={"a": 1, "b": 2},
            schema_version="1",
            target_name="target",
            task="classification",
            split_policy={"folds": 3, "strategy": "kfold"},
            split_seed=42,
        )
        assert first == second

    def test_non_secret_token_settings_remain_part_of_identity(self) -> None:
        first = canonical_json_bytes({"max_tokens": 100, "token_count": 12})
        second = canonical_json_bytes({"max_tokens": 200, "token_count": 12})

        assert first != second
        assert b"100" in first

    def test_unordered_binary_values_normalize_deterministically(self) -> None:
        first = canonical_json_bytes({"values": {b"a", b"b"}})
        second = canonical_json_bytes({"values": {b"b", b"a"}})

        assert first == second

    def test_mapping_key_collisions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="collide after string normalization"):
            canonical_json_bytes({1: "integer", "1": "string"})

    def test_unknown_manifest_schema_or_fields_are_rejected(self) -> None:
        payload = _manifest("strict-run").model_dump(mode="json")
        payload["schema_version"] = "2"
        with pytest.raises(ValueError, match="schema_version"):
            RunManifest.model_validate(payload)
        payload["schema_version"] = "1"
        payload["future_field"] = True
        with pytest.raises(ValueError, match="future_field"):
            RunManifest.model_validate(payload)

    def test_layer_fingerprints_invalidate_only_downstream_inputs(self) -> None:
        bronze = bronze_fingerprint(source_checksum="source")
        silver = silver_fingerprint(
            bronze_fingerprint_value=bronze,
            target_name="target",
            task="classification",
            canonicalization_config={"missing": "median"},
            split_policy={"folds": 3},
            split_seed=42,
        )
        gold = gold_input_fingerprint(
            silver_fingerprint_value=silver,
            method_name="malmas",
            method_version="1",
            method_config={"rounds": 2},
            prompt_bundle_fingerprint="prompt-a",
            generated_contract_version="1",
            selection_policy={"gain": 0.0},
        )
        platinum = platinum_input_fingerprint(
            gold_fingerprint_value=gold,
            model_name="random_forest",
            model_version="1",
            model_config={},
            metric="auc",
            fold_fingerprint="folds",
            evaluation_policy={"bootstrap": 100},
            uncertainty_policy={"confidence": 0.95},
        )

        # Model and metric are intentionally absent from Silver and Gold identities.
        assert silver == silver_fingerprint(
            bronze_fingerprint_value=bronze,
            target_name="target",
            task="classification",
            canonicalization_config={"missing": "median"},
            split_policy={"folds": 3},
            split_seed=42,
        )
        assert gold != gold_input_fingerprint(
            silver_fingerprint_value=silver,
            method_name="malmas",
            method_version="1",
            method_config={"rounds": 2},
            prompt_bundle_fingerprint="prompt-b",
            generated_contract_version="1",
            selection_policy={"gain": 0.0},
        )
        assert platinum != platinum_input_fingerprint(
            gold_fingerprint_value=gold,
            model_name="random_forest",
            model_version="1",
            model_config={},
            metric="rmse",
            fold_fingerprint="folds",
            evaluation_policy={"bootstrap": 100},
            uncertainty_policy={"confidence": 0.95},
        )
        assert silver != silver_fingerprint(
            bronze_fingerprint_value=bronze,
            target_name="target",
            task="classification",
            canonicalization_config={"missing": "drop"},
            split_policy={"folds": 3},
            split_seed=42,
        )


class TestLocalArtifactStore:
    def test_commit_verify_and_resolve(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        namespace = ArtifactNamespace(layer=Layer.GOLD, run_id="run-1")
        staging = store.begin(namespace)
        descriptor = staging.write_json("report.json", {"score": 0.9})
        staging.set_manifest(_manifest("run-1"))

        ref = store.commit(staging)
        report = store.verify(ref)

        assert report.valid
        assert report.checked_artifacts == ["report.json"]
        assert store.resolve(
            ArtifactRef(layer=Layer.GOLD, run_id="run-1", relative_path=descriptor.relative_path)
        ).read_text(encoding="utf-8")
        assert (tmp_path / "lake/03_gold/runs/run-1/_SUCCESS").is_file()
        assert (tmp_path / "lake/03_gold/runs/run-1/_SUCCESS").read_text().strip() == ref.sha256
        assert staging.manifest is not None
        assert staging.manifest.artifacts == [descriptor]

    def test_completion_marker_is_staged_before_atomic_publication(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="atomic-run"))
        staging.write_text("result.txt", "complete")
        staging.set_manifest(_manifest("atomic-run"))
        original_publish = local_storage.atomic_publish_directory

        def publish_with_marker_check(staging_path: Path, destination: Path) -> None:
            assert (staging_path / "_SUCCESS").is_file()
            original_publish(staging_path, destination)

        monkeypatch.setattr(local_storage, "atomic_publish_directory", publish_with_marker_check)

        store.commit(staging)

    def test_manifest_artifact_layer_must_match_namespace(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        staging = store.begin(ArtifactNamespace(layer=Layer.SILVER, run_id="layer-run"))
        manifest = _manifest("layer-run", Layer.SILVER).model_copy(
            update={
                "artifacts": [
                    ArtifactDescriptor(
                        schema_version="1",
                        name="optional",
                        layer=Layer.GOLD,
                        media_type="text/plain",
                        relative_path="optional.txt",
                        sha256="0" * 64,
                        size_bytes=0,
                        required=False,
                    )
                ]
            }
        )
        staging.set_manifest(manifest)

        with pytest.raises(ValueError, match="layer does not match namespace"):
            store.commit(staging)

    def test_corruption_is_detected(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        staging = store.begin(ArtifactNamespace(layer=Layer.SILVER, run_id="run-2"))
        staging.write_text("data.txt", "original")
        staging.set_manifest(_manifest("run-2", Layer.SILVER))
        ref = store.commit(staging)

        (tmp_path / "lake/02_silver/runs/run-2/data.txt").write_text("changed", encoding="utf-8")
        report = store.verify(ref)

        assert not report.valid
        assert "artifact hash mismatch: data.txt" in report.errors

    def test_manifest_tampering_is_rejected_by_safe_resolution(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="tampered-manifest"))
        staging.write_text("data.txt", "original")
        staging.set_manifest(_manifest("tampered-manifest"))
        store.commit(staging)
        manifest_path = tmp_path / "lake/03_gold/runs/tampered-manifest/manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["case_fingerprint"] = "tampered"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="manifest hash does not match reference"):
            store.resolve(
                ArtifactRef(
                    layer=Layer.GOLD,
                    run_id="tampered-manifest",
                    relative_path="data.txt",
                )
            )

    def test_completion_marker_tampering_is_rejected(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="tampered-marker"))
        staging.write_text("data.txt", "original")
        staging.set_manifest(_manifest("tampered-marker"))
        store.commit(staging)
        (tmp_path / "lake/03_gold/runs/tampered-marker/_SUCCESS").write_text(
            "0" * 64 + "\n", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="failed verification"):
            store.get_manifest_ref(ArtifactNamespace(layer=Layer.GOLD, run_id="tampered-marker"))

    def test_optional_corruption_warns_but_safe_resolution_rejects(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="optional-run"))
        staging.write_text("optional.txt", "original", required=False)
        staging.set_manifest(_manifest("optional-run"))
        ref = store.commit(staging)
        (tmp_path / "lake/03_gold/runs/optional-run/optional.txt").write_text(
            "changed", encoding="utf-8"
        )

        report = store.verify(ref)

        assert report.valid
        assert "artifact hash mismatch: optional.txt" in report.warnings
        with pytest.raises(ValueError, match="failed integrity verification"):
            store.resolve(
                ArtifactRef(
                    layer=Layer.GOLD,
                    run_id="optional-run",
                    relative_path="optional.txt",
                )
            )

    def test_undeclared_file_cannot_be_resolved(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="undeclared-run"))
        staging.write_text("declared.txt", "declared")
        staging.set_manifest(_manifest("undeclared-run"))
        store.commit(staging)
        (tmp_path / "lake/03_gold/runs/undeclared-run/extra.txt").write_text(
            "extra", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="not declared in manifest"):
            store.resolve(
                ArtifactRef(
                    layer=Layer.GOLD,
                    run_id="undeclared-run",
                    relative_path="extra.txt",
                )
            )

    def test_existing_namespace_cannot_be_overwritten(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        namespace = ArtifactNamespace(layer=Layer.BRONZE, run_id="same-run")

        first = store.begin(namespace)
        first.write_text("source.txt", "one")
        first.set_manifest(_manifest("same-run", Layer.BRONZE))
        store.commit(first)

        second = store.begin(namespace)
        second.write_text("source.txt", "two")
        second.set_manifest(_manifest("same-run", Layer.BRONZE))
        with pytest.raises(FileExistsError, match="already exists"):
            store.commit(second)

        assert (tmp_path / "lake/01_bronze/runs/same-run/source.txt").read_text() == "one"

    def test_paths_cannot_escape_staging_namespace(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="safe-run"))
        with pytest.raises(ValueError, match="normalized and relative"):
            staging.write_text("../outside.txt", "nope")
        with pytest.raises(ValueError, match="normalized and relative"):
            staging.write_text(r"..\outside.txt", "nope")

    def test_incomplete_package_cannot_be_resolved(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        package = tmp_path / "lake/03_gold/runs/incomplete-run"
        package.mkdir(parents=True)
        (package / "report.json").write_text("{}", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="package is incomplete"):
            store.resolve(
                ArtifactRef(
                    layer=Layer.GOLD,
                    run_id="incomplete-run",
                    relative_path="report.json",
                )
            )
        with pytest.raises(ValueError, match="package is incomplete"):
            store.get_manifest_ref(ArtifactNamespace(layer=Layer.GOLD, run_id="incomplete-run"))

    def test_unknown_store_schema_version_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unsupported artifact schema version"):
            LocalArtifactStore(tmp_path / "lake", schema_version=cast(Any, "2"))

    def test_reserved_completion_files_cannot_be_artifacts(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "lake")
        staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="reserved-run"))
        with pytest.raises(ValueError, match="reserved package paths"):
            staging.write_text("_SUCCESS", "nope")

    def test_manifest_reference_path_is_fixed(self) -> None:
        with pytest.raises(ValueError, match=r"manifest\.json"):
            ManifestRef(
                layer=Layer.SILVER,
                run_id="safe-run",
                relative_path="other.json",
                sha256="0" * 64,
            )
