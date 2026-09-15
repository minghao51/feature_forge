"""Focused tests for the PR 2A durable contract and storage foundation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from feature_forge.contracts import (
    ArtifactNamespace,
    ArtifactRef,
    EnvironmentSnapshot,
    Layer,
    MethodIdentity,
    ResumePolicy,
    RunManifest,
    RunRequest,
    RunState,
    StageDisposition,
    bronze_fingerprint,
    gold_input_fingerprint,
    platinum_input_fingerprint,
    silver_fingerprint,
)
from feature_forge.experiment.resume import build_resume_plan, execute_resume_plan
from feature_forge.storage.local import LocalArtifactStore


def _manifest(
    run_id: str, layer: Layer = Layer.GOLD, upstream: list[object] | None = None
) -> RunManifest:
    return RunManifest(
        layer=layer,
        package_kind=layer.value,
        layer_fingerprint=f"{layer.value}-fingerprint",
        run_id=run_id,
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="demo", method="dummy", model="rf", seed=42),
        environment=EnvironmentSnapshot(
            python_version="3.12",
            operating_system="linux",
            architecture="x86_64",
            feature_forge_version="0+test",
        ),
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        upstream_manifests=upstream or [],
    )


class TestContracts:
    def test_unknown_fields_and_unsafe_paths_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactNamespace(layer=Layer.GOLD, run_id="run", extra=True)  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="normalized relative"):
            ArtifactRef(layer=Layer.GOLD, run_id="run", relative_path="../secret")

    def test_manifest_package_kind_matches_layer(self) -> None:
        with pytest.raises(ValidationError, match="package_kind"):
            RunManifest.model_validate(
                _manifest("run", Layer.SILVER).model_dump(mode="json") | {"package_kind": "gold"}
            )

    def test_layer_identity_changes_only_when_its_inputs_change(self) -> None:
        bronze = bronze_fingerprint(source_checksum="source-a")
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
            method_name="dummy",
            method_version="1",
            method_config={"rounds": 2},
            prompt_bundle_fingerprint="prompt-a",
            generated_contract_version="1",
            selection_policy={"gain": 0.0},
        )
        platinum = platinum_input_fingerprint(
            gold_fingerprint_value=gold,
            model_name="rf",
            model_version="1",
            model_config={},
            metric="auc",
            fold_fingerprint="folds-a",
            evaluation_policy={},
            uncertainty_policy={},
        )

        assert bronze != bronze_fingerprint(source_checksum="source-b")
        assert silver != silver_fingerprint(
            bronze_fingerprint_value=bronze,
            target_name="target",
            task="classification",
            canonicalization_config={"missing": "drop"},
            split_policy={"folds": 3},
            split_seed=42,
        )
        assert platinum != platinum_input_fingerprint(
            gold_fingerprint_value=gold,
            model_name="rf",
            model_version="1",
            model_config={},
            metric="rmse",
            fold_fingerprint="folds-a",
            evaluation_policy={},
            uncertainty_policy={},
        )

    def test_method_identity_is_typed_and_secret_free(self) -> None:
        identity = MethodIdentity(
            registry_name="dummy",
            qualified_class_name="tests.DummyMethod",
            feature_forge_version="0+test",
            llm={"model": "offline", "api_key": "secret"},
        )
        equivalent = identity.model_copy(update={"llm": {"model": "offline", "api_key": "other"}})
        assert identity.fingerprint() == equivalent.fingerprint()


class TestLocalArtifactStore:
    def test_commit_verify_and_resolve(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "artifacts")
        staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="run-1"))
        descriptor = staging.write_json("report.json", {"score": 0.9})
        staging.set_manifest(_manifest("run-1"))

        ref = store.commit(staging)
        assert store.verify(ref).valid
        assert store.resolve(
            ArtifactRef(layer=Layer.GOLD, run_id="run-1", relative_path=descriptor.relative_path)
        ).read_text(encoding="utf-8")

        package = tmp_path / "artifacts/03_gold/runs/run-1"
        assert (package / "_SUCCESS").read_text(encoding="utf-8").strip() == ref.sha256

    def test_corruption_is_detected_and_namespace_is_write_once(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "artifacts")
        namespace = ArtifactNamespace(layer=Layer.SILVER, run_id="run-2")
        first = store.begin(namespace)
        first.write_text("data.txt", "original")
        first.set_manifest(_manifest("run-2", Layer.SILVER))
        ref = store.commit(first)

        package = tmp_path / "artifacts/02_silver/runs/run-2"
        (package / "data.txt").write_text("changed", encoding="utf-8")
        report = store.verify(ref)
        assert not report.valid
        assert "artifact hash mismatch: data.txt" in report.errors

        second = store.begin(namespace)
        second.write_text("data.txt", "replacement")
        second.set_manifest(_manifest("run-2", Layer.SILVER))
        with pytest.raises(FileExistsError, match="already exists"):
            store.commit(second)


class TestResume:
    def test_disabled_resume_executes_all_planned_layers(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "artifacts")
        plan = build_resume_plan(
            store=store,
            run_id="attempt",
            expected_fingerprints={Layer.BRONZE: "bronze-fingerprint"},
            policy=ResumePolicy(enabled=False),
            dry_run=True,
        )
        assert plan.decisions[0].disposition is StageDisposition.EXECUTED
        assert plan.complete is False

    def test_execute_resume_verifies_each_new_package(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path / "artifacts")
        plan = build_resume_plan(
            store=store,
            run_id="attempt",
            expected_fingerprints={Layer.BRONZE: "bronze-fingerprint"},
            policy=ResumePolicy(enabled=False),
        )

        def execute(layer: Layer, upstream: object) -> object:
            del upstream
            staging = store.begin(ArtifactNamespace(layer=layer, run_id="attempt"))
            staging.write_text("payload.txt", layer.value)
            staging.set_manifest(_manifest("attempt", layer))
            return store.commit(staging)

        completed = execute_resume_plan(
            store=store,
            plan=plan,
            execute_stage=execute,  # type: ignore[arg-type]
            semantic_validator=lambda *_args: None,
        )
        assert completed.complete
        assert completed.decisions[0].state is RunState.SUCCEEDED
