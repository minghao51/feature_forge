"""Offline Gold generation, package integrity, and replay tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from feature_forge.config import LLMConfig
from feature_forge.contracts import (
    ArtifactNamespace,
    BronzeRecord,
    CheckResult,
    EnvironmentSnapshot,
    GoldRequest,
    Layer,
    ManifestRef,
    RunManifest,
    RunRequest,
    RunState,
    SilverPackage,
    gold_input_fingerprint,
)
from feature_forge.dataflows.gold import (
    candidate_execution_batches,
    candidate_feature_specs,
    candidate_selection,
    candidate_verification,
    execute_gold,
    gold_manifest,
    gold_materialization,
    load_gold_package,
    method_generation_request,
    replay_gold_package,
)
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import DatasetError, ReplayProviderError
from feature_forge.llm.factory import create_llm_client
from feature_forge.llm.replay import replay_provider_guard
from feature_forge.storage.local import LocalArtifactStore


class DeterministicMethod:
    name = "deterministic"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> DeterministicMethod:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"sum_ab": X["a"] + X["b"]})

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        return self.transform(X_train)

    @property
    def generated_scripts(self) -> list[str]:
        return [
            "def generate_features(df):\n"
            "    return pd.DataFrame({'sum_ab': df['a'] + df['b']}, index=df.index)\n"
        ]

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"name": "sum_ab", "base_columns": ["a", "b"], "agent_name": "fixture"}]

    def get_artifacts(self) -> dict[str, Any]:
        return {}


class RecordedMalmasMethod(DeterministicMethod):
    """Recorded two-round MALMAS-shaped fixture with no provider dependency."""

    name = "malmas"

    @property
    def generated_scripts(self) -> list[str]:
        return [
            "def generate_features(df):\n"
            "    return pd.DataFrame({'first': df['a'] + df['b']}, index=df.index)\n",
            "def generate_features(df):\n"
            "    return pd.DataFrame({'second': df['first'] * 2}, index=df.index)\n",
        ]

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [
            {"name": "first", "base_columns": ["a", "b"], "agent_name": "unary"},
            {"name": "second", "base_columns": ["first"], "agent_name": "cross"},
        ]


class FailedCandidateMethod(DeterministicMethod):
    @property
    def generated_scripts(self) -> list[str]:
        return ["def generate_features(df):\n    raise ValueError('recorded failure')\n"]

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"name": "broken", "base_columns": ["a"]}]


class FailedThenSuccessfulMethod(DeterministicMethod):
    @property
    def generated_scripts(self) -> list[str]:
        return [
            "def generate_features(df):\n    raise ValueError('first batch failed')\n",
            "def generate_features(df):\n"
            "    return pd.DataFrame({'later': df['a'] * 3}, index=df.index)\n",
        ]

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [
            {"name": "broken", "batch_index": 0, "base_columns": ["a"]},
            {"name": "later", "batch_index": 1, "base_columns": ["a"]},
        ]


class RejectedCandidateMethod(DeterministicMethod):
    @property
    def generated_scripts(self) -> list[str]:
        return [
            "def generate_features(df):\n"
            "    return pd.DataFrame({'infinite': [float('inf')] * len(df)}, index=df.index)\n"
        ]

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"name": "infinite", "batch_index": 0, "base_columns": ["a"]}]


class StagedMethod(DeterministicMethod):
    @property
    def generated_scripts(self) -> list[str]:
        raise AssertionError("coarse generated_scripts path should not be used")

    def gold_generation_batches(self) -> list[dict[str, Any]]:
        return [
            {
                "code": (
                    "def generate_features(df):\n"
                    "    return pd.DataFrame({'staged': df['a'] - df['b']}, index=df.index)\n"
                ),
                "specifications": [{"name": "staged", "base_columns": ["a", "b"]}],
            }
        ]


def _silver(store: LocalArtifactStore) -> tuple[SilverPackage, ManifestRef]:
    bronze = BronzeRecord(
        source="fixture",
        source_checksum="source",
        target="target",
        task="classification",
        snapshot_mode="snapshot",
        row_count=3,
        column_count=3,
    )
    bronze_manifest = RunManifest(
        layer=Layer.BRONZE,
        package_kind="bronze",
        layer_fingerprint="bronze-fingerprint",
        run_id="bronze-run",
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(
            dataset="fixture",
            method="bronze",
            model="none",
            seed=42,
            options={
                "source_policy": "snapshot",
                "source": {"source_checksum": "source"},
            },
        ),
        environment=_environment(),
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    bronze_staging = store.begin(ArtifactNamespace(layer=Layer.BRONZE, run_id="bronze-run"))
    bronze_staging.write_json("bronze.json", bronze.model_dump(mode="json"))
    bronze_staging.write_dataframe(
        "train.parquet",
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]}),
    )
    bronze_staging.set_manifest(bronze_manifest)
    bronze_ref = store.commit(bronze_staging)

    manifest = RunManifest(
        layer=Layer.SILVER,
        package_kind="silver",
        layer_fingerprint="silver-fingerprint",
        run_id="silver-run",
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
    row_ids = pd.DataFrame({"row_id": ["r0", "r1", "r2"]})
    features = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    target = pd.DataFrame({"row_id": row_ids.row_id, "target": [0, 1, 0]})
    folds = pd.DataFrame({"row_id": row_ids.row_id, "fold": [0, 1, 0]})
    checks = [CheckResult(check_id="SILVER.OK", passed=True, message="ok")]
    staging = store.begin(ArtifactNamespace(layer=Layer.SILVER, run_id="silver-run"))
    staging.write_dataframe("canonical_features.parquet", features)
    staging.write_dataframe("canonical_target.parquet", target)
    staging.write_dataframe("row_ids.parquet", row_ids)
    staging.write_dataframe("fold_assignments.parquet", folds)
    staging.write_json("profile.json", {"dataset_fingerprint": "silver-fingerprint"})
    staging.write_json("checks.json", [item.model_dump(mode="json") for item in checks])
    staging.set_manifest(manifest)
    ref = store.commit(staging)
    committed = store.load_manifest(ref)
    return (
        SilverPackage(
            manifest=committed,
            bronze=bronze,
            canonical_features=features,
            canonical_target=target,
            row_ids=row_ids,
            fold_assignments=folds,
            profile={"dataset_fingerprint": "silver-fingerprint"},
            checks=checks,
        ),
        ref,
    )


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.13",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
        random_seeds={"method": 42},
    )


def _request(
    ref: ManifestRef,
    *,
    persist_candidates: bool = True,
    method_name: str = "deterministic",
    run_id: str | None = None,
) -> GoldRequest:
    identity = gold_input_fingerprint(
        silver_fingerprint_value="silver-fingerprint",
        method_name=method_name,
        method_version="1",
        method_config={"n": 1},
        prompt_bundle_fingerprint="no-prompts",
        generated_contract_version="1",
        selection_policy={"policy": "validation"},
    )
    return GoldRequest(
        run_id=run_id or ("gold-run" if persist_candidates else "gold-run-light"),
        case_fingerprint="case-fingerprint",
        silver_manifest=ref,
        silver_fingerprint="silver-fingerprint",
        gold_input_fingerprint=identity,
        method_name=method_name,
        method_version="1",
        method_config={"n": 1},
        prompt_bundle_fingerprint="no-prompts",
        persist_candidates=persist_candidates,
    )


def _evidence(
    method: DeterministicMethod,
    silver: SilverPackage,
    request: GoldRequest,
    sandbox: SandboxedExecutor,
) -> Any:
    validated = method_generation_request(request)
    plan = candidate_feature_specs(method, silver, validated)
    executed = candidate_execution_batches(plan, sandbox)
    verified = candidate_verification(executed)
    return candidate_selection(verified)


@pytest.mark.parametrize("persist_candidates", [True, False])
def test_deterministic_gold_package_replays_offline(
    tmp_path: Path, persist_candidates: bool
) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    silver, silver_ref = _silver(store)
    request = _request(silver_ref, persist_candidates=persist_candidates)
    sandbox = SandboxedExecutor(timeout_seconds=5)
    evidence = _evidence(DeterministicMethod(), silver, request, sandbox)
    materialized = gold_materialization(
        evidence,
        request,
        _environment(),
        ExecutionProfile.DEVELOPMENT,
        store,
        gold_manifest(evidence, request, _environment()),
    )
    assert materialized.manifest_ref is not None
    package = replay_gold_package(store, materialized.manifest_ref, silver, sandbox)

    assert package.accepted_features.to_dict("list") == {
        "row_id": ["r0", "r1", "r2"],
        "sum_ab": [5, 7, 9],
    }
    assert (package.candidate_features is not None) is persist_candidates
    assert len(package.candidates) == len(package.provenance) == len(package.decisions) == 1
    assert package.manifest.request.options["selection_semantics"] == "validation-accepted"


@pytest.mark.parametrize(
    "relative_path",
    [
        "generated_code/batch_0000.py",
        "candidates.json",
        "decisions.json",
        "accepted_features.parquet",
    ],
)
def test_corrupt_gold_artifacts_fail_verified_loading(tmp_path: Path, relative_path: str) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    silver, silver_ref = _silver(store)
    request = _request(silver_ref)
    result = gold_materialization(
        (
            evidence := _evidence(
                DeterministicMethod(), silver, request, SandboxedExecutor(timeout_seconds=5)
            )
        ),
        request,
        _environment(),
        ExecutionProfile.DEVELOPMENT,
        store,
        gold_manifest(evidence, request, _environment()),
    )
    assert result.manifest_ref is not None
    artifact_path = tmp_path / "lake/03_gold/runs/gold-run" / relative_path
    artifact_path.write_bytes(artifact_path.read_bytes() + b"corrupt")
    with pytest.raises(DatasetError, match="Existing Gold package failed verification"):
        gold_materialization(
            evidence,
            request,
            _environment(),
            ExecutionProfile.DEVELOPMENT,
            store,
            gold_manifest(evidence, request, _environment()),
        )
    with pytest.raises(DatasetError, match="Unable to load verified Gold package"):
        load_gold_package(store, result.manifest_ref)


@pytest.mark.asyncio
async def test_replay_guard_blocks_injected_client_completion(fake_llm: Any) -> None:
    with replay_provider_guard(), pytest.raises(ReplayProviderError, match="forbidden"):
        await fake_llm.complete([{"role": "user", "content": "must not run"}])
    with replay_provider_guard(), pytest.raises(ReplayProviderError, match="JSON"):
        await fake_llm.complete_json(
            [{"role": "user", "content": "must not run"}], schema_description="{}"
        )


def test_replay_guard_blocks_provider_construction() -> None:
    with replay_provider_guard(), pytest.raises(ReplayProviderError, match="construction"):
        create_llm_client(LLMConfig(model="gpt-4", provider="openai"))


def test_recorded_malmas_batches_replay_cumulatively_without_provider(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    silver, silver_ref = _silver(store)
    request = _request(
        silver_ref,
        run_id="gold-malmas",
        method_name="malmas",
    )
    sandbox = SandboxedExecutor(timeout_seconds=5)
    result = gold_materialization(
        (evidence := _evidence(RecordedMalmasMethod(), silver, request, sandbox)),
        request,
        _environment(),
        ExecutionProfile.DEVELOPMENT,
        store,
        gold_manifest(evidence, request, _environment()),
    )
    assert result.manifest_ref is not None

    package = replay_gold_package(store, result.manifest_ref, silver, sandbox)

    assert package.accepted_features["second"].tolist() == [10, 14, 18]
    assert len(package.code) == 2


def test_failed_candidate_is_committed_and_replayed_as_evidence(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    silver, silver_ref = _silver(store)
    request = _request(silver_ref, run_id="gold-failure")
    sandbox = SandboxedExecutor(timeout_seconds=5)
    evidence = _evidence(FailedCandidateMethod(), silver, request, sandbox)
    assert evidence.decisions[0].state.value == "error"
    result = gold_materialization(
        evidence,
        request,
        _environment(),
        ExecutionProfile.DEVELOPMENT,
        store,
        gold_manifest(evidence, request, _environment()),
    )
    assert result.manifest_ref is not None

    package = replay_gold_package(store, result.manifest_ref, silver, sandbox)

    assert package.decisions[0].reason_code == "CodeExecutionError"
    assert package.accepted_features.columns.tolist() == ["row_id"]


def test_failed_batch_ownership_preserves_later_success(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    silver, silver_ref = _silver(store)
    request = _request(
        silver_ref,
        persist_candidates=False,
        run_id="gold-partial-failure",
    )
    sandbox = SandboxedExecutor(timeout_seconds=5)
    evidence = _evidence(FailedThenSuccessfulMethod(), silver, request, sandbox)

    assert [item.name for item in evidence.candidates] == ["broken", "later"]
    assert [item.state.value for item in evidence.decisions] == ["error", "accepted"]
    assert evidence.accepted_features["later"].tolist() == [3, 6, 9]

    result = gold_materialization(
        evidence,
        request,
        _environment(),
        ExecutionProfile.DEVELOPMENT,
        store,
        gold_manifest(evidence, request, _environment()),
    )
    assert result.manifest_ref is not None
    replayed = replay_gold_package(store, result.manifest_ref, silver, sandbox)
    assert replayed.accepted_features["later"].tolist() == [3, 6, 9]


def test_rejected_outcome_replays_without_candidate_matrix(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    silver, silver_ref = _silver(store)
    request = _request(
        silver_ref,
        persist_candidates=False,
        run_id="gold-rejected",
    )
    sandbox = SandboxedExecutor(timeout_seconds=5)
    evidence = _evidence(RejectedCandidateMethod(), silver, request, sandbox)
    assert evidence.decisions[0].state.value == "rejected"
    result = gold_materialization(
        evidence,
        request,
        _environment(),
        ExecutionProfile.DEVELOPMENT,
        store,
        gold_manifest(evidence, request, _environment()),
    )
    assert result.manifest_ref is not None

    package = replay_gold_package(store, result.manifest_ref, silver, sandbox)

    assert package.candidate_features is None
    assert package.counts.failure_counts == {"non_finite_values": 1}


def test_gold_request_boundary_rejects_stale_input_fingerprint(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    _, silver_ref = _silver(store)
    stale = _request(silver_ref).model_copy(update={"method_config": {"n": 2}})

    with pytest.raises(ValueError, match="gold_input_fingerprint"):
        method_generation_request(stale)


def test_replay_rejects_unverified_in_memory_silver(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    silver, silver_ref = _silver(store)
    request = _request(silver_ref)
    sandbox = SandboxedExecutor(timeout_seconds=5)
    evidence = _evidence(DeterministicMethod(), silver, request, sandbox)
    result = gold_materialization(
        evidence,
        request,
        _environment(),
        ExecutionProfile.DEVELOPMENT,
        store,
        gold_manifest(evidence, request, _environment()),
    )
    assert result.manifest_ref is not None
    mismatched = silver.model_copy(
        update={"canonical_features": silver.canonical_features.assign(a=[9, 9, 9])}
    )

    with pytest.raises(DatasetError, match="differs from its verified manifest"):
        replay_gold_package(store, result.manifest_ref, mismatched, sandbox)


def test_explicit_gold_execution_reports_exact_counts(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    _, silver_ref = _silver(store)
    result = execute_gold(
        store=store,
        request=_request(silver_ref, run_id="gold-entrypoint"),
        method=DeterministicMethod(),
        sandbox=SandboxedExecutor(timeout_seconds=5),
        environment=_environment(),
        profile=ExecutionProfile.DEVELOPMENT,
    )

    assert result.counts.model_dump() == {
        "schema_version": "1",
        "candidates_proposed": 1,
        "candidates_executed": 1,
        "candidates_accepted": 1,
        "accepted_output_columns": 1,
        "failure_counts": {},
    }


def test_optional_staged_method_bypasses_coarse_adapter(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    silver, silver_ref = _silver(store)

    evidence = _evidence(
        StagedMethod(),
        silver,
        _request(silver_ref, run_id="gold-staged"),
        SandboxedExecutor(timeout_seconds=5),
    )

    assert evidence.accepted_features["staged"].tolist() == [-3, -3, -3]
