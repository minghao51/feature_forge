"""Offline Gold generation and replay coverage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from feature_forge.contracts import (
    BronzeRecord,
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
from feature_forge.dataflows.driver import build_gold_driver
from feature_forge.dataflows.gold import (
    candidate_execution_batches,
    candidate_feature_specs,
    candidate_selection,
    candidate_verification,
    gold_manifest,
    gold_materialization,
    replay_gold_package,
)
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import DatasetError, ReplayProviderError
from feature_forge.llm.replay import ensure_provider_allowed, replay_provider_guard
from feature_forge.storage.local import LocalArtifactStore


class _DeterministicMethod:
    generated_scripts: ClassVar[list[str]] = [
        "def generate_features(df):\n"
        "    return pd.DataFrame({'sum_ab': df['a'] + df['b']}, index=df.index)\n"
    ]
    feature_metadata: ClassVar[list[dict[str, object]]] = [
        {"name": "sum_ab", "base_columns": ["a", "b"]}
    ]


class _InlineSandbox(SandboxedExecutor):
    """Deterministic in-process stand-in for the process-isolated sandbox.

    Spawning sandbox subprocesses from Hamilton worker threads is unreliable in
    constrained environments; this subclass keeps the driver-wiring test
    offline and deterministic while still validating the produced columns.
    """

    def execute(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
    ) -> pd.DataFrame:
        namespace: dict[str, object] = {"pd": pd}
        exec(compile(code, "<inline>", "exec"), namespace)
        generated = namespace["generate_features"]
        assert callable(generated)
        result = generated(df)
        assert isinstance(result, pd.DataFrame)
        return result


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.12",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
    )


def _silver() -> SilverPackage:
    now = datetime.now(UTC)
    manifest = RunManifest(
        layer=Layer.SILVER,
        package_kind="silver",
        layer_fingerprint="silver-fingerprint",
        run_id="silver-run",
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="fixture", method="silver", model="none", seed=42),
        environment=_environment(),
        created_at=now,
        completed_at=now,
    )
    row_ids = ["row-000000000", "row-000000001", "row-000000002"]
    features = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    return SilverPackage(
        manifest=manifest,
        bronze=BronzeRecord(
            source="fixture",
            source_checksum="source",
            target="target",
            task="classification",
            snapshot_mode="snapshot",
            row_count=3,
            column_count=3,
        ),
        canonical_features=features,
        canonical_target=pd.DataFrame({"row_id": row_ids, "target": [0, 1, 0]}),
        row_ids=pd.DataFrame({"row_id": row_ids}),
        fold_assignments=pd.DataFrame({"row_id": row_ids, "fold": [0, 1, 0]}),
        profile={"dataset_fingerprint": "silver-fingerprint"},
        checks=[],
    )


def _request() -> GoldRequest:
    value = gold_input_fingerprint(
        silver_fingerprint_value="silver-fingerprint",
        method_name="deterministic",
        method_version="1",
        method_config={},
        prompt_bundle_fingerprint="fixture",
        generated_contract_version="1",
        selection_policy={"policy": "validation"},
    )
    return GoldRequest(
        run_id="gold-run",
        case_fingerprint="case-fingerprint",
        silver_manifest=ManifestRef(layer=Layer.SILVER, run_id="silver-run", sha256="a" * 64),
        silver_fingerprint="silver-fingerprint",
        gold_input_fingerprint=value,
        method_name="deterministic",
        method_version="1",
        prompt_bundle_fingerprint="fixture",
    )


def _evidence() -> tuple[dict[str, object], SilverPackage, GoldRequest]:
    silver = _silver()
    request = _request()
    evidence = candidate_feature_specs(_DeterministicMethod(), silver, request)
    evidence = candidate_execution_batches(evidence, SandboxedExecutor(timeout_seconds=5))
    evidence = candidate_selection(candidate_verification(evidence))
    return evidence, silver, request


def _node_tags(function: object) -> dict[str, str]:
    tags: dict[str, str] = {}
    for decorator in getattr(function, "decorate_nodes", []):
        tags.update(getattr(decorator, "tags", {}))
        tags.update(getattr(decorator, "cache_tags", {}))
    return tags


def test_per_attempt_manifest_node_is_marked_recompute() -> None:
    assert _node_tags(gold_manifest)["persistence"] == "recompute"
    assert _node_tags(gold_manifest).get("cache.behavior") == "recompute"


def test_gold_driver_executes_materialization_in_one_call(tmp_path: Path) -> None:
    """``execute(['gold_materialization'])`` resolves the manifest node itself."""
    artifact_store = LocalArtifactStore(tmp_path / "lake")
    driver = build_gold_driver(
        profile=ExecutionProfile.DEVELOPMENT,
        artifact_store=artifact_store,
        cache_dir=tmp_path / "hamilton-cache",
    )
    result = driver.execute(
        ["gold_manifest", "gold_materialization"],
        inputs={
            "method": _DeterministicMethod(),
            "verified_silver_package": _silver(),
            "gold_request": _request(),
            "environment_snapshot": _environment(),
            "sandbox": _InlineSandbox(),
            "artifact_store": artifact_store,
        },
    )
    materialization = result["gold_materialization"]

    assert materialization.persisted is True
    assert materialization.manifest_ref is not None
    assert materialization.manifest.layer is Layer.GOLD
    assert materialization.manifest.layer_fingerprint == result["gold_manifest"].layer_fingerprint
    assert materialization.counts.candidates_accepted == 1


def test_gold_materialization_rejects_mismatched_manifest(tmp_path: Path) -> None:
    evidence, _silver, request = _evidence()
    manifest = gold_manifest(evidence, request, _environment())
    with pytest.raises(DatasetError, match="does not match selected evidence"):
        gold_materialization(
            evidence,
            request,
            _environment(),
            ExecutionProfile.DEVELOPMENT,
            LocalArtifactStore(tmp_path / "lake"),
            manifest.model_copy(update={"layer_fingerprint": "bogus-fingerprint"}),
        )


def test_deterministic_gold_persists_and_replays_without_provider(tmp_path: Path) -> None:
    evidence, silver, request = _evidence()
    store = LocalArtifactStore(tmp_path / "lake")
    materialization = gold_materialization(
        evidence,
        request,
        _environment(),
        ExecutionProfile.DEVELOPMENT,
        store,
    )

    assert materialization.manifest_ref is not None
    package = replay_gold_package(
        store,
        materialization.manifest_ref,
        silver,
        SandboxedExecutor(timeout_seconds=5),
    )
    assert package.accepted_features.to_dict("list") == {
        "row_id": ["row-000000000", "row-000000001", "row-000000002"],
        "sum_ab": [5, 7, 9],
    }
    assert package.counts.accepted_output_columns == 1


def test_replay_guard_blocks_provider_boundary() -> None:
    with replay_provider_guard(), pytest.raises(ReplayProviderError):
        ensure_provider_allowed()


def test_replay_rejects_changed_committed_values(tmp_path: Path) -> None:
    evidence, silver, request = _evidence()
    store = LocalArtifactStore(tmp_path / "lake")
    materialization = gold_materialization(
        evidence,
        request,
        _environment(),
        ExecutionProfile.DEVELOPMENT,
        store,
    )
    assert materialization.manifest_ref is not None
    path = tmp_path / "lake/03_gold/runs/gold-run/accepted_features.parquet"
    corrupted = pd.read_parquet(path)
    corrupted.loc[0, "sum_ab"] = 999
    corrupted.to_parquet(path)
    with pytest.raises(DatasetError):
        replay_gold_package(
            store,
            materialization.manifest_ref,
            silver,
            SandboxedExecutor(timeout_seconds=5),
        )
