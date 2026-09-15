"""Hamilton package corruption-recovery drills (plan 21 §16 legacy-removal gate 4).

Covers the corruption-recovery requirements that gate legacy removal in
``docs/plan/21_hamilton_default_execution_handoff.md``:

* Deleting the Hamilton node cache leaves verified durable packages reusable
  (§16 gate 4 "cache deletion"; §9.4 "cache deletion or corruption falls back
  to recomputation without affecting durable reuse").
* A hash-tampered Silver package fails verification, is skipped as a cross-run
  reuse candidate, and the affected suffix recomputes successfully on a new
  attempt (§10 "a corrupt or incompatible package stops reuse at that layer";
  ``LocalArtifactStore.find_reusable`` returns only hash/manifest-verified
  packages while an invalid preferred package fails closed).
* Provider-free Gold replay stays possible from verified Gold evidence after
  recovery (§14.2 "Gold replay with provider construction forbidden").

Every test allocates fresh temporary artifact and cache roots; no repository
or legacy artifact root is touched, and no network or provider is required.
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Any

import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.contracts import (
    ArtifactNamespace,
    ArtifactRef,
    Layer,
    ManifestRef,
)
from feature_forge.data import DatasetRegistry
from feature_forge.dataflows._io import load_silver_package, replay_gold_package
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.experiment.execution import ExperimentCase, ExperimentResult
from feature_forge.experiment.hamilton_executor import HamiltonLayerExecutor
from feature_forge.methods import MethodRegistry
from feature_forge.methods.base import BaseMethod
from feature_forge.storage.local import LocalArtifactStore

ATTEMPT_ID = "recovery-attempt-1"
NEXT_ATTEMPT_ID = "recovery-attempt-2"
DATASET_NAME = "recovery-xor"
METHOD_NAME = "recovery_deterministic"


class RecoveryDeterministicMethod(BaseMethod):
    """Offline deterministic method emitting one sandbox-safe numeric feature."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name=METHOD_NAME)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> RecoveryDeterministicMethod:
        self._artifacts["generated_code"] = (
            "import pandas as pd\n"
            "def generate_features(df):\n"
            "    return pd.DataFrame({'diag_sq': (df['a'] - df['b']) ** 2}, index=df.index)\n"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(diag_sq=(X["a"] - X["b"]) ** 2)

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"name": "diag_sq", "base_columns": ["a", "b"]}]


class InlineReplaySandbox(SandboxedExecutor):
    """Deterministic in-process stand-in for the process-isolated sandbox.

    Replaying committed Gold evidence must not depend on sandbox subprocess
    scheduling to stay deterministic; the generated code is executed in-process
    against the same contract.
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
        exec(compile(code, "<inline-replay>", "exec"), namespace)
        generated = namespace["generate_features"]
        assert callable(generated)
        result = generated(df)
        assert isinstance(result, pd.DataFrame)
        return result


def _local_numeric_dataset(tmp_path: pathlib.Path) -> dict[str, Any]:
    """Write the deterministic XOR-style dataset and return its registry info."""
    frame = pd.DataFrame(
        {
            "a": [0.0, 0.0, 1.0, 1.0, 0.1, 0.2, 0.9, 0.8],
            "b": [0.0, 1.0, 0.0, 1.0, 0.9, 0.8, 0.1, 0.2],
            "c": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [0, 1, 1, 0, 1, 1, 1, 1],
        }
    )
    sample_dir = tmp_path / "recovery-xor-data"
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(sample_dir / "train.csv", index=False)
    (sample_dir / "metadata.json").write_text('{"target": "target", "task": "classification"}')
    return {
        "source": "local",
        "path": str(sample_dir),
        "target": "target",
        "task": "classification",
    }


def _settings(tmp_path: pathlib.Path) -> Settings:
    """Hamilton settings with one fresh artifact root and one fresh cache root."""
    return Settings(
        metric="acc",
        dataflow={
            "engine": "hamilton",
            "artifact_root": str(tmp_path / "artifacts"),
            "cache": {"path": str(tmp_path / "hamilton-cache")},
        },
    )


def _executor(tmp_path: pathlib.Path, dataset_info: dict[str, Any]) -> HamiltonLayerExecutor:
    settings = _settings(tmp_path)
    registry = DatasetRegistry()
    registry.register(DATASET_NAME, dataset_info)
    return HamiltonLayerExecutor(
        settings=settings,
        artifact_store=LocalArtifactStore(settings.dataflow.artifact_root),
        dataset_registry=registry,
        method_classes={
            **MethodRegistry.get_all_methods(),
            METHOD_NAME: RecoveryDeterministicMethod,
        },
    )


def _run_attempt(
    tmp_path: pathlib.Path, dataset_info: dict[str, Any], attempt_id: str
) -> ExperimentResult:
    case = ExperimentCase(
        dataset=DATASET_NAME,
        method=METHOD_NAME,
        model="random_forest",
        seed=42,
        cv_folds=2,
        run_id=attempt_id,
        attempt_id=attempt_id,
    )
    return _executor(tmp_path, dataset_info).execute_case(case)


def _stage_map(result: ExperimentResult) -> dict[Layer, Any]:
    return {stage.layer: stage for stage in result.stages}


def _tamper_silver_features(store: LocalArtifactStore, attempt_id: str) -> pathlib.Path:
    """Rewrite one committed Silver artifact so its descriptor hash no longer matches."""
    path = (
        store.package_path(ArtifactNamespace(layer=Layer.SILVER, run_id=attempt_id))
        / "canonical_features.parquet"
    )
    frame = pd.read_parquet(path)
    frame.loc[0, frame.columns[0]] = 999.0
    frame.to_parquet(path)
    return path


def _dispositions(result: ExperimentResult) -> list[str]:
    return [stage.disposition.value for stage in result.stages]


class TestCacheDeletionReuse:
    """Deleting the Hamilton node cache keeps verified durable packages reusable."""

    def test_cache_deletion_leaves_verified_packages_reusable(self, tmp_path: pathlib.Path) -> None:
        dataset_info = _local_numeric_dataset(tmp_path)
        first = _run_attempt(tmp_path, dataset_info, ATTEMPT_ID)
        assert _dispositions(first) == ["executed"] * 4
        settings = _settings(tmp_path)
        cache_root = pathlib.Path(settings.dataflow.cache.path)  # type: ignore[arg-type]
        assert cache_root.is_dir()

        shutil.rmtree(cache_root)
        assert not cache_root.exists()

        # A same-namespace retry finds only verified durable evidence: the
        # Silver/Gold/Platinum packages are reused without any Hamilton cache.
        second = _run_attempt(tmp_path, dataset_info, ATTEMPT_ID)
        assert second.error is None
        assert _dispositions(second) == ["executed", "reused", "reused", "reused"]

        store = LocalArtifactStore(settings.dataflow.artifact_root)
        stages = _stage_map(first)
        silver_stage = stages[Layer.SILVER]
        silver_ref = silver_stage.manifest_ref
        assert isinstance(silver_ref, ManifestRef)
        assert store.verify(silver_ref).valid
        reusable = store.find_reusable(
            silver_stage.reuse_fingerprint,
            Layer.SILVER,
            upstreams=[stages[Layer.BRONZE].manifest_ref],
            preferred_run_id=ATTEMPT_ID,
        )
        assert reusable == silver_ref

    def test_corrupt_cache_metadata_is_quarantined_before_reuse(
        self, tmp_path: pathlib.Path
    ) -> None:
        dataset_info = _local_numeric_dataset(tmp_path)
        first = _run_attempt(tmp_path, dataset_info, ATTEMPT_ID)
        assert first.error is None
        settings = _settings(tmp_path)
        cache_root = pathlib.Path(settings.dataflow.cache.path)  # type: ignore[arg-type]
        for name in ("metadata_store.db-wal", "metadata_store.db-shm", "metadata_store.db"):
            (cache_root / name).unlink(missing_ok=True)
        (cache_root / "metadata_store.db").write_bytes(b"corrupt sqlite metadata")

        second = _run_attempt(tmp_path, dataset_info, NEXT_ATTEMPT_ID)

        assert second.error is None
        assert _dispositions(second) == ["reused"] * 4
        assert list(cache_root.glob("metadata_store.db.corrupt-*"))
        assert (
            LocalArtifactStore(settings.dataflow.artifact_root)
            .verify(_stage_map(first)[Layer.PLATINUM].manifest_ref)
            .valid
        )


class TestTamperedSilverRecovery:
    """A hash-tampered Silver package stops reuse and the suffix recomputes."""

    def test_invalid_preferred_package_fails_closed(self, tmp_path: pathlib.Path) -> None:
        dataset_info = _local_numeric_dataset(tmp_path)
        first = _run_attempt(tmp_path, dataset_info, ATTEMPT_ID)
        store = LocalArtifactStore(_settings(tmp_path).dataflow.artifact_root)
        stages = _stage_map(first)
        bronze_ref = stages[Layer.BRONZE].manifest_ref
        silver_ref = stages[Layer.SILVER].manifest_ref
        assert isinstance(silver_ref, ManifestRef)
        _tamper_silver_features(store, ATTEMPT_ID)

        report = store.verify(silver_ref)
        assert not report.valid
        assert any("canonical_features.parquet" in error for error in report.errors)

        # The preferred (current-attempt) package fails closed instead of
        # silently returning unverified evidence.
        with pytest.raises(ValueError, match="preferred silver package is invalid"):
            store.find_reusable(
                stages[Layer.SILVER].reuse_fingerprint,
                Layer.SILVER,
                upstreams=[bronze_ref],
                preferred_run_id=ATTEMPT_ID,
            )

    def test_tampered_silver_is_skipped_and_suffix_recomputes_on_new_attempt(
        self, tmp_path: pathlib.Path
    ) -> None:
        dataset_info = _local_numeric_dataset(tmp_path)
        first = _run_attempt(tmp_path, dataset_info, ATTEMPT_ID)
        store = LocalArtifactStore(_settings(tmp_path).dataflow.artifact_root)
        first_stages = _stage_map(first)
        silver_ref = first_stages[Layer.SILVER].manifest_ref
        assert isinstance(silver_ref, ManifestRef)
        tampered = _tamper_silver_features(store, ATTEMPT_ID)
        assert store.verify(silver_ref).valid is False

        # Cross-run candidates that fail verification are skipped so policy can
        # recompute; they never surface as reuse references.
        assert (
            store.find_reusable(
                first_stages[Layer.SILVER].reuse_fingerprint,
                Layer.SILVER,
                upstreams=[first_stages[Layer.BRONZE].manifest_ref],
                preferred_run_id=NEXT_ATTEMPT_ID,
            )
            is None
        )

        second = _run_attempt(tmp_path, dataset_info, NEXT_ATTEMPT_ID)
        assert second.error is None
        # The tampered Silver stopped durable reuse exactly at Silver: the
        # verified Bronze package before it is still reused, while the damaged
        # layer and every later layer recomputed on the new attempt.
        assert _dispositions(second) == ["reused", "executed", "executed", "executed"]
        assert second.stages[0].manifest_ref is not None
        assert second.stages[0].manifest_ref.run_id == ATTEMPT_ID
        assert second.stages[1].manifest_ref is not None
        assert second.stages[1].manifest_ref.run_id == NEXT_ATTEMPT_ID
        # Deterministic recomputation reproduces the content-scoped Silver
        # identity and the metric outputs of the damaged attempt.
        assert [(stage.layer.value, stage.layer_fingerprint) for stage in second.stages[:2]] == [
            (stage.layer.value, stage.layer_fingerprint) for stage in first.stages[:2]
        ]
        assert second.cv_score == first.cv_score
        assert second.gain == first.gain

        second_silver_ref = _stage_map(second)[Layer.SILVER].manifest_ref
        assert isinstance(second_silver_ref, ManifestRef)
        assert second_silver_ref.run_id == NEXT_ATTEMPT_ID
        assert second_silver_ref != silver_ref
        assert store.verify(second_silver_ref).valid
        # The damaged package stays in its write-once namespace, still invalid.
        assert store.verify(silver_ref).valid is False
        assert tampered.is_file()


class TestGoldReplayAfterRecovery:
    """Verified Gold evidence supports provider-free replay."""

    def test_verified_gold_evidence_replays_without_provider(self, tmp_path: pathlib.Path) -> None:
        dataset_info = _local_numeric_dataset(tmp_path)
        first = _run_attempt(tmp_path, dataset_info, ATTEMPT_ID)
        assert first.error is None
        store = LocalArtifactStore(_settings(tmp_path).dataflow.artifact_root)
        stages = _stage_map(first)
        silver_ref = stages[Layer.SILVER].manifest_ref
        gold_ref = stages[Layer.GOLD].manifest_ref
        assert isinstance(silver_ref, ManifestRef)
        assert isinstance(gold_ref, ManifestRef)
        assert store.verify(gold_ref).valid

        silver = load_silver_package(store, silver_ref)
        committed = pd.read_parquet(
            store.resolve(
                ArtifactRef(
                    layer=Layer.GOLD,
                    run_id=gold_ref.run_id,
                    relative_path="accepted_features.parquet",
                )
            )
        )
        replayed = replay_gold_package(
            store, gold_ref, silver, InlineReplaySandbox(timeout_seconds=30)
        )
        again = replay_gold_package(
            store, gold_ref, silver, InlineReplaySandbox(timeout_seconds=30)
        )
        # Replay reconstructs the committed evidence exactly, without any
        # provider construction (replay runs under the provider guard).
        assert replayed.accepted_features.to_dict("list") == committed.to_dict("list")
        assert again.accepted_features.to_dict("list") == committed.to_dict("list")
        assert replayed.counts.accepted_output_columns == 1
        assert replayed.accepted_features.columns.tolist() == ["row_id", "diag_sq"]
