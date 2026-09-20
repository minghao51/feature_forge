"""Platinum v2 identity reuse rejection and directional result-field coverage.

Covers plan 23 PR 4 slice C (ADR 0018 decisions 6-8): the versioned Platinum
identity schema (``platinum_input_fingerprint`` gains
``identity_schema_version``, so pre-ADR v1 packages stay integrity-verified
and loadable but can never satisfy v2 reuse fingerprints), the manifest
provenance option, the additive ``ExperimentResult`` directional fields,
tracker rendering of the directional interval, and executor population of the
new fields.

Fixture style (``_packages``/``_materialized_package``) is copied from
``tests/unit/test_platinum_dataflow.py`` and the executor-population pattern
from ``tests/integration/test_hamilton_recovery.py``; both are deliberately
duplicated so these units stay reviewable independently of those files.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from feature_forge.config import Settings
from feature_forge.contracts import (
    ArtifactNamespace,
    BronzeRecord,
    EnvironmentSnapshot,
    FeatureCandidate,
    FeatureDecision,
    FeatureDecisionState,
    FeatureProvenance,
    GoldFeatureCounts,
    GoldPackage,
    GoldRequest,
    Layer,
    ManifestRef,
    PlatinumRequest,
    RunManifest,
    RunRequest,
    RunState,
    SilverPackage,
)
from feature_forge.contracts.identity import (
    PLATINUM_IDENTITY_SCHEMA_VERSION,
    gold_input_fingerprint,
    platinum_input_fingerprint,
    silver_fingerprint,
)
from feature_forge.data import DatasetRegistry
from feature_forge.dataflows.platinum import (
    aggregate_metrics,
    baseline_fold_evidence,
    build_platinum_request,
    candidate_fold_evidence,
    load_platinum_package,
    platinum_checks,
    platinum_manifest,
    platinum_materialization,
    platinum_request,
    selection_decisions,
    uncertainty_summary,
)
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.experiment import ExperimentTracker
from feature_forge.experiment.execution import ExperimentCase, ExperimentResult
from feature_forge.experiment.hamilton_executor import HamiltonLayerExecutor
from feature_forge.methods import BaseMethod, MethodRegistry
from feature_forge.platform import ExperimentalPlatform
from feature_forge.storage.local import LocalArtifactStore


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.12",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
    )


def _packages() -> tuple[SilverPackage, GoldPackage, ManifestRef]:
    """Build a verified Silver/Gold fixture pair (style of test_platinum_dataflow)."""
    now = datetime.now(UTC)
    silver_ref = ManifestRef(layer=Layer.SILVER, run_id="silver-run", sha256="a" * 64)
    gold_ref = ManifestRef(layer=Layer.GOLD, run_id="gold-run", sha256="b" * 64)
    silver_manifest = RunManifest(
        layer=Layer.SILVER,
        package_kind="silver",
        layer_fingerprint="silver-fingerprint",
        run_id="silver-run",
        case_fingerprint="case",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="fixture", method="silver", model="none", seed=42),
        environment=_environment(),
        created_at=now,
        completed_at=now,
    )
    row_ids = [f"row-{index:09d}" for index in range(6)]
    features = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
    silver = SilverPackage(
        manifest=silver_manifest,
        bronze=BronzeRecord(
            source="fixture",
            source_checksum="source",
            target="target",
            task="classification",
            snapshot_mode="snapshot",
            row_count=6,
            column_count=2,
        ),
        canonical_features=features,
        canonical_target=pd.DataFrame({"row_id": row_ids, "target": [0, 0, 0, 1, 1, 1]}),
        row_ids=pd.DataFrame({"row_id": row_ids}),
        fold_assignments=pd.DataFrame(
            {
                "row_id": row_ids,
                "fold": [0, 1, 2, 0, 1, 2],
                # Post-ADR-0018 Silver always labels partitions; holdout
                # platinum scopes fail closed on a missing column.
                "partition": ["discovery"] * 3 + ["evaluation"] * 3,
            }
        ),
        profile={},
        checks=[],
    )
    gold_input = gold_input_fingerprint(
        silver_fingerprint_value="silver-fingerprint",
        method_name="fixture",
        method_version="1",
        method_config={},
        prompt_bundle_fingerprint="fixture",
        generated_contract_version="1",
        selection_policy={"policy": "validation"},
    )
    gold_request = GoldRequest(
        run_id="gold-run",
        case_fingerprint="case",
        silver_manifest=silver_ref,
        silver_fingerprint="silver-fingerprint",
        gold_input_fingerprint=gold_input,
        method_name="fixture",
        method_version="1",
        prompt_bundle_fingerprint="fixture",
    )
    candidate = FeatureCandidate(
        candidate_id="candidate-0",
        name="double_a",
        batch_index=0,
        code_path="generated_code/batch_0000.py",
        specification={},
    )
    provenance = [
        FeatureProvenance(
            candidate_id="candidate-0",
            method="fixture",
            method_version="1",
            prompt_bundle_fingerprint="fixture",
            code_sha256="c" * 64,
        )
    ]
    decisions = [
        FeatureDecision(
            candidate_id="candidate-0",
            state=FeatureDecisionState.ACCEPTED,
            reason_code="accepted",
            reason="fixture",
        )
    ]
    accepted_features = pd.DataFrame({"row_id": row_ids, "double_a": [0, 2, 4, 6, 8, 10]})
    counts = GoldFeatureCounts(
        candidates_proposed=1,
        candidates_executed=1,
        candidates_accepted=1,
        accepted_output_columns=1,
    )
    gold_manifest = RunManifest(
        layer=Layer.GOLD,
        package_kind="gold",
        layer_fingerprint="gold-fingerprint",
        run_id="gold-run",
        case_fingerprint="case",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="silver", method="fixture", model="none", seed=42),
        environment=_environment(),
        upstream_manifests=[silver_ref],
        created_at=now,
        completed_at=now,
    )
    gold = GoldPackage(
        manifest=gold_manifest,
        request=gold_request,
        candidates=[candidate],
        provenance=provenance,
        decisions=decisions,
        accepted_features=accepted_features,
        checks=[],
        code={},
        dependencies={},
        counts=counts,
    )
    return silver, gold, gold_ref


def _materialized_package(
    tmp_path: pathlib.Path,
) -> tuple[LocalArtifactStore, Any, dict[str, Any], PlatinumRequest]:
    """Run the full Platinum node chain and persist the package under ``tmp_path``."""
    silver, gold, gold_ref = _packages()
    request = build_platinum_request(
        run_id="platinum-run",
        case_fingerprint="case",
        silver=silver,
        gold=gold,
        gold_ref=gold_ref,
        model_name="random_forest",
        metric="acc",
        seed=42,
    )
    request = platinum_request(request)
    baseline = baseline_fold_evidence(request, silver)
    candidates = candidate_fold_evidence(request, silver, gold)
    aggregate = aggregate_metrics(request, baseline, candidates)
    uncertainty = uncertainty_summary(request, aggregate)
    decisions = selection_decisions(request, gold, aggregate, uncertainty)
    checks = platinum_checks(request, silver, baseline, aggregate)
    manifest = platinum_manifest(request, _environment(), checks, aggregate, uncertainty, decisions)
    store = LocalArtifactStore(tmp_path / "lake")
    materialization = platinum_materialization(
        manifest,
        request,
        silver,
        baseline,
        candidates,
        aggregate,
        uncertainty,
        decisions,
        checks,
        ExecutionProfile.DEVELOPMENT,
        store,
    )
    return store, materialization, aggregate, request


def _platinum_fingerprint(request: PlatinumRequest, *, identity_schema_version: str) -> str:
    """Recompute the request's input fingerprint under a chosen identity version."""
    return platinum_input_fingerprint(
        gold_fingerprint_value=request.gold_fingerprint,
        model_name=request.model.name,
        model_version=request.model.distribution_version,
        model_config=request.model.model_dump(mode="json"),
        metric=request.metric,
        fold_fingerprint=request.fold_fingerprint,
        evaluation_policy={
            **request.evaluation_policy.model_dump(mode="json"),
            "metric_direction": request.metric_direction.value,
            "selection_policy": request.selection_policy.model_dump(mode="json"),
        },
        uncertainty_policy=request.uncertainty_policy.model_dump(mode="json"),
        identity_schema_version=identity_schema_version,
    )


# ── 1. Versioned identity schema ─────────────────────────────


def test_platinum_identity_bump_is_deterministic_and_layer_scoped() -> None:
    common: dict[str, Any] = {
        "gold_fingerprint_value": "gold-fingerprint",
        "model_name": "random_forest",
        "model_version": "1",
        "model_config": {},
        "metric": "acc",
        "fold_fingerprint": "folds",
        "evaluation_policy": {},
        "uncertainty_policy": {},
    }
    v2 = platinum_input_fingerprint(**common)
    v2_again = platinum_input_fingerprint(**common)
    v1 = platinum_input_fingerprint(**common, identity_schema_version="1")
    v1_again = platinum_input_fingerprint(**common, identity_schema_version="1")

    assert PLATINUM_IDENTITY_SCHEMA_VERSION == "2"
    assert v2 == v2_again
    assert v1 == v1_again
    # The version enters the fingerprint: v1 packages can never satisfy v2 reuse.
    assert v2 != v1

    silver_kwargs: dict[str, Any] = {
        "bronze_fingerprint_value": "bronze-fingerprint",
        "target_name": "target",
        "task": "classification",
        "canonicalization_config": {},
        "split_policy": {"folds": 3},
        "split_seed": 42,
    }
    silver = silver_fingerprint(**silver_kwargs)
    assert silver == silver_fingerprint(**silver_kwargs)
    assert silver != v2
    assert silver != v1

    gold_kwargs: dict[str, Any] = {
        "silver_fingerprint_value": "silver-fingerprint",
        "method_name": "fixture",
        "method_version": "1",
        "method_config": {},
        "prompt_bundle_fingerprint": "prompt",
        "generated_contract_version": "1",
        "selection_policy": {},
    }
    gold = gold_input_fingerprint(**gold_kwargs)
    assert gold == gold_input_fingerprint(**gold_kwargs)
    assert gold != v2
    assert gold != v1


# ── 2. Reuse rejection + readability at the store level ──────


def test_v2_package_is_reusable_while_v1_identity_is_rejected_but_readable(
    tmp_path: pathlib.Path,
) -> None:
    store, materialization, _aggregate, request = _materialized_package(tmp_path)
    ref = materialization.manifest_ref
    assert ref is not None

    manifest = store.load_manifest(ref)
    assert manifest.reuse_fingerprint == request.platinum_input_fingerprint
    assert manifest.request.options["platinum_identity_schema_version"] == "2"
    assert manifest.request.options["platinum_identity_schema_version"] == (
        PLATINUM_IDENTITY_SCHEMA_VERSION
    )

    upstreams = [request.silver_manifest, request.gold_manifest]
    found = store.find_reusable(
        request.platinum_input_fingerprint, Layer.PLATINUM, upstreams=upstreams
    )
    assert found == ref

    v1_fingerprint = _platinum_fingerprint(request, identity_schema_version="1")
    assert v1_fingerprint != request.platinum_input_fingerprint
    assert store.find_reusable(v1_fingerprint, Layer.PLATINUM, upstreams=upstreams) is None

    # The committed v2 package stays integrity-bound and readable.
    package = load_platinum_package(store, ref)
    assert package.manifest.reuse_fingerprint == request.platinum_input_fingerprint


# ── 3. Result fields round-trip ──────────────────────────────


def test_result_directional_fields_round_trip_and_cancelled_rows_stay_none() -> None:
    result = ExperimentResult(
        dataset="titanic",
        method="fixture",
        model="random_forest",
        seed=42,
        directional_gain=0.05,
        gain_lower_bound=0.01,
        gain_upper_bound=0.09,
        evaluation_protocol="holdout",
    )
    data = ExperimentalPlatform._result_to_dict(result)
    assert data["directional_gain"] == 0.05
    assert data["gain_lower_bound"] == 0.01
    assert data["gain_upper_bound"] == 0.09
    assert data["evaluation_protocol"] == "holdout"

    cancelled = ExperimentalPlatform._result_to_dict(
        ExperimentResult(
            dataset="titanic",
            method="fixture",
            model="random_forest",
            seed=42,
            state=RunState.CANCELLED,
        )
    )
    assert cancelled["state"] == "cancelled"
    assert cancelled["directional_gain"] is None
    assert cancelled["gain_lower_bound"] is None
    assert cancelled["gain_upper_bound"] is None
    assert cancelled["evaluation_protocol"] is None


# ── 4. Tracker rendering ─────────────────────────────────────


class _RecordingTracker(ExperimentTracker):
    """Minimal in-memory tracker capturing init_run config and metrics."""

    def __init__(self) -> None:
        super().__init__(project="unit")
        self.configs: list[dict[str, Any]] = []
        self.metrics: list[dict[str, float]] = []
        self.finished = 0

    def init_run(self, run_name: str, config: dict[str, Any]) -> None:
        self.configs.append(dict(config))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.append(dict(metrics))

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_artifact(self, path: str, artifact_type: str = "dataset") -> None:
        pass

    def finish(self) -> None:
        self.finished += 1

    def _log_dataframe(self, key: str, df: pd.DataFrame) -> None:
        pass

    def _log_code(self, key: str, code: str) -> None:
        pass


def _tracker_settings(tmp_path: pathlib.Path) -> Settings:
    return Settings(
        metric="acc",
        dataflow={
            "engine": "hamilton",
            "artifact_root": str(tmp_path / "artifacts"),
            "cache": {"path": str(tmp_path / "hamilton-cache")},
        },
    )


def _result_with_directional_fields(**overrides: Any) -> ExperimentResult:
    values: dict[str, Any] = {
        "dataset": "titanic",
        "method": "fixture",
        "model": "random_forest",
        "seed": 42,
        "cv_score": 0.82,
        "gain": 0.05,
        "baseline_score": 0.77,
        "directional_gain": 0.05,
        "gain_lower_bound": 0.01,
        "gain_upper_bound": 0.09,
        "evaluation_protocol": "holdout",
        "run_id": "attempt-1",
    }
    values.update(overrides)
    return ExperimentResult(**values)


def _render(
    tmp_path: pathlib.Path, result: ExperimentResult
) -> tuple[_RecordingTracker, dict[str, Any], dict[str, float]]:
    tracker = _RecordingTracker()
    case = ExperimentCase(dataset="titanic", method="fixture", model="random_forest", seed=42)
    ExperimentalPlatform()._apply_tracker_effects(
        tracker, _tracker_settings(tmp_path), case, result
    )
    assert tracker.finished == 1
    assert len(tracker.configs) == 1
    assert len(tracker.metrics) == 1
    return tracker, tracker.configs[0], tracker.metrics[0]


def test_tracker_renders_protocol_config_and_directional_metrics(tmp_path: pathlib.Path) -> None:
    _tracker, config, metrics = _render(tmp_path, _result_with_directional_fields())

    assert config["evaluation_protocol"] == "holdout"
    assert config["selection_biased"] is False
    assert metrics["directional_gain"] == 0.05
    assert metrics["gain_lower_bound"] == 0.01
    assert metrics["gain_upper_bound"] == 0.09
    assert metrics["gain"] == 0.05


def test_tracker_flags_compatibility_results_selection_biased(tmp_path: pathlib.Path) -> None:
    # A 0.0 lower bound is a real value, not an absence: it must be logged as
    # 0.0, never dropped or coerced.
    _tracker, config, metrics = _render(
        tmp_path,
        _result_with_directional_fields(
            evaluation_protocol="compatibility",
            gain_lower_bound=0.0,
        ),
    )

    assert config["evaluation_protocol"] == "compatibility"
    assert config["selection_biased"] is True
    assert metrics["gain_lower_bound"] == 0.0


def test_tracker_legacy_results_keep_stable_key_sets(tmp_path: pathlib.Path) -> None:
    _tracker, config, metrics = _render(
        tmp_path,
        _result_with_directional_fields(
            directional_gain=None,
            gain_lower_bound=None,
            gain_upper_bound=None,
            evaluation_protocol=None,
        ),
    )

    assert "evaluation_protocol" not in config
    assert "selection_biased" not in config
    assert "directional_gain" not in metrics
    assert "gain_lower_bound" not in metrics
    assert "gain_upper_bound" not in metrics
    assert set(metrics) == {"cv_score", "gain", "baseline_score"}


# ── 5. Executor population ───────────────────────────────────

DATASET_NAME = "reuse-fields-xor"
METHOD_NAME = "reuse_fields_deterministic"


class _OfflineDeterministicMethod(BaseMethod):
    """Offline deterministic method emitting one sandbox-safe numeric feature."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name=METHOD_NAME)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> _OfflineDeterministicMethod:
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


def _local_numeric_dataset(tmp_path: pathlib.Path) -> dict[str, Any]:
    """Deterministic XOR-style dataset tiled to 24 rows (holdout-protocol safe)."""
    block = pd.DataFrame(
        {
            "a": [0.0, 0.0, 1.0, 1.0, 0.1, 0.2, 0.9, 0.8],
            "b": [0.0, 1.0, 0.0, 1.0, 0.9, 0.8, 0.1, 0.2],
            "c": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "target": [0, 1, 1, 0, 1, 1, 1, 1],
        }
    )
    frame = pd.concat([block] * 3, ignore_index=True)
    sample_dir = tmp_path / "reuse-fields-xor-data"
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(sample_dir / "train.csv", index=False)
    (sample_dir / "metadata.json").write_text('{"target": "target", "task": "classification"}')
    return {
        "source": "local",
        "path": str(sample_dir),
        "target": "target",
        "task": "classification",
    }


def test_executor_populates_directional_result_fields(tmp_path: pathlib.Path) -> None:
    dataset_info = _local_numeric_dataset(tmp_path)
    settings = Settings(
        metric="acc",
        dataflow={
            "engine": "hamilton",
            "artifact_root": str(tmp_path / "artifacts"),
            "cache": {"path": str(tmp_path / "hamilton-cache")},
        },
    )
    registry = DatasetRegistry()
    registry.register(DATASET_NAME, dataset_info)
    executor = HamiltonLayerExecutor(
        settings=settings,
        artifact_store=LocalArtifactStore(settings.dataflow.artifact_root),
        dataset_registry=registry,
        method_classes={
            **MethodRegistry.get_all_methods(),
            METHOD_NAME: _OfflineDeterministicMethod,
        },
    )
    result = executor.execute_case(
        ExperimentCase(
            dataset=DATASET_NAME,
            method=METHOD_NAME,
            model="random_forest",
            seed=42,
            cv_folds=2,
            run_id="reuse-fields-attempt-1",
            attempt_id="reuse-fields-attempt-1",
        )
    )

    assert result.resolved_state is RunState.SUCCEEDED
    assert result.directional_gain is not None
    assert result.gain_lower_bound is not None
    assert result.gain_upper_bound is not None
    assert result.evaluation_protocol == "holdout"

    store = LocalArtifactStore(settings.dataflow.artifact_root)
    ref = store.get_manifest_ref(
        ArtifactNamespace(layer=Layer.PLATINUM, run_id=result.run_id or "")
    )
    assert ref is not None
    package = load_platinum_package(store, ref)
    assert result.gain == package.aggregate.legacy_gain
    assert result.directional_gain == package.aggregate.directional_gain
    assert result.gain_lower_bound == package.uncertainty.lower_bound
    assert result.gain_upper_bound == package.uncertainty.upper_bound
