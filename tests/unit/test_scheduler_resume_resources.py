"""Focused PR5 scheduler, resource, resume, lifecycle, and recovery tests."""

from __future__ import annotations

import os
import time
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from pickle import PicklingError
from types import SimpleNamespace

import pytest

from feature_forge.contracts import (
    ArtifactNamespace,
    EnvironmentSnapshot,
    Layer,
    ManifestRef,
    RunManifest,
    RunRequest,
    RunState,
)
from feature_forge.contracts.orchestration import (
    EffectiveResourcePlan,
    ResourceConfig,
    ResumePlan,
    ResumePolicy,
    ResumeStageDecision,
    StageDisposition,
)
from feature_forge.exceptions import DatasetError, EvaluationError
from feature_forge.experiment.case_executor import CaseComputation, CaseComputationInput, run_case
from feature_forge.experiment.execution import (
    ExperimentCase,
    ExperimentResult,
    ProcessPoolExecutionAdapter,
    classify_failure,
)
from feature_forge.experiment.lifecycle import LocalRunRepository
from feature_forge.experiment.resources import resolve_resource_plan, safer_resource_plan
from feature_forge.experiment.resume import (
    ResumeExecutionError,
    _load_semantic_package,
    build_resume_plan,
    execute_resume_plan,
)
from feature_forge.experiment.tracker import NoOpTracker
from feature_forge.llm.retry import is_transient_llm_error
from feature_forge.platform import ExperimentalPlatform
from feature_forge.storage import atomic as atomic_storage
from feature_forge.storage.atomic import atomic_write_json
from feature_forge.storage.local import LocalArtifactStore


def _pool_worker(case: ExperimentCase) -> ExperimentResult:
    if case.dataset == "broken":
        raise ValueError("deterministic boom")
    return ExperimentResult(
        dataset=case.dataset,
        method=case.method,
        model=case.model,
        seed=case.seed,
        state=RunState.SUCCEEDED.value,
    )


def _portable_layer_executor(
    case: ExperimentCase, layer: Layer, upstream: dict[Layer, ManifestRef]
) -> ManifestRef:
    del upstream
    return ManifestRef(layer=layer, run_id=case.run_id or "portable", sha256="f" * 64)


def _layer_payload_worker(payload: CaseComputationInput) -> ExperimentResult:
    if payload.layer_executor is None:
        raise ValueError("missing layer executor")
    ref = payload.layer_executor(payload.case, Layer.BRONZE, {})
    return ExperimentResult(
        payload.case.dataset,
        payload.case.method,
        payload.case.model,
        payload.case.seed,
        state="succeeded" if ref.layer is Layer.BRONZE else "failed",
    )


def _case(name: str) -> ExperimentCase:
    return ExperimentCase(dataset=name, method="dummy", model="rf", seed=42, run_id=name)


class _TrackerSpy(NoOpTracker):
    def __init__(self) -> None:
        super().__init__(project="test")
        self.inits = 0
        self.logs = 0
        self.finishes = 0

    def init_run(self, run_name: str, config: dict[str, object]) -> None:
        self.inits += 1

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logs += 1

    def finish(self) -> None:
        self.finishes += 1


class _UnpickleableCase:
    dataset = "unpickleable"
    method = "dummy"
    model = "rf"
    seed = 42
    run_id = "unpickleable"

    def __reduce__(self) -> object:
        raise PicklingError("deliberately unpickleable")


def _manifest(
    run_id: str,
    layer: Layer,
    fingerprint: str,
    upstream: list[object],
) -> RunManifest:
    return RunManifest(
        layer=layer,
        package_kind=layer.value,
        layer_fingerprint=fingerprint,
        run_id=run_id,
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="demo", method="dummy", model="rf", seed=42),
        environment=EnvironmentSnapshot(
            python_version="3.13",
            operating_system="darwin",
            architecture="arm64",
            feature_forge_version="test",
        ),
        upstream_manifests=upstream,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


def _commit_chain(store: LocalArtifactStore, run_id: str) -> dict[Layer, str]:
    fingerprints = {layer: f"{layer.value}-fingerprint" for layer in Layer}
    upstream = []
    for layer in Layer:
        staging = store.begin(ArtifactNamespace(layer=layer, run_id=run_id))
        staging.write_text(f"{layer.value}.txt", layer.value)
        staging.set_manifest(_manifest(run_id, layer, fingerprints[layer], upstream))
        upstream = [store.commit(staging)]
    return fingerprints


def test_experiment_result_keeps_legacy_field_prefix() -> None:
    assert [item.name for item in fields(ExperimentResult)[:9]] == [
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


def test_process_scheduler_isolates_failure_and_preserves_order() -> None:
    cases = [_case("first"), _case("broken"), _case("last")]
    results = ProcessPoolExecutionAdapter(max_workers=2).run(cases, _pool_worker, progress=False)
    assert [item.dataset for item in results] == ["first", "broken", "last"]
    assert results[0].state == RunState.SUCCEEDED.value
    assert results[1].failure is not None
    assert results[1].failure["failure_class"] == "deterministic"
    assert results[2].state == RunState.SUCCEEDED.value


def test_fail_fast_cancels_never_started_cases_predictably() -> None:
    cases = [_case("broken"), _case("second"), _case("third")]
    started: list[str] = []
    results = ProcessPoolExecutionAdapter(max_workers=1).run(
        cases,
        _pool_worker,
        progress=False,
        fail_fast=True,
        on_start=lambda case: started.append(case.dataset),
    )
    assert [item.state for item in results] == ["failed", "cancelled", "cancelled"]
    assert started == ["broken"]


def test_unportable_worker_is_rejected_as_terminal_serialization_failure() -> None:
    results = ProcessPoolExecutionAdapter(max_workers=1).run(
        [_case("one"), _case("two")],
        lambda case: ExperimentResult(case.dataset, case.method, case.model, case.seed),
        progress=False,
    )
    assert [item.state for item in results] == ["failed", "failed"]
    assert all(item.failure and item.failure["stage"] == "serialization" for item in results)


def test_worker_serialization_failure_honors_fail_fast() -> None:
    results = ProcessPoolExecutionAdapter(max_workers=1).run(
        [_case("one"), _case("two")],
        lambda case: ExperimentResult(case.dataset, case.method, case.model, case.seed),
        progress=False,
        fail_fast=True,
    )
    assert [item.state for item in results] == ["failed", "cancelled"]


def test_case_pickling_failure_stops_submission_under_fail_fast() -> None:
    results = ProcessPoolExecutionAdapter(max_workers=1).run(
        [_UnpickleableCase(), _case("never-started")],
        _pool_worker,
        progress=False,
        fail_fast=True,
    )
    assert [item.state for item in results] == ["failed", "cancelled"]
    assert results[0].failure is not None
    assert results[0].failure["error_type"] == "PicklingError"


def test_immediate_submit_failure_stops_and_cancels_under_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectingExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def submit(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("executor is shutting down")

        def shutdown(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr("feature_forge.experiment.execution.ProcessPoolExecutor", RejectingExecutor)
    results = ProcessPoolExecutionAdapter(max_workers=1).run(
        [_case("rejected"), _case("never-started")],
        _pool_worker,
        progress=False,
        fail_fast=True,
    )
    assert [item.state for item in results] == ["failed", "cancelled"]
    assert results[0].failure is not None
    assert results[0].failure["stage"] == "scheduler.submit"


def test_worker_local_tracker_initializes_logs_and_finishes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _TrackerSpy()
    monkeypatch.setattr(
        "feature_forge.experiment.case_executor.create_tracker_from_config",
        lambda config: tracker,
    )
    monkeypatch.setattr(
        CaseComputation,
        "compute",
        lambda self, payload: ExperimentResult(
            payload.case.dataset,
            payload.case.method,
            payload.case.model,
            payload.case.seed,
            cv_score=0.8,
            gain=0.1,
            baseline_score=0.7,
            state="succeeded",
        ),
    )
    plan = resolve_resource_plan(ResourceConfig())
    result = run_case(
        CaseComputationInput(
            case=_case("tracked"),
            settings_data={},
            tracker_config={"backend": "none", "project": "test"},
            resource_plan=plan.model_dump(mode="json"),
        )
    )
    assert result.error is None
    assert (tracker.inits, tracker.logs, tracker.finishes) == (1, 1, 1)


def test_resource_plan_has_one_heavy_layer_and_retry_is_safer() -> None:
    plan = resolve_resource_plan(
        ResourceConfig(
            experiment_workers=4,
            llm_concurrency=5,
            sandbox_workers=3,
            candidate_batch_size=8,
            cv_workers=6,
        )
    )
    assert plan.heavy_parallel_layer == "outer"
    assert (plan.llm_concurrency, plan.sandbox_workers, plan.cv_workers) == (1, 1, 1)
    safer = safer_resource_plan(plan)
    assert safer is not None
    assert safer.experiment_workers < plan.experiment_workers


def test_outer_and_cv_parallelism_force_single_model_and_blas_threads() -> None:
    outer = resolve_resource_plan(
        ResourceConfig(experiment_workers=3, model_threads=4, blas_threads=5)
    )
    cv = resolve_resource_plan(
        ResourceConfig(
            llm_concurrency=1,
            cv_workers=3,
            model_threads=4,
            blas_threads=5,
        )
    )
    assert (outer.model_threads, outer.blas_threads) == (1, 1)
    assert (cv.model_threads, cv.blas_threads) == (1, 1)
    with pytest.raises(ValueError, match="multiplies nested limits"):
        EffectiveResourcePlan(
            experiment_workers=1,
            llm_concurrency=1,
            sandbox_workers=2,
            candidate_batch_size=2,
            cv_workers=1,
            cv_backend="threading",
            model_threads=1,
            blas_threads=1,
            max_attempts=1,
            backoff_base_seconds=1,
            backoff_max_seconds=2,
            heavy_parallel_layer="none",
        )


def test_generic_os_and_connection_errors_are_not_implicitly_transient() -> None:
    from feature_forge.exceptions import LLMError

    assert classify_failure(OSError("local disk failure")).value == "deterministic"
    wrapped = LLMError("provider failed")
    wrapped.__cause__ = ConnectionError("generic connection")
    assert not is_transient_llm_error(wrapped)


def test_retry_synchronizes_terminal_attempt_and_safer_resource_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resource_failure(self: CaseComputation, payload: CaseComputationInput) -> ExperimentResult:
        return ExperimentResult(
            payload.case.dataset,
            payload.case.method,
            payload.case.model,
            payload.case.seed,
            error="resource exhausted",
            state="failed",
            failure={
                "schema_version": "1",
                "failure_class": "resource",
                "error_type": "MemoryError",
                "message": "resource exhausted",
                "stage": "case",
                "retryable": False,
                "attempt": 1,
                "cause_types": [],
            },
        )

    monkeypatch.setattr(CaseComputation, "compute", resource_failure)
    first = resolve_resource_plan(
        ResourceConfig(experiment_workers=2, max_attempts=2, backoff_base_seconds=0.001)
    )
    result = run_case(
        CaseComputationInput(
            case=_case("retry"),
            settings_data={},
            tracker_config={"backend": "none"},
            resource_plan=first.model_dump(mode="json"),
        )
    )
    assert result.attempt == 2
    assert result.failure is not None
    assert result.failure["attempt"] == 2
    assert result.resource_plan is not None
    assert result.resource_plan["experiment_workers"] == 1


def test_lifecycle_journal_is_append_only_and_has_current_snapshot(tmp_path: Path) -> None:
    repository = LocalRunRepository(tmp_path / "control")
    repository.record(
        run_id="run-1", case_id="case-1", event_type="case_planned", state=RunState.PLANNED
    )
    repository.record(
        run_id="run-1", case_id="case-1", event_type="case_terminal", state=RunState.SUCCEEDED
    )
    assert [item.event_type for item in repository.load_events("run-1")] == [
        "case_planned",
        "case_terminal",
    ]
    assert (tmp_path / "control/runs/run-1/state.json").is_file()


def test_platform_emits_typed_stage_and_terminal_lifecycle_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver_ref = ManifestRef(layer=Layer.SILVER, run_id="typed-run", sha256="e" * 64)

    def completed_case(self: CaseComputation, payload: CaseComputationInput) -> ExperimentResult:
        return ExperimentResult(
            payload.case.dataset,
            payload.case.method,
            payload.case.model,
            payload.case.seed,
            run_id=payload.case.run_id,
            state="succeeded",
            silver_fingerprint="silver-fingerprint",
            attempt=2,
            resource_plan=payload.resource_plan,
            execution_plan={
                "resume": {
                    "schema_version": "1",
                    "run_id": payload.case.run_id,
                    "dry_run": False,
                    "decisions": [
                        {
                            "schema_version": "1",
                            "layer": "silver",
                            "expected_fingerprint": "silver-fingerprint",
                            "disposition": "reused",
                            "manifest_ref": silver_ref.model_dump(mode="json"),
                            "reason": "verified",
                        }
                    ],
                    "next_layer": None,
                    "complete": True,
                }
            },
        )

    monkeypatch.setattr(CaseComputation, "compute", completed_case)
    control = tmp_path / "control"
    ExperimentalPlatform().run(
        datasets=["titanic"],
        methods=["openfe"],
        lifecycle_dir=control,
        progress=False,
    )
    events = LocalRunRepository(control).load_events("run_titanic_openfe_xgboost_42")
    stage = next(event for event in events if event.event_type == "stage_terminal")
    terminal = events[-1]
    assert (stage.layer, stage.disposition, stage.fingerprint, stage.manifest_ref) == (
        Layer.SILVER,
        "reused",
        "silver-fingerprint",
        silver_ref,
    )
    assert terminal.attempt == 2
    assert terminal.fingerprint == "silver-fingerprint"
    assert terminal.details["resource_plan"] is not None


def test_platform_emits_partial_stage_events_when_layer_execution_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run_titanic_openfe_xgboost_42"
    bronze_ref = ManifestRef(layer=Layer.BRONZE, run_id="source", sha256="1" * 64)
    silver_ref = ManifestRef(layer=Layer.SILVER, run_id=run_id, sha256="2" * 64)
    partial = ResumePlan(
        run_id=run_id,
        decisions=[
            ResumeStageDecision(
                layer=Layer.BRONZE,
                expected_fingerprint="bronze-fingerprint",
                disposition=StageDisposition.REUSED,
                state=RunState.SUCCEEDED,
                manifest_ref=bronze_ref,
                reason="verified",
            ),
            ResumeStageDecision(
                layer=Layer.SILVER,
                expected_fingerprint="silver-fingerprint",
                disposition=StageDisposition.EXECUTED,
                state=RunState.SUCCEEDED,
                manifest_ref=silver_ref,
                reason="executed",
            ),
            ResumeStageDecision(
                layer=Layer.GOLD,
                expected_fingerprint="gold-fingerprint",
                disposition=StageDisposition.EXECUTED,
                state=RunState.FAILED,
                reason="execution failed: DatasetError: gold failed",
            ),
        ],
        next_layer=Layer.GOLD,
        complete=False,
    )

    def fail_layered(
        payload: CaseComputationInput,
        resource_plan: EffectiveResourcePlan | None,
    ) -> ExperimentResult:
        del payload, resource_plan
        cause = DatasetError("gold failed")
        raise ResumeExecutionError(partial, Layer.GOLD, cause)

    monkeypatch.setattr(CaseComputation, "_compute_layered", staticmethod(fail_layered))
    control = tmp_path / "control"
    result = ExperimentalPlatform().run(
        datasets=["titanic"],
        methods=["openfe"],
        artifact_policy="layer_boundaries",
        artifact_root=tmp_path / "lake",
        lifecycle_dir=control,
        layer_fingerprints={run_id: {layer.value: f"{layer.value}-fingerprint" for layer in Layer}},
        layer_executor=_portable_layer_executor,
        progress=False,
    )[0]
    assert result["state"] == "failed"
    assert result["execution_plan"]["resume"]["next_layer"] == "gold"
    events = LocalRunRepository(control).load_events(run_id)
    stages = [event for event in events if event.event_type == "stage_terminal"]
    assert [event.state for event in stages] == [
        RunState.SUCCEEDED,
        RunState.SUCCEEDED,
        RunState.FAILED,
    ]
    assert stages[-1].failure is not None
    assert stages[-1].failure.stage == "resume.gold"
    assert events[-1].event_type == "case_terminal"
    assert events[-1].state is RunState.FAILED


def test_platform_dry_run_has_no_tracker_or_persistent_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_tracker(*args: object, **kwargs: object) -> NoOpTracker:
        raise AssertionError("tracker must not be constructed during dry-run")

    monkeypatch.setattr("feature_forge.platform.create_tracker_from_config", forbidden_tracker)
    output = ExperimentalPlatform().run(
        datasets=["titanic"],
        methods=["openfe"],
        dry_run=True,
        lifecycle_dir=tmp_path / "control",
        progress=False,
    )
    assert output[0]["state"] == "planned"
    assert not (tmp_path / "control").exists()


def test_dry_run_validates_metric_against_dataset_task(tmp_path: Path) -> None:
    platform = ExperimentalPlatform()
    platform.register_dataset(
        "regression-local",
        {"source": "local", "task": "regression", "target": "target"},
    )
    with pytest.raises(EvaluationError, match="incompatible with regression"):
        platform.run(
            datasets=["regression-local"],
            methods=["openfe"],
            metric="auc",
            dry_run=True,
            progress=False,
        )


def test_nonlegacy_dry_run_builds_real_read_only_resume_plan(tmp_path: Path) -> None:
    run_id = "run_titanic_openfe_xgboost_42"
    root = tmp_path / "lake"
    result = ExperimentalPlatform().run(
        datasets=["titanic"],
        methods=["openfe"],
        artifact_policy="layer_boundaries",
        artifact_root=root,
        layer_fingerprints={run_id: {layer.value: f"{layer.value}-fingerprint" for layer in Layer}},
        dry_run=True,
        progress=False,
    )[0]
    resume = result["execution_plan"]["resume"]
    assert [decision["disposition"] for decision in resume["decisions"]] == ["executed"] * 4
    assert not root.exists()


def test_nonlegacy_execution_fails_closed_without_layer_executor(tmp_path: Path) -> None:
    run_id = "run_titanic_openfe_xgboost_42"
    with pytest.raises(ValueError, match="layer_executor"):
        ExperimentalPlatform().run(
            datasets=["titanic"],
            methods=["openfe"],
            artifact_policy="layer_boundaries",
            artifact_root=tmp_path / "lake",
            layer_fingerprints={
                run_id: {layer.value: f"{layer.value}-fingerprint" for layer in Layer}
            },
            progress=False,
        )


def test_nonlegacy_plan_rejects_incomplete_layer_fingerprints(tmp_path: Path) -> None:
    run_id = "run_titanic_openfe_xgboost_42"
    with pytest.raises(ValueError, match="requires exactly"):
        ExperimentalPlatform().run(
            datasets=["titanic"],
            methods=["openfe"],
            artifact_policy="layer_boundaries",
            artifact_root=tmp_path / "lake",
            layer_fingerprints={run_id: {"bronze": "bronze-fingerprint"}},
            dry_run=True,
            progress=False,
        )


def test_nonlegacy_platform_dispatches_to_layered_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []

    def layered(
        payload: CaseComputationInput,
        resource_plan: EffectiveResourcePlan | None,
    ) -> ExperimentResult:
        called.append(payload.case.artifact_policy)
        return ExperimentResult(
            payload.case.dataset,
            payload.case.method,
            payload.case.model,
            payload.case.seed,
            run_id=payload.case.run_id,
            state="succeeded",
        )

    monkeypatch.setattr(CaseComputation, "_compute_layered", staticmethod(layered))
    run_id = "run_titanic_openfe_xgboost_42"
    result = ExperimentalPlatform().run(
        datasets=["titanic"],
        methods=["openfe"],
        artifact_policy="layer_boundaries",
        artifact_root=tmp_path / "lake",
        layer_fingerprints={run_id: {layer.value: f"{layer.value}-fingerprint" for layer in Layer}},
        layer_executor=lambda case, layer, upstream: next(iter(upstream.values())),
        progress=False,
    )[0]
    assert called == ["layer_boundaries"]
    assert result["state"] == "succeeded"


def test_layer_executor_payload_is_portable_under_spawn() -> None:
    payload = CaseComputationInput(
        case=_case("portable"),
        settings_data={},
        layer_executor=_portable_layer_executor,
    )
    result = ProcessPoolExecutionAdapter(max_workers=1).run(
        [payload], _layer_payload_worker, progress=False
    )[0]
    assert result.state == "succeeded"


def test_resume_reuses_only_verified_contiguous_chain(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    fingerprints = _commit_chain(store, "run-1")
    plan = build_resume_plan(
        store=store,
        run_id="run-1",
        expected_fingerprints=fingerprints,
        policy=ResumePolicy(enabled=True),
        semantic_validator=lambda store, ref, chain: None,
    )
    assert plan.complete
    assert [item.disposition.value for item in plan.decisions] == ["reused"] * 4


def test_resume_fails_closed_for_corrupt_preferred_package(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    fingerprints = _commit_chain(store, "run-1")
    (tmp_path / "lake/02_silver/runs/run-1/silver.txt").write_text("corrupt")
    with pytest.raises(ValueError, match="preferred silver package is invalid"):
        build_resume_plan(
            store=store,
            run_id="run-1",
            expected_fingerprints=fingerprints,
            policy=ResumePolicy(enabled=True),
            semantic_validator=lambda store, ref, chain: None,
        )


@pytest.mark.parametrize("error_type", [ValueError, DatasetError])
def test_semantic_loader_failure_recomputes_instead_of_reusing(
    tmp_path: Path,
    error_type: type[Exception],
) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    fingerprints = _commit_chain(store, "run-1")

    def reject_semantics(
        store: LocalArtifactStore,
        ref: ManifestRef,
        chain: object,
    ) -> None:
        raise error_type("semantic mismatch")

    plan = build_resume_plan(
        store=store,
        run_id="run-1",
        expected_fingerprints={Layer.BRONZE: fingerprints[Layer.BRONZE]},
        policy=ResumePolicy(enabled=True, recompute_invalid=True),
        semantic_validator=reject_semantics,
    )
    decision = plan.decisions[0]
    assert decision.disposition is StageDisposition.EXECUTED
    assert decision.manifest_ref is None
    assert "will be recomputed" in decision.reason


def test_mid_suffix_failure_preserves_partial_resume_plan(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    fingerprints = _commit_chain(store, "source")
    plan = build_resume_plan(
        store=store,
        run_id="new-run",
        expected_fingerprints={
            Layer.BRONZE: fingerprints[Layer.BRONZE],
            Layer.SILVER: "new-silver",
            Layer.GOLD: "new-gold",
        },
        policy=ResumePolicy(enabled=True),
        semantic_validator=lambda store, ref, chain: None,
    )

    def execute_stage(
        layer: Layer,
        chain: dict[Layer, ManifestRef],
    ) -> ManifestRef:
        if layer is Layer.GOLD:
            raise DatasetError("gold execution failed")
        staging = store.begin(ArtifactNamespace(layer=layer, run_id="new-run"))
        staging.write_text(f"{layer.value}.txt", layer.value)
        staging.set_manifest(
            _manifest(
                "new-run",
                layer,
                "new-silver",
                list(chain.values()),
            )
        )
        return store.commit(staging)

    with pytest.raises(ResumeExecutionError) as captured:
        execute_resume_plan(
            store=store,
            plan=plan,
            execute_stage=execute_stage,
            semantic_validator=lambda store, ref, chain: None,
        )
    partial = captured.value.plan
    assert [item.disposition for item in partial.decisions] == [
        StageDisposition.REUSED,
        StageDisposition.EXECUTED,
        StageDisposition.EXECUTED,
    ]
    assert [item.state for item in partial.decisions] == [
        RunState.SUCCEEDED,
        RunState.SUCCEEDED,
        RunState.FAILED,
    ]
    assert partial.next_layer is Layer.GOLD
    assert not partial.complete


def test_semantic_resume_loaders_require_exact_authoritative_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bronze = ManifestRef(layer=Layer.BRONZE, run_id="bronze", sha256="a" * 64)
    silver = ManifestRef(layer=Layer.SILVER, run_id="silver", sha256="b" * 64)
    gold = ManifestRef(layer=Layer.GOLD, run_id="gold", sha256="c" * 64)
    platinum = ManifestRef(layer=Layer.PLATINUM, run_id="platinum", sha256="d" * 64)
    loaded: list[str] = []

    class FakeStore:
        def load_manifest(self, ref: ManifestRef) -> object:
            loaded.append(ref.layer.value)
            return SimpleNamespace(upstream_manifests=[])

    monkeypatch.setattr(
        "feature_forge.dataflows.silver.load_silver_package",
        lambda store, ref: SimpleNamespace(manifest=SimpleNamespace(upstream_manifests=[bronze])),
    )
    monkeypatch.setattr(
        "feature_forge.dataflows.gold.load_gold_package",
        lambda store, ref: SimpleNamespace(
            request=SimpleNamespace(silver_manifest=silver),
            manifest=SimpleNamespace(upstream_manifests=[silver]),
        ),
    )
    monkeypatch.setattr(
        "feature_forge.dataflows.platinum.load_platinum_package",
        lambda store, ref: SimpleNamespace(
            request=SimpleNamespace(silver_manifest=silver, gold_manifest=gold),
            manifest=SimpleNamespace(upstream_manifests=[silver, gold]),
        ),
    )
    store = FakeStore()
    _load_semantic_package(store, bronze, {})  # type: ignore[arg-type]
    _load_semantic_package(store, silver, {Layer.BRONZE: bronze})  # type: ignore[arg-type]
    _load_semantic_package(store, gold, {Layer.SILVER: silver})  # type: ignore[arg-type]
    _load_semantic_package(  # type: ignore[arg-type]
        store,
        platinum,
        {Layer.SILVER: silver, Layer.GOLD: gold},
    )
    assert loaded == ["bronze"]
    with pytest.raises(ValueError, match="exact planned Silver/Gold chain"):
        _load_semantic_package(  # type: ignore[arg-type]
            store,
            platinum,
            {Layer.SILVER: silver, Layer.GOLD: bronze},
        )


def test_recovery_removes_dead_local_state_but_retains_live_and_committed(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    _commit_chain(store, "committed")
    runs = tmp_path / "lake/01_bronze/runs"
    abandoned = runs / ".abandoned.tmp-dead"
    abandoned.mkdir()
    atomic_write_json(
        abandoned / ".staging-owner.json",
        {"pid": 99_999_999, "hostname": __import__("socket").gethostname()},
    )
    live = runs / ".live.tmp-active"
    live.mkdir()
    atomic_write_json(
        live / ".staging-owner.json",
        {"pid": os.getpid(), "hostname": __import__("socket").gethostname()},
    )
    old = time.time() - 3600
    os.utime(abandoned, (old, old))
    os.utime(live, (old, old))

    report = store.recover_abandoned(
        retain_staging_seconds=1,
        stale_lock_seconds=1,
    )
    assert str(abandoned) in report["removed_staging"]
    assert live.is_dir()
    assert (runs / "committed/_SUCCESS").is_file()


def test_interruption_after_success_marker_leaves_recoverable_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="interrupted"))
    staging.write_text("gold.txt", "gold")
    staging.set_manifest(_manifest("interrupted", Layer.GOLD, "gold-fp", []))
    monkeypatch.setattr(
        "feature_forge.storage.local.atomic_publish_directory",
        lambda staging_path, destination: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        store.commit(staging)
    assert (staging.path / "manifest.json").is_file()
    assert (staging.path / "_SUCCESS").is_file()
    old = time.time() - 3600
    os.utime(staging.path, (old, old))
    report = store.recover_abandoned(retain_staging_seconds=1, stale_lock_seconds=1)
    assert str(staging.path) in report["removed_staging"]


def test_stale_post_rename_lock_is_removed_without_touching_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "01_bronze/runs/committed"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}")
    monkeypatch.setattr(atomic_storage, "_remove_owned_lock", lambda lock, token: None)
    atomic_storage.atomic_publish_directory(staging, destination)
    lock = destination.with_name(f".{destination.name}.commit-lock")
    owner = lock / "owner.json"
    payload = __import__("json").loads(owner.read_text())
    payload["pid"] = 99_999_999
    atomic_write_json(owner, payload)
    old = time.time() - 3600
    os.utime(lock, (old, old))

    store = LocalArtifactStore(tmp_path)
    report = store.recover_abandoned(retain_staging_seconds=1, stale_lock_seconds=1)
    assert str(lock) in report["removed_locks"]
    assert (destination / "manifest.json").is_file()
