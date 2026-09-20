"""Worker-local orchestration of verified Hamilton medallion stages."""

from __future__ import annotations

import importlib.resources
import inspect
import platform as runtime_platform
import sqlite3
import sys
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import pandas as pd

from feature_forge.config import Settings
from feature_forge.contracts import (
    CaseExecutionPlan,
    DatasetRequest,
    EnvironmentSnapshot,
    EvaluationPolicy,
    FailureClass,
    GoldRequest,
    Layer,
    MethodIdentity,
    ResumePolicy,
    SilverPackage,
    StageDisposition,
    StageExecution,
    gold_input_fingerprint,
)
from feature_forge.contracts.orchestration import FailureRecord
from feature_forge.contracts.stages import RunState
from feature_forge.data import DatasetRegistry
from feature_forge.dataflows._io import (
    build_platinum_request,
    load_gold_package,
    load_platinum_package,
    load_silver_package,
)
from feature_forge.dataflows.driver import (
    BRONZE_FINAL_VARS,
    GOLD_FINAL_VARS,
    PLATINUM_FINAL_VARS,
    SILVER_FINAL_VARS,
    build_bronze_driver,
    build_gold_driver,
    build_platinum_driver,
    build_silver_driver,
    resolve_cache_path,
)
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.dataflows.silver import dataset_fingerprint_value
from feature_forge.evaluation import CVEvaluator, ModelFactory
from feature_forge.evaluation.metrics import MetricRegistry
from feature_forge.evaluation.model_factory import ModelRegistry
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import (
    DatasetError,
    LLMError,
    SandboxTimeoutError,
    SandboxValidationError,
)
from feature_forge.experiment.execution import (
    CaseComputationInput,
    ExperimentCase,
    ExperimentResult,
    allocate_attempt_id,
)
from feature_forge.experiment.lifecycle import LocalRunRepository
from feature_forge.llm.factory import create_llm_client
from feature_forge.methods import BaseMethod, MethodRegistry
from feature_forge.observability.hamilton_adapter import (
    HamiltonTelemetryRecorder,
    create_run_event_sink,
)
from feature_forge.storage.hashing import fingerprint, normalize_secret_free, sha256_bytes
from feature_forge.storage.local import LocalArtifactStore


def _package_version() -> str:
    try:
        return version("feature-forge")
    except PackageNotFoundError:
        return "0+unknown"


def _environment(settings: Settings | None = None) -> EnvironmentSnapshot:
    evaluation = settings.evaluation if settings is not None else None
    # ``unknown`` (not a fabricated profile) when no settings are supplied:
    # provenance must never claim a containment level it did not observe.
    profile = evaluation.sandbox_profile.value if evaluation is not None else "unknown"
    return EnvironmentSnapshot(
        python_version=runtime_platform.python_version(),
        operating_system=runtime_platform.system().lower() or sys.platform,
        architecture=runtime_platform.machine() or "unknown",
        feature_forge_version=_package_version(),
        sandbox_profile=profile,
        sandbox_degraded=profile == "degraded_development",
    )


def _prompt_identity(method_class: type[BaseMethod]) -> dict[str, Any]:
    """Hash colocated versioned prompt files without importing a provider."""
    package = f"{method_class.__module__.rsplit('.', maxsplit=1)[0]}.prompts"
    try:
        root = importlib.resources.files(package)
        prompt_files = sorted(
            (item for item in root.iterdir() if item.name.endswith(".yaml")),
            key=lambda item: item.name,
        )
        return {item.name: {"sha256": sha256_bytes(item.read_bytes())} for item in prompt_files}
    except (ModuleNotFoundError, OSError, TypeError):
        return {}


def _method_identity(
    name: str,
    method_class: type[BaseMethod],
    settings: Settings,
    mode: str | None,
) -> MethodIdentity:
    """Describe method behavior without constructing a provider or method instance."""
    try:
        source = inspect.getsource(method_class).encode("utf-8")
    except (OSError, TypeError):
        source = f"{method_class.__module__}.{method_class.__qualname__}".encode()
    source_hash = sha256_bytes(source)
    return MethodIdentity(
        registry_name=name,
        qualified_class_name=f"{method_class.__module__}.{method_class.__qualname__}",
        distribution_name="feature-forge",
        distribution_version=_package_version(),
        mode=mode,
        configuration={
            "mode": mode,
            "n_rounds": settings.n_rounds,
            "max_selected_features": settings.max_selected_features,
            "source_hash": source_hash,
            "llm": normalize_secret_free(settings.llm.model_dump(mode="python")),
        },
        prompts=_prompt_identity(method_class),
        llm=normalize_secret_free(settings.llm.model_dump(mode="python")),
        feature_forge_version=_package_version(),
        source_hash=source_hash,
    )


def _case_key(case: ExperimentCase, settings: Settings, identity: MethodIdentity) -> str:
    return fingerprint(
        {
            "kind": "experiment-case",
            "dataset": case.dataset,
            "method": identity.model_dump(mode="json"),
            "model": case.model,
            "metric": settings.metric,
            "seed": case.seed,
            "cv_folds": case.cv_folds or settings.evaluation.cv_folds,
            "task_intent": settings.task,
            "evaluation": settings.evaluation.model_dump(mode="json"),
        }
    )


def _classify_failure(error: Exception) -> FailureClass:
    """Classify recovery semantics without exposing exception contents."""
    if isinstance(error, SandboxValidationError):
        return FailureClass.POLICY
    if isinstance(error, (MemoryError, sqlite3.OperationalError)):
        return FailureClass.RESOURCE
    if isinstance(error, (LLMError, SandboxTimeoutError, TimeoutError)):
        return FailureClass.TRANSIENT
    return FailureClass.DETERMINISTIC


def _dataset_request(
    settings: Settings,
    *,
    dataset: str,
    target: str | None,
    task: str,
    run_id: str,
    case_fingerprint: str,
    split_seed: int,
    cv_folds: int,
) -> DatasetRequest:
    """Build the attempt's DatasetRequest, threading the configured evaluation protocol.

    Under ``holdout`` the configured fraction reserves a deterministic evaluation
    partition; ``compatibility`` reproduces legacy all-row evaluation with a zero
    fraction (ADR 0018 decisions 1-2).
    """
    evaluation = settings.evaluation
    protocol = evaluation.protocol
    fraction = evaluation.evaluation_holdout_fraction if protocol == "holdout" else 0.0
    return DatasetRequest(
        name=dataset,
        target=target,
        task=task,
        run_id=run_id,
        case_fingerprint=case_fingerprint,
        split_seed=split_seed,
        cv_folds=cv_folds,
        evaluation_protocol=protocol,
        evaluation_holdout_fraction=fraction,
        source_policy="snapshot",
    )


def _fit_method_on_discovery(method: BaseMethod, silver: SilverPackage) -> None:
    """Fit the method on discovery-partition rows only (ADR 0018 decision 3).

    Evaluation-partition rows and targets never reach method fitting or the
    method-internal candidate trials/selection that run inside ``fit``.
    Generated code may still execute on the complete feature frame after
    fitting, but never receives evaluation targets. (Platinum-side
    selection/reporting partition enforcement lands in plan 23 PR 3.)
    """
    partitions = silver.fold_assignments.get("partition")
    if not isinstance(partitions, pd.Series):
        raise DatasetError(
            "Silver fold_assignments lacks a 'partition' column; regenerate the "
            "Silver package under a partition-aware evaluation protocol (ADR 0018) "
            "before fitting methods"
        )
    mask = (partitions == "discovery").to_numpy()
    if not mask.any():
        raise DatasetError(
            "Silver package contains no discovery-partition rows; methods cannot "
            "be fit without an independent discovery partition (ADR 0018)"
        )
    features = silver.canonical_features.loc[mask].reset_index(drop=True)
    target_series = silver.canonical_target.loc[mask, silver.bronze.target].reset_index(drop=True)
    method.fit(features, target_series)


class HamiltonLayerExecutor:
    """Coordinate four stage-specific drivers and verify every durable edge."""

    def __init__(
        self,
        *,
        settings: Settings,
        artifact_store: LocalArtifactStore,
        dataset_registry: DatasetRegistry,
        method_classes: dict[str, type[BaseMethod]],
        model_overrides: dict[str, Any] | None = None,
        metric_overrides: dict[str, Any] | None = None,
        lifecycle_repository: LocalRunRepository | None = None,
        resume_policy: ResumePolicy | None = None,
        profile: ExecutionProfile = ExecutionProfile.PRODUCTION,
    ) -> None:
        self.settings = settings
        self.artifact_store = artifact_store
        self.dataset_registry = dataset_registry
        self.method_classes = method_classes
        self.model_overrides = model_overrides or {}
        self.metric_overrides = metric_overrides or {}
        self.lifecycle_repository = lifecycle_repository
        self.resume_policy = resume_policy or ResumePolicy(
            enabled=True,
            recompute_invalid=True,
            allow_cross_run_reuse=True,
        )
        self.profile = profile

    def plan_case(self, case: ExperimentCase) -> CaseExecutionPlan:
        """Resolve static registry metadata without source, driver, cache, or writes."""
        metadata = self.dataset_registry.info(case.dataset)
        method_class = self.method_classes.get(case.method)
        if method_class is None:
            raise ValueError(f"Method '{case.method}' not found")
        identity = _method_identity(case.method, method_class, self.settings, case.mode)
        key = case.case_key or _case_key(case, self.settings, identity)
        attempt = case.attempt_id or case.run_id or allocate_attempt_id(key)
        return CaseExecutionPlan(
            case_key=key,
            attempt_id=attempt,
            dataset=case.dataset,
            method=case.method,
            model=case.model,
            resolved_metadata=normalize_secret_free(metadata),
            unresolved_layers=list(Layer),
        )

    def execute_case(self, case: ExperimentCase) -> ExperimentResult:
        """Execute or reuse a verified contiguous Bronze→Platinum chain."""
        stages: list[StageExecution] = []
        fallback_case_id = case.case_key or fingerprint(
            {
                "kind": "unresolved-experiment-case",
                "dataset": case.dataset,
                "method": case.method,
                "model": case.model,
                "seed": case.seed,
                "mode": case.mode,
            }
        )
        fallback_run_id = case.attempt_id or case.run_id or allocate_attempt_id(fallback_case_id)
        try:
            plan = self.plan_case(case)
            case = replace(
                case,
                case_key=plan.case_key,
                attempt_id=plan.attempt_id,
                run_id=plan.attempt_id,
            )
            self._record(
                run_id=plan.attempt_id,
                case_id=plan.case_key,
                event_type="case_running",
                state=RunState.RUNNING,
            )
            metadata = plan.resolved_metadata
            target = metadata.get("target")
            task = metadata.get("task", self.settings.task)
            if task not in {"classification", "regression"}:
                raise ValueError(f"Dataset '{case.dataset}' has invalid task '{task}'")
            request = _dataset_request(
                self.settings,
                dataset=case.dataset,
                target=str(target) if target else None,
                task=task,
                run_id=plan.attempt_id,
                case_fingerprint=plan.case_key,
                split_seed=case.seed,
                cv_folds=case.cv_folds or self.settings.evaluation.cv_folds,
            )
            environment = _environment(self.settings)
            cache_dir = resolve_cache_path(settings=self.settings)

            bronze_result = build_bronze_driver(
                profile=self.profile,
                settings=self.settings,
                artifact_store=self.artifact_store,
                cache_dir=cache_dir,
                telemetry_recorder=self._telemetry(plan.attempt_id, plan.case_key),
            ).execute(
                BRONZE_FINAL_VARS,
                inputs={
                    "dataset_request": request,
                    "dataset_registry": self.dataset_registry,
                    "environment_snapshot": environment,
                    "resume_policy": self.resume_policy,
                    "artifact_store": self.artifact_store,
                },
            )
            bronze = bronze_result["bronze_materialization"]
            bronze_ref = self._materialized_ref(bronze, Layer.BRONZE)
            self._verify_ref(bronze_ref, expected_upstreams=[])
            stages.append(self._stage(bronze, bronze_ref, plan.attempt_id))

            silver_reuse = dataset_fingerprint_value(
                bronze_result["bronze_snapshot"],
                bronze_result["dataset_computation_request"],
            )
            silver_ref = self._find_reusable(
                silver_reuse, Layer.SILVER, [bronze_ref], plan.attempt_id
            )
            if silver_ref is None:
                silver_result = build_silver_driver(
                    profile=self.profile,
                    settings=self.settings,
                    artifact_store=self.artifact_store,
                    cache_dir=cache_dir,
                    telemetry_recorder=self._telemetry(plan.attempt_id, plan.case_key),
                ).execute(
                    SILVER_FINAL_VARS,
                    inputs={
                        "dataset_request": request,
                        "dataset_computation_request": bronze_result["dataset_computation_request"],
                        "raw_dataset": bronze_result["raw_dataset"],
                        "bronze_snapshot": bronze_result["bronze_snapshot"],
                        "source_metadata": bronze_result["source_metadata"],
                        "bronze_materialization": bronze,
                        "environment_snapshot": environment,
                        "artifact_store": self.artifact_store,
                    },
                )
                silver_materialization = silver_result["silver_materialization"]
                silver_ref = self._materialized_ref(silver_materialization, Layer.SILVER)
                silver_disposition = StageDisposition.EXECUTED
            else:
                silver_materialization = None
                silver_disposition = StageDisposition.REUSED
            self._verify_ref(silver_ref, expected_upstreams=[bronze_ref])
            silver = load_silver_package(self.artifact_store, silver_ref)
            stages.append(
                self._loaded_stage(
                    silver_ref,
                    silver.manifest.reuse_fingerprint or silver_reuse,
                    silver.manifest.layer_fingerprint,
                    silver_disposition,
                )
            )

            method_class = self.method_classes[case.method]
            identity = _method_identity(case.method, method_class, self.settings, case.mode)
            prompt_fingerprint = fingerprint(identity.prompts)
            method_version = identity.distribution_version or identity.source_hash or "unknown"
            gold_reuse = gold_input_fingerprint(
                silver_fingerprint_value=silver.manifest.layer_fingerprint,
                method_name=case.method,
                method_version=method_version,
                method_config=identity.configuration,
                prompt_bundle_fingerprint=prompt_fingerprint,
                generated_contract_version="1",
                evaluation_protocol=self.settings.evaluation.protocol,
                selection_policy={"policy": "validation"},
            )
            gold_request = GoldRequest(
                run_id=plan.attempt_id,
                case_fingerprint=plan.case_key,
                silver_manifest=silver_ref,
                silver_fingerprint=silver.manifest.layer_fingerprint,
                gold_input_fingerprint=gold_reuse,
                method_name=case.method,
                method_version=method_version,
                method_config=identity.configuration,
                prompt_bundle_fingerprint=prompt_fingerprint,
                evaluation_protocol=self.settings.evaluation.protocol,
            )
            gold_ref = self._find_reusable(gold_reuse, Layer.GOLD, [silver_ref], plan.attempt_id)
            if gold_ref is None:
                method = self._build_method(method_class, case)
                _fit_method_on_discovery(method, silver)
                sandbox = SandboxedExecutor(
                    timeout_seconds=self.settings.evaluation.sandbox_timeout_seconds,
                    max_memory_mb=self.settings.evaluation.sandbox_max_memory_mb,
                    profile=self.settings.evaluation.sandbox_profile,
                )
                gold_result = build_gold_driver(
                    profile=self.profile,
                    settings=self.settings,
                    artifact_store=self.artifact_store,
                    cache_dir=cache_dir,
                    telemetry_recorder=self._telemetry(plan.attempt_id, plan.case_key),
                ).execute(
                    GOLD_FINAL_VARS,
                    inputs={
                        "gold_request": gold_request,
                        "method": method,
                        "verified_silver_package": silver,
                        "sandbox": sandbox,
                        "environment_snapshot": environment,
                        "artifact_store": self.artifact_store,
                    },
                )
                gold_materialization = gold_result["gold_materialization"]
                gold_ref = self._materialized_ref(gold_materialization, Layer.GOLD)
                gold_disposition = StageDisposition.EXECUTED
            else:
                gold_disposition = StageDisposition.REUSED
            self._verify_ref(gold_ref, expected_upstreams=[silver_ref])
            gold = load_gold_package(self.artifact_store, gold_ref)
            stages.append(
                self._loaded_stage(
                    gold_ref,
                    gold.manifest.reuse_fingerprint or gold_reuse,
                    gold.manifest.layer_fingerprint,
                    gold_disposition,
                )
            )

            for name, factory in self.model_overrides.items():
                ModelRegistry.register(name, factory)
            for name, metric in self.metric_overrides.items():
                MetricRegistry.register(name, metric)
            platinum_request = build_platinum_request(
                run_id=plan.attempt_id,
                case_fingerprint=plan.case_key,
                silver=silver,
                gold=gold,
                gold_ref=gold_ref,
                model_name=case.model,
                metric=self.settings.metric,
                seed=case.seed,
                evaluation_policy=EvaluationPolicy(
                    evaluation_protocol=self.settings.evaluation.protocol
                ),
            )
            platinum_reuse = platinum_request.platinum_input_fingerprint
            platinum_ref = self._find_reusable(
                platinum_reuse, Layer.PLATINUM, [silver_ref, gold_ref], plan.attempt_id
            )
            if platinum_ref is None:
                platinum_result = build_platinum_driver(
                    profile=self.profile,
                    settings=self.settings,
                    artifact_store=self.artifact_store,
                    cache_dir=cache_dir,
                    telemetry_recorder=self._telemetry(plan.attempt_id, plan.case_key),
                ).execute(
                    PLATINUM_FINAL_VARS,
                    inputs={
                        "platinum_request_input": platinum_request,
                        "verified_silver_package": silver,
                        "verified_gold_package": gold,
                        "environment_snapshot": environment,
                        "artifact_store": self.artifact_store,
                    },
                )
                platinum_materialization = platinum_result["platinum_materialization"]
                platinum_ref = self._materialized_ref(platinum_materialization, Layer.PLATINUM)
                platinum_disposition = StageDisposition.EXECUTED
            else:
                platinum_disposition = StageDisposition.REUSED
            self._verify_ref(platinum_ref, expected_upstreams=[silver_ref, gold_ref])
            platinum = load_platinum_package(self.artifact_store, platinum_ref)
            stages.append(
                self._loaded_stage(
                    platinum_ref,
                    platinum.manifest.reuse_fingerprint or platinum_reuse,
                    platinum.manifest.layer_fingerprint,
                    platinum_disposition,
                )
            )
            for stage in stages:
                self._record_stage(plan.attempt_id, plan.case_key, stage)
            self._record(
                run_id=plan.attempt_id,
                case_id=plan.case_key,
                event_type="case_succeeded",
                state=RunState.SUCCEEDED,
            )
            return ExperimentResult(
                dataset=case.dataset,
                method=case.method,
                model=case.model,
                seed=case.seed,
                # Legacy compat fields: ``gain`` is the raw (non-directional)
                # gain and ``cv_score`` the enhanced score. The headline is
                # the directional gain plus its paired Student-t interval
                # (ADR 0018 decisions 6-7, plan 23 PR 4).
                cv_score=platinum.aggregate.enhanced_score,
                gain=platinum.aggregate.legacy_gain,
                baseline_score=platinum.aggregate.baseline_score,
                directional_gain=platinum.aggregate.directional_gain,
                gain_lower_bound=platinum.uncertainty.lower_bound,
                gain_upper_bound=platinum.uncertainty.upper_bound,
                evaluation_protocol=platinum.request.evaluation_policy.evaluation_protocol,
                num_features_generated=len(gold.candidates),
                run_id=plan.attempt_id,
                case_fingerprint=plan.case_key,
                stages=stages,
            )
        except Exception as exc:
            failure = FailureRecord(
                failure_class=_classify_failure(exc),
                error_type=type(exc).__name__,
                message=str(exc),
                stage=stages[-1].layer.value if stages else "case",
            )
            run_id = case.attempt_id or case.run_id or fallback_run_id
            case_id = case.case_key or fallback_case_id
            for stage in stages:
                self._record_stage(run_id, case_id, stage)
            self._record(
                run_id=run_id,
                case_id=case_id,
                event_type="case_failed",
                state=RunState.FAILED,
                failure=failure,
            )
            return ExperimentResult(
                dataset=case.dataset,
                method=case.method,
                model=case.model,
                seed=case.seed,
                error=str(exc),
                run_id=case.attempt_id or case.run_id or fallback_run_id,
                case_fingerprint=case.case_key or fallback_case_id,
                stages=stages,
                failure=failure,
            )

    def _record(self, **values: Any) -> None:
        if self.lifecycle_repository is not None:
            self.lifecycle_repository.record(**values)

    def _telemetry(self, run_id: str, case_id: str) -> HamiltonTelemetryRecorder:
        sink = (
            create_run_event_sink(self.lifecycle_repository)
            if self.lifecycle_repository is not None
            else None
        )
        return HamiltonTelemetryRecorder(
            run_id=run_id,
            case_id=case_id,
            sink=sink,
            max_events=self.settings.dataflow.cache.telemetry_max_events,
        )

    def _record_stage(self, run_id: str, case_id: str, stage: StageExecution) -> None:
        self._record(
            run_id=run_id,
            case_id=case_id,
            event_type="stage_succeeded",
            state=RunState.SUCCEEDED,
            stage=stage.layer.value,
            layer=stage.layer,
            disposition=stage.disposition,
            manifest_ref=stage.manifest_ref,
            fingerprint=stage.layer_fingerprint,
        )

    def _build_method(self, method_class: type[BaseMethod], case: ExperimentCase) -> BaseMethod:
        parameters = inspect.signature(method_class.__init__).parameters
        kwargs: dict[str, Any] = {}
        if "config" in parameters:
            kwargs["config"] = self.settings
        if "mode" in parameters and case.mode is not None:
            kwargs["mode"] = case.mode
        if "artifact_config" in parameters:
            kwargs["artifact_config"] = None
        if "evaluator" in parameters:
            model_factory = ModelFactory(
                random_state=case.seed,
                booster_n_jobs=self.settings.evaluation.booster_n_jobs,
            )
            kwargs["evaluator"] = CVEvaluator(
                config=self.settings,
                model_factory=model_factory,
                default_model_name=case.model,
            )
        if "llm_client" in parameters:
            kwargs["llm_client"] = create_llm_client(
                self.settings.llm,
                self.settings.retry,
            )
        method = method_class(**kwargs)
        method.name = case.method
        return method

    def _find_reusable(
        self,
        reuse_fingerprint: str,
        layer: Layer,
        upstreams: list[Any],
        attempt_id: str,
    ) -> Any | None:
        if not self.resume_policy.enabled:
            return None
        try:
            return self.artifact_store.find_reusable(
                reuse_fingerprint,
                layer,
                upstreams=upstreams,
                preferred_run_id=attempt_id,
                allow_cross_run=self.resume_policy.allow_cross_run_reuse,
            )
        except (OSError, ValueError):
            if not self.resume_policy.recompute_invalid:
                raise
            return None

    def _verify_ref(self, ref: Any, *, expected_upstreams: list[Any]) -> None:
        report = self.artifact_store.verify(ref)
        if not report.valid:
            raise ValueError(
                f"{ref.layer.value} package failed verification: {'; '.join(report.errors)}"
            )
        manifest = self.artifact_store.load_manifest(ref)
        if manifest.upstream_manifests != expected_upstreams:
            raise ValueError(f"{ref.layer.value} package lineage does not match verified parents")

    @staticmethod
    def _materialized_ref(materialization: Any, layer: Layer) -> Any:
        if not materialization.persisted or materialization.manifest_ref is None:
            raise ValueError(f"{layer.value} stage did not publish durable evidence")
        return materialization.manifest_ref

    @staticmethod
    def _loaded_stage(
        ref: Any,
        reuse_fingerprint: str,
        layer_fingerprint: str,
        disposition: StageDisposition,
    ) -> StageExecution:
        return StageExecution(
            layer=ref.layer,
            disposition=disposition,
            manifest_ref=ref,
            reuse_fingerprint=reuse_fingerprint,
            layer_fingerprint=layer_fingerprint,
        )

    def _stage(self, materialization: Any, ref: Any, attempt_id: str) -> StageExecution:
        disposition = (
            StageDisposition.EXECUTED if ref.run_id == attempt_id else StageDisposition.REUSED
        )
        manifest = materialization.manifest
        return self._loaded_stage(
            ref,
            manifest.reuse_fingerprint or manifest.layer_fingerprint,
            manifest.layer_fingerprint,
            disposition,
        )


def build_worker_local_hamilton_executor(
    settings: Settings,
    payload: CaseComputationInput,
) -> HamiltonLayerExecutor:
    """Construct all mutable runtime services inside the worker process."""
    registry = DatasetRegistry()
    for name, info in (payload.dataset_overrides or {}).items():
        registry.register(name, info)
    methods = dict(MethodRegistry.get_all_methods())
    methods.update(payload.method_overrides or {})
    return HamiltonLayerExecutor(
        settings=settings,
        artifact_store=LocalArtifactStore(settings.dataflow.artifact_root),
        dataset_registry=registry,
        method_classes=methods,
        model_overrides=payload.model_overrides,
        metric_overrides=payload.metric_overrides,
        lifecycle_repository=LocalRunRepository(
            settings.dataflow.artifact_root / "control" / "lifecycle"
        ),
    )


def run_hamilton_case(payload: CaseComputationInput) -> ExperimentResult:
    """Top-level process worker; no tracker or live service crosses the seam."""
    settings = Settings.model_validate(payload.settings_data)
    return build_worker_local_hamilton_executor(settings, payload).execute_case(payload.case)
