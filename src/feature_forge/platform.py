"""Unified experimental platform for feature_forge.

Provides a one-liner API for running method comparison experiments.
Wraps DatasetRegistry, MethodRegistry, CVEvaluator, ModelFactory,
ExperimentCaseExecutor, and Reporter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from feature_forge.config import Settings, get_settings
from feature_forge.contracts.orchestration import (
    FailureRecord,
    ResourceConfig,
    ResumePlan,
    ResumePolicy,
)
from feature_forge.contracts.stages import Layer, RunState
from feature_forge.data import DatasetRegistry
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.evaluation import MetricRegistry, ModelRegistry
from feature_forge.experiment import ExperimentTracker, Reporter, create_tracker_from_config
from feature_forge.experiment.case_executor import (
    CaseComputationInput,
    ExperimentCaseExecutor,
    LayerExecutor,
    run_case,
)
from feature_forge.experiment.context import TaskType, validate_metric_for_task
from feature_forge.experiment.execution import (
    ExperimentCase,
    ExperimentResult,
    ProcessPoolExecutionAdapter,
    SequentialExecutionAdapter,
)
from feature_forge.experiment.lifecycle import LocalRunRepository
from feature_forge.experiment.resources import resolve_resource_plan
from feature_forge.experiment.resume import build_resume_plan
from feature_forge.methods import BaseMethod, MethodRegistry
from feature_forge.observability.structlog_config import get_logger
from feature_forge.storage.local import LocalArtifactStore

logger = get_logger(__name__)


class ExperimentalPlatform:
    """Unified facade for feature engineering method comparison.

    Usage::

        platform = ExperimentalPlatform()
        results = platform.run(
            datasets=["titanic"],
            methods=["malmus", "caafe"],
            models=["xgboost"],
        )
        platform.report(results)
    """

    def __init__(
        self,
        config: dict[str, Any] | Settings | None = None,
    ) -> None:
        self._config = config
        self._settings: Settings | None = None
        self._dataset_registry: DatasetRegistry | None = None
        self._extra_methods: dict[str, type[BaseMethod]] = {}
        self._extra_datasets: dict[str, dict[str, Any]] = {}
        self._extra_models: dict[str, Any] = {}
        self._extra_metrics: dict[str, Any] = {}

    # ── Lazy initializers ──────────────────────────────────────

    def _get_settings(self, **overrides: Any) -> Settings:
        if overrides:
            base = self._config_settings()
            run_settings = base.model_copy(deep=True)
            for key, value in overrides.items():
                setattr(run_settings, key, value)
            return run_settings
        return self._config_settings()

    def _config_settings(self) -> Settings:
        if self._settings is None:
            if isinstance(self._config, Settings):
                self._settings = self._config
            elif isinstance(self._config, dict):
                self._settings = Settings(**self._config)
            else:
                self._settings = get_settings()
        return self._settings

    def _get_dataset_registry(self) -> DatasetRegistry:
        if self._dataset_registry is None:
            self._dataset_registry = DatasetRegistry()
        return self._dataset_registry

    # ── Registration ───────────────────────────────────────────

    def register_method(self, name: str, cls: type[BaseMethod]) -> None:
        """Register a method class programmatically."""
        self._extra_methods[name] = cls

    def register_dataset(self, name: str, info: dict[str, Any]) -> None:
        """Register a dataset programmatically."""
        self._extra_datasets[name] = dict(info)
        self._get_dataset_registry().register(name, info)

    def register_model(self, name: str, factory_fn: Any) -> None:
        """Register a model factory function programmatically (instance-local)."""
        self._extra_models[name] = factory_fn

    def register_metric(self, name: str, fn: Any) -> None:
        """Register a metric function programmatically (instance-local)."""
        self._extra_metrics[name] = fn

    # ── Listing ────────────────────────────────────────────────

    def list_methods(self) -> list[str]:
        """List all available methods (built-in + discovered + registered)."""
        builtin = list(MethodRegistry.get_all_methods().keys())
        extra = list(self._extra_methods.keys())
        return sorted(set(builtin + extra))

    def list_datasets(self) -> list[str]:
        """List all available datasets."""
        return self._get_dataset_registry().list()

    def list_models(self) -> list[str]:
        """List all available models."""
        return sorted(set(ModelRegistry.list()) | set(self._extra_models.keys()))

    def list_metrics(self) -> list[str]:
        """List all available metrics."""
        return sorted(set(MetricRegistry.get_all().keys()) | set(self._extra_metrics.keys()))

    # ── Execution ──────────────────────────────────────────────

    def run(
        self,
        datasets: list[str],
        methods: list[str],
        models: list[str] | None = None,
        mode: str | None = None,
        cv_folds: int | None = None,
        seeds: list[int] | None = None,
        tracker: ExperimentTracker | None = None,
        parallel: bool = False,
        max_workers: int = 1,
        progress: bool = True,
        metric: str | None = None,
        execution_profile: ExecutionProfile = ExecutionProfile.DEVELOPMENT,
        artifact_policy: str = "legacy",
        fail_fast: bool = False,
        dry_run: bool = False,
        resources: ResourceConfig | dict[str, Any] | None = None,
        lifecycle_dir: str | Path | None = None,
        artifact_root: str | Path = ".feature_forge_artifacts",
        resume_policy: ResumePolicy | dict[str, Any] | None = None,
        layer_fingerprints: dict[str, dict[str, str]] | None = None,
        layer_executor: LayerExecutor | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a method comparison experiment.

        Args:
            datasets: Dataset names to evaluate on.
            methods: Method names to compare.
            models: Model names for CV evaluation (default: ``['xgboost']``).
            mode: Method-specific mode (e.g. ``'single_shot'``, ``'iterative'``).
            cv_folds: Override CV folds for this run.
            seeds: Random seeds (default: ``[42]``).
            tracker: Optional experiment tracker override. When omitted, the
                tracker is built from ``settings.tracker`` (``backend`` /
                ``project`` / ``entity``) via ``create_tracker_from_config``.
            parallel: Run experiments in parallel via process pool.
            max_workers: Max parallel workers when ``parallel=True``.
            progress: Show ``tqdm`` progress bar.

        Returns:
            List of result dicts with keys: dataset, method, model, seed,
            cv_score, gain, baseline_score, num_features_generated.
        """
        models = models or ["xgboost"]
        seeds = seeds or [42]

        cases: list[ExperimentCase] = []
        for ds_name in datasets:
            for method_name in methods:
                for model_name in models:
                    for seed in seeds:
                        cases.append(
                            ExperimentCase(
                                dataset=ds_name,
                                method=method_name,
                                model=model_name,
                                seed=seed,
                                mode=mode,
                                cv_folds=cv_folds,
                                run_id=f"run_{ds_name}_{method_name}_{model_name}_{seed}",
                                metric=metric,
                                execution_profile=execution_profile,
                                artifact_policy=artifact_policy,
                            )
                        )

        run_settings = self._get_settings()
        requested_resources = (
            ResourceConfig.model_validate(resources)
            if resources is not None
            else run_settings.resources
        )
        effective_plan = resolve_resource_plan(
            requested_resources,
            experiment_workers=max_workers if parallel else 1,
        )
        logger.info("resource_plan_resolved", **effective_plan.model_dump(mode="json"))
        resolved_resume_policy = (
            ResumePolicy.model_validate(resume_policy)
            if resume_policy is not None
            else ResumePolicy(enabled=artifact_policy != "legacy")
        )

        if artifact_policy not in {"legacy", "layer_boundaries"}:
            raise ValueError(f"unsupported artifact policy: {artifact_policy}")
        if artifact_policy != "legacy" and layer_fingerprints is None:
            raise ValueError(
                "non-legacy artifact policy requires per-run resolved layer_fingerprints"
            )
        if artifact_policy != "legacy":
            expected_layers = {layer.value for layer in Layer}
            for case in cases:
                run_id = case.effective_run_id
                case_fingerprints = (layer_fingerprints or {}).get(run_id)
                actual_layers = set(case_fingerprints or {})
                if actual_layers != expected_layers:
                    raise ValueError(
                        f"non-legacy run '{run_id}' requires exactly {sorted(expected_layers)} "
                        f"layer fingerprints; received {sorted(actual_layers)}"
                    )

        if dry_run:
            available_datasets = set(self.list_datasets())
            available_methods = set(self.list_methods())
            available_models = set(self.list_models())
            unknown = {
                "datasets": sorted(set(datasets) - available_datasets),
                "methods": sorted(set(methods) - available_methods),
                "models": sorted(set(models) - available_models),
                "metrics": [metric]
                if metric is not None and metric not in set(self.list_metrics())
                else [],
            }
            invalid = {key: values for key, values in unknown.items() if values}
            if invalid:
                raise ValueError(f"dry-run registry resolution failed: {invalid}")
            for dataset_name in datasets:
                metadata = self._get_dataset_registry().info(dataset_name)
                declared_task = metadata.get("task") if isinstance(metadata, dict) else None
                task = declared_task or run_settings.task
                if task not in {"classification", "regression"}:
                    raise ValueError(
                        f"dry-run dataset '{dataset_name}' has unsupported task: {task}"
                    )
                requested_metric = metric or (
                    run_settings.metric
                    if run_settings.metric
                    in (
                        {"auc", "acc", "f1"}
                        if task == "classification"
                        else {"rmse", "mae", "r2", "nrmse"}
                    )
                    else ("auc" if task == "classification" else "r2")
                )
                validate_metric_for_task(
                    requested_metric,
                    task=cast(TaskType, task),
                    dataset_name=dataset_name,
                )
            stages = (
                ["legacy_case"]
                if artifact_policy == "legacy"
                else ["bronze", "silver", "gold", "platinum"]
            )
            planned: list[ExperimentResult] = []
            store = LocalArtifactStore(artifact_root)
            for case in cases:
                resume_plan: ResumePlan | None = None
                if artifact_policy != "legacy":
                    run_id = case.run_id or (
                        f"run_{case.dataset}_{case.method}_{case.model}_{case.seed}"
                    )
                    case_fingerprints = (layer_fingerprints or {}).get(run_id)
                    if case_fingerprints is None:
                        raise ValueError(
                            f"dry-run is missing resolved layer fingerprints for '{run_id}'"
                        )
                    resume_plan = build_resume_plan(
                        store=store,
                        run_id=run_id,
                        expected_fingerprints={
                            Layer(name): value for name, value in case_fingerprints.items()
                        },
                        policy=resolved_resume_policy,
                        dry_run=True,
                    )
                planned.append(
                    ExperimentResult(
                        dataset=case.dataset,
                        method=case.method,
                        model=case.model,
                        seed=case.seed,
                        run_id=case.run_id,
                        state=RunState.PLANNED.value,
                        resource_plan=effective_plan.model_dump(mode="json"),
                        execution_plan={
                            "stages": stages,
                            "execution_profile": execution_profile.value,
                            "artifact_policy": artifact_policy,
                            "network_allowed": False,
                            "persistent_writes": False,
                            "resume": (
                                resume_plan.model_dump(mode="json")
                                if resume_plan is not None
                                else None
                            ),
                        },
                    )
                )
            return [result.__dict__ for result in planned]

        if artifact_policy != "legacy" and layer_executor is None:
            raise ValueError("non-legacy artifact policy requires a layer_executor")

        journal = (
            LocalRunRepository(lifecycle_dir)
            if lifecycle_dir is not None
            else (
                LocalRunRepository(".feature_forge_artifacts/control")
                if artifact_policy != "legacy"
                else None
            )
        )
        if journal is not None:
            for case in cases:
                run_id = case.effective_run_id
                journal.record(
                    run_id=run_id,
                    case_id=run_id,
                    event_type="case_planned",
                    state=RunState.PLANNED,
                    attempt=1,
                    details={"resource_plan": effective_plan.model_dump(mode="json")},
                )

        def record_case_running(value: ExperimentCase | CaseComputationInput) -> None:
            if journal is None:
                return
            case = value.case if isinstance(value, CaseComputationInput) else value
            run_id = case.effective_run_id
            journal.record(
                run_id=run_id,
                case_id=run_id,
                event_type="case_running",
                state=RunState.RUNNING,
                attempt=1,
            )

        if parallel:
            if tracker is not None:
                raise ValueError(
                    "parallel execution requires tracker configuration or a serializable tracker "
                    "factory; explicit tracker instances remain sequential-only"
                )
            pool_backend = ProcessPoolExecutionAdapter(max_workers=max_workers)
            payloads = [
                CaseComputationInput(
                    case=c,
                    settings_data=run_settings.model_dump(),
                    dataset_overrides=self._extra_datasets,
                    method_overrides=self._extra_methods,
                    model_overrides=self._extra_models,
                    metric_overrides=self._extra_metrics,
                    tracker_config=run_settings.tracker.model_dump(),
                    resource_plan=effective_plan.model_dump(mode="json"),
                    artifact_root=str(artifact_root),
                    resume_policy=resolved_resume_policy.model_dump(mode="json"),
                    layer_fingerprints=(layer_fingerprints or {}).get(c.effective_run_id),
                    layer_executor=layer_executor,
                )
                for c in cases
            ]
            backend_results = pool_backend.run(
                payloads,
                run_case,
                progress=progress,
                fail_fast=fail_fast,
                on_start=record_case_running,
            )
        else:
            run_tracker = tracker or create_tracker_from_config(run_settings.tracker)
            executor = ExperimentCaseExecutor(
                settings=run_settings,
                tracker=run_tracker,
                extra_methods=self._extra_methods,
                extra_datasets=self._extra_datasets,
                extra_models=self._extra_models,
                extra_metrics=self._extra_metrics,
                resource_plan=effective_plan,
                artifact_root=str(artifact_root),
                resume_policy=resolved_resume_policy,
                layer_fingerprints=layer_fingerprints,
                layer_executor=layer_executor,
                tracker_failure_policy=run_settings.tracker.failure_policy,
            )
            seq_backend = SequentialExecutionAdapter()
            backend_results = seq_backend.run(
                cases,
                executor.execute,
                progress=progress,
                fail_fast=fail_fast,
                on_start=record_case_running,
            )

        if journal is not None:
            for result in backend_results:
                run_id = (
                    result.run_id
                    or f"run_{result.dataset}_{result.method}_{result.model}_{result.seed}"
                )
                state = RunState(result.state or ("failed" if result.error else "succeeded"))
                resume_value = (result.execution_plan or {}).get("resume")
                completed_plan = (
                    ResumePlan.model_validate(resume_value)
                    if isinstance(resume_value, dict)
                    else None
                )
                if completed_plan is not None:
                    for decision in completed_plan.decisions:
                        stage_state = decision.state or RunState.SUCCEEDED
                        journal.record(
                            run_id=run_id,
                            case_id=run_id,
                            event_type="stage_terminal",
                            state=stage_state,
                            stage=decision.layer.value,
                            layer=decision.layer,
                            disposition=decision.disposition,
                            manifest_ref=decision.manifest_ref,
                            fingerprint=decision.expected_fingerprint,
                            attempt=result.attempt,
                            failure=(
                                FailureRecord.model_validate(result.failure)
                                if stage_state is RunState.FAILED and result.failure is not None
                                else None
                            ),
                        )
                terminal_ref = (
                    next(
                        (
                            decision.manifest_ref
                            for decision in reversed(completed_plan.decisions)
                            if decision.manifest_ref is not None
                        ),
                        None,
                    )
                    if completed_plan is not None
                    else None
                )
                terminal_fingerprint = (
                    result.platinum_fingerprint
                    or result.gold_fingerprint
                    or result.silver_fingerprint
                    or result.case_fingerprint
                    or (
                        next(
                            (
                                decision.expected_fingerprint
                                for decision in reversed(completed_plan.decisions)
                                if decision.state is RunState.SUCCEEDED
                            ),
                            None,
                        )
                        if completed_plan is not None
                        else None
                    )
                )
                journal.record(
                    run_id=run_id,
                    case_id=run_id,
                    event_type="case_terminal",
                    state=state,
                    stage="case",
                    failure=(
                        FailureRecord.model_validate(result.failure)
                        if result.failure is not None
                        else None
                    ),
                    manifest_ref=terminal_ref,
                    fingerprint=terminal_fingerprint,
                    attempt=result.attempt,
                    details={"resource_plan": result.resource_plan},
                )

        return [result.__dict__ for result in backend_results]

    # ── Reporting ──────────────────────────────────────────────

    def report(self, results: list[dict[str, Any]]) -> str:
        """Generate a markdown comparison table from results."""
        return Reporter(results).to_markdown()

    def report_best(
        self,
        results: list[dict[str, Any]],
        metric: str = "cv_score",
        group_by: str = "dataset",
    ) -> pd.DataFrame:
        """Get best result per group."""
        return Reporter(results).get_best(metric=metric, group_by=group_by)

    @staticmethod
    def to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert results to a raw pandas DataFrame."""
        return pd.DataFrame(results)
