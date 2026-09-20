"""Unified experimental platform for feature_forge.

Provides a one-liner API for running method comparison experiments.
Wraps DatasetRegistry, MethodRegistry, CVEvaluator, ModelFactory,
HamiltonLayerExecutor, and Reporter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any

import pandas as pd
from tqdm import tqdm

from feature_forge.config import FailurePolicy, Settings, get_settings
from feature_forge.contracts.stages import RunState
from feature_forge.data import DatasetRegistry
from feature_forge.dataflows.driver import resolve_cache_path
from feature_forge.evaluation import MetricRegistry, ModelRegistry
from feature_forge.experiment import ExperimentTracker, Reporter, create_tracker_from_config
from feature_forge.experiment.execution import (
    KEYBOARD_INTERRUPT_STOP_REASON,
    CancellationToken,
    CaseComputationInput,
    ExperimentCase,
    ExperimentResult,
    ProcessPoolExecutionAdapter,
    allocate_attempt_id,
    cancelled_failure_record,
)
from feature_forge.experiment.hamilton_executor import (
    HamiltonLayerExecutor,
    run_hamilton_case,
)
from feature_forge.experiment.lifecycle import LocalRunRepository
from feature_forge.methods import BaseMethod, MethodRegistry
from feature_forge.observability.structlog_config import get_logger
from feature_forge.runtime.intel_bootstrap import runtime_info
from feature_forge.storage.hamilton_cache import HamiltonCacheManager
from feature_forge.storage.hashing import fingerprint
from feature_forge.storage.local import LocalArtifactStore

logger = get_logger(__name__)

# Stop reason recorded on results/events drained after a fail-fast stop
# (ADR 0017). Token stops use the token's own winning reason instead.
FAIL_FAST_STOP_REASON = "fail_fast"


class ExperimentalPlatform:
    """Unified facade for feature engineering method comparison.

    Usage::

        platform = ExperimentalPlatform()
        results = platform.run(
            datasets=["titanic"],
            methods=["malmus", "caafe"],
            models=["random_forest"],
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

    def plan(
        self,
        datasets: list[str],
        methods: list[str],
        models: list[str] | None = None,
        mode: str | None = None,
        cv_folds: int | None = None,
        seeds: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return source-independent Hamilton plans without execution side effects."""
        settings = self._get_settings()
        executor = self._make_hamilton_executor(settings)
        plans = []
        for dataset in datasets:
            for method in methods:
                for model in models or ["random_forest"]:
                    for seed in seeds or [42]:
                        plan = executor.plan_case(
                            ExperimentCase(
                                dataset=dataset,
                                method=method,
                                model=model,
                                seed=seed,
                                mode=mode,
                                cv_folds=cv_folds,
                            )
                        )
                        plans.append(plan.model_dump(mode="json"))
        return plans

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
        *,
        failure_policy: FailurePolicy | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a method comparison experiment.

        Args:
            datasets: Dataset names to evaluate on.
            methods: Method names to compare.
            models: Model names for CV evaluation (default: ``['random_forest']``).
            mode: Method-specific mode (e.g. ``'single_shot'``, ``'iterative'``).
            cv_folds: Override CV folds for this run.
            seeds: Random seeds (default: ``[42]``).
            tracker: Optional experiment tracker override. When omitted, the
                tracker is built from ``settings.tracker`` (``backend`` /
                ``project`` / ``entity``) via ``create_tracker_from_config``.
            parallel: Run experiments in parallel via process pool.
            max_workers: Max parallel workers when ``parallel=True``.
            progress: Show ``tqdm`` progress bar.
            failure_policy: Optional per-run override of
                ``settings.execution.failure_policy`` (ADR 0017). Under
                ``fail_fast`` the scheduler stops scheduling new cases after
                the first terminal failed result — identically on the
                sequential and process paths — and later cases are returned
                as typed cancelled results. The override wins only for this
                invocation and never mutates cached settings.
            cancellation_token: Optional parent-process cooperative
                cancellation signal checked at case boundaries on both
                paths. A case in flight when cancellation arrives finishes
                with its real result; never-started cases are returned as
                cancelled results. On the process path at most
                ``max_workers`` futures are in flight (plan 22 §2.5);
                pending futures are cancelled and already-running cases keep
                their real results. Fail-fast is not hard termination.

        Returns:
            List of result dicts with keys: dataset, method, model, seed,
            cv_score, gain, baseline_score, num_features_generated, plus the
            additive directional evaluation fields ``directional_gain``,
            ``gain_lower_bound``, ``gain_upper_bound``, and
            ``evaluation_protocol`` (ADR 0018, plan 23 PR 4). ``gain`` is the
            legacy raw gain (compat-only, not the headline);
            ``directional_gain`` together with ``gain_lower_bound`` /
            ``gain_upper_bound`` carries the headline directional Student-t
            interval. Every requested case yields exactly one row in original
            matrix order, including cancelled never-started cases (additive
            ``state`` field, ADR 0017).

        Raises:
            KeyboardInterrupt: Propagated unchanged from either path after
                cleanup; the library never converts an interrupt into an
                ordinary failed experiment (plan 22 §2.6).
        """
        models = models or ["random_forest"]
        seeds = seeds or [42]
        policy_override = FailurePolicy(failure_policy) if failure_policy is not None else None

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
                            )
                        )

        run_settings = self._get_settings()
        effective_policy = run_settings.execution.failure_policy
        if policy_override is not None and policy_override != effective_policy:
            # Run-scoped snapshot (ADR 0017): the override wins for this
            # invocation only and must never mutate the cached/global Settings
            # instance that later runs and other callers see.
            run_settings = run_settings.model_copy(deep=True)
            run_settings.execution = run_settings.execution.model_copy(
                update={"failure_policy": policy_override}
            )
            effective_policy = policy_override

        run_tracker = tracker or create_tracker_from_config(run_settings.tracker)

        backend_results = self._run_hamilton(
            cases,
            run_settings,
            run_tracker,
            parallel,
            max_workers,
            progress,
            failure_policy=effective_policy,
            cancellation_token=cancellation_token,
        )

        return [self._result_to_dict(result) for result in backend_results]

    # ── Hamilton execution (sole engine, ADR 0016) ───────────

    def _run_hamilton(
        self,
        cases: list[ExperimentCase],
        run_settings: Settings,
        run_tracker: ExperimentTracker,
        parallel: bool,
        max_workers: int,
        progress: bool,
        *,
        failure_policy: FailurePolicy,
        cancellation_token: CancellationToken | None,
    ) -> list[ExperimentResult]:
        """Execute cases through ``HamiltonLayerExecutor``.

        The parent allocates a stable ``case_key`` and a unique ``attempt_id``
        before scheduling, then owns all tracker side effects. Workers never
        touch the tracker; ``run_hamilton_case`` reconstructs each worker-local
        executor from serializable payload maps.

        The failure policy and cancellation token are parent-scheduler
        concerns (ADR 0017), honored identically on both paths: sequential
        checks them at case boundaries; the process path drives the bounded
        submission window with a parent-side ``stop_on_result`` callback
        (fail-fast) plus the token, and drains never-started cases through
        the same typed cancelled-result factory used sequentially (plan 22
        §2.5).
        """
        if parallel:
            planner = self._make_hamilton_executor(run_settings)
            repository = self._lifecycle_repository(run_settings)
            payloads = [self._build_hamilton_payload(planner, case, run_settings) for case in cases]
            pool_backend = ProcessPoolExecutionAdapter(max_workers=max_workers)
            worker_results = pool_backend.run(
                payloads,
                run_hamilton_case,
                progress=progress,
                stop_on_result=(
                    # Fail-fast is a parent-side verdict per completed result
                    # (ADR 0017): the first terminal failure stops the window.
                    (lambda result: result.resolved_state is RunState.FAILED)
                    if failure_policy is FailurePolicy.FAIL_FAST
                    else None
                ),
                stop_reason=FAIL_FAST_STOP_REASON,
                token=cancellation_token,
                cancelled_result=self._cancelled_payload_result(repository),
            )
        else:
            executor = self._make_hamilton_executor(run_settings)
            worker_results = self._run_sequential(
                executor,
                self._lifecycle_repository(run_settings),
                cases,
                failure_policy=failure_policy,
                token=cancellation_token,
                progress=progress,
            )

        # Retention runs once after all workers close, never against an active
        # driver sharing the cache root.
        self._maintain_hamilton_cache(run_settings)

        # Parent owns tracker effects; workers never initialize a tracker.
        # Never-started (cancelled) cases initialize no tracker run (plan 22 §3).
        for case, result in zip(cases, worker_results, strict=True):
            if result.resolved_state is RunState.CANCELLED:
                continue
            self._apply_tracker_effects(run_tracker, run_settings, case, result)
        return worker_results

    def _run_sequential(
        self,
        executor: HamiltonLayerExecutor,
        repository: LocalRunRepository,
        cases: list[ExperimentCase],
        *,
        failure_policy: FailurePolicy,
        token: CancellationToken | None,
        progress: bool,
    ) -> list[ExperimentResult]:
        """Execute cases in matrix order under the failure/cancellation policy.

        Plan 22 §2.4 / ADR 0017: under ``continue`` every case executes
        exactly as before. Under ``fail_fast`` the first terminal failed
        result stops scheduling. The token is checked before each case
        starts; a case in flight when cancellation arrives finishes with its
        real result. Cases that never start are drained in order into typed
        cancelled results — no method, provider, driver, or tracker run is
        ever constructed for them, and each gets one redacted
        ``case_cancelled`` lifecycle event.
        """
        results: list[ExperimentResult] = []
        stop_reason: str | None = None
        iterator = tqdm(cases, total=len(cases), desc="Experiments") if progress else cases
        for position, case in enumerate(iterator):
            if stop_reason is None and token is not None and token.is_cancelled():
                stop_reason = token.reason or "operator_request"
            if stop_reason is not None:
                results.append(
                    self._cancelled_result(executor, repository, case=case, reason=stop_reason)
                )
                continue
            plan = executor.plan_case(case)
            scheduled = replace(
                case,
                case_key=plan.case_key,
                attempt_id=plan.attempt_id,
                run_id=plan.attempt_id,
            )
            try:
                result = executor.execute_case(scheduled)
            except KeyboardInterrupt:
                # plan 22 §2.6: the in-flight case is left to its own timeout
                # governance; journal redacted cancellation for the cases that
                # never started, then re-raise. An interrupt is never converted
                # into an ordinary failed experiment.
                for unstarted in cases[position + 1 :]:
                    self._cancelled_result(
                        executor,
                        repository,
                        case=unstarted,
                        reason=KEYBOARD_INTERRUPT_STOP_REASON,
                    )
                raise
            results.append(result)
            if (
                failure_policy is FailurePolicy.FAIL_FAST
                and result.resolved_state is RunState.FAILED
            ):
                stop_reason = FAIL_FAST_STOP_REASON
        return results

    def _cancelled_result(
        self,
        executor: HamiltonLayerExecutor,
        repository: LocalRunRepository,
        *,
        case: ExperimentCase,
        reason: str,
    ) -> ExperimentResult:
        """Build the typed cancelled result for a case that never started.

        The parent allocates the case/attempt identity without executing
        anything, then delegates to the shared identity-based factory.
        """
        case_fingerprint, attempt_id = self._allocate_cancelled_identity(executor, case)
        return self._cancelled_result_with_identity(
            repository,
            case=case,
            case_fingerprint=case_fingerprint,
            attempt_id=attempt_id,
            reason=reason,
        )

    def _cancelled_result_with_identity(
        self,
        repository: LocalRunRepository,
        *,
        case: ExperimentCase,
        case_fingerprint: str,
        attempt_id: str,
        reason: str,
    ) -> ExperimentResult:
        """Typed cancelled row plus one redacted ``case_cancelled`` event.

        Shared by the sequential drain and the process-path cancelled-row
        factory so both produce indistinguishable rows: no stage packages,
        no fabricated score, explicit ``CANCELLED`` state, and a journal
        event carrying only the cancellation category and identity (plan 22
        §3).
        """
        failure = cancelled_failure_record(case_fingerprint=case_fingerprint, reason=reason)
        repository.record(
            run_id=attempt_id,
            case_id=case_fingerprint,
            event_type="case_cancelled",
            state=RunState.CANCELLED,
            fingerprint=case_fingerprint,
            failure=failure,
        )
        return ExperimentResult(
            dataset=case.dataset,
            method=case.method,
            model=case.model,
            seed=case.seed,
            run_id=attempt_id,
            case_fingerprint=case_fingerprint,
            error=failure.message,
            failure=failure,
            state=RunState.CANCELLED,
        )

    def _cancelled_payload_result(
        self,
        repository: LocalRunRepository,
    ) -> Callable[[CaseComputationInput, str], ExperimentResult]:
        """Cancelled-row factory for process cases (plan 22 §2.5).

        Process-case identity is already allocated parent-side in
        ``_build_hamilton_payload``, so cancelled rows reuse the payload's
        case identity — no extra planning calls — and journal the same
        redacted ``case_cancelled`` event as the sequential path.
        """

        def factory(payload: CaseComputationInput, reason: str) -> ExperimentResult:
            scheduled = payload.case
            case_fingerprint = scheduled.case_key or self._unresolved_case_fingerprint(scheduled)
            attempt_id = (
                scheduled.attempt_id or scheduled.run_id or allocate_attempt_id(case_fingerprint)
            )
            return self._cancelled_result_with_identity(
                repository,
                case=scheduled,
                case_fingerprint=case_fingerprint,
                attempt_id=attempt_id,
                reason=reason,
            )

        return factory

    @staticmethod
    def _unresolved_case_fingerprint(case: ExperimentCase) -> str:
        """Content fingerprint used when registry identity cannot be resolved."""
        return case.case_key or fingerprint(
            {
                "kind": "unresolved-experiment-case",
                "dataset": case.dataset,
                "method": case.method,
                "model": case.model,
                "seed": case.seed,
                "mode": case.mode,
            }
        )

    @staticmethod
    def _allocate_cancelled_identity(
        executor: HamiltonLayerExecutor,
        case: ExperimentCase,
    ) -> tuple[str, str]:
        """Allocate the parent-side identity for a never-started case.

        ``plan_case`` resolves registry metadata without source, driver,
        cache, or writes. If even that static metadata is unavailable (e.g.
        an unregistered method draining after a fail-fast stop), fall back to
        the same content fingerprint the worker uses for unresolved cases so
        the cancelled row still carries a stable, unique identity.
        """
        try:
            plan = executor.plan_case(case)
        except Exception as exc:
            logger.warning("cancelled_case_identity_fallback", error_type=type(exc).__name__)
            fallback = ExperimentalPlatform._unresolved_case_fingerprint(case)
            return fallback, case.attempt_id or case.run_id or allocate_attempt_id(fallback)
        return plan.case_key, plan.attempt_id

    @staticmethod
    def _lifecycle_repository(run_settings: Settings) -> LocalRunRepository:
        """Parent-side lifecycle journal under the run's artifact root."""
        return LocalRunRepository(run_settings.dataflow.artifact_root / "control" / "lifecycle")

    @staticmethod
    def _maintain_hamilton_cache(run_settings: Settings) -> None:
        policy = run_settings.dataflow.cache
        if policy.max_age_days is None and policy.max_size_mb is None:
            return
        try:
            HamiltonCacheManager(resolve_cache_path(settings=run_settings)).collect(
                max_age_days=policy.max_age_days,
                max_size_mb=policy.max_size_mb,
            )
        except (OSError, ValueError) as exc:
            logger.warning("hamilton_cache_gc_failed", error_type=type(exc).__name__)

    def _build_hamilton_payload(
        self,
        planner: HamiltonLayerExecutor,
        case: ExperimentCase,
        run_settings: Settings,
    ) -> CaseComputationInput:
        """Allocate case identity, then send only serializable maps to the worker."""
        plan = planner.plan_case(case)
        scheduled = replace(
            case,
            case_key=plan.case_key,
            attempt_id=plan.attempt_id,
            run_id=plan.attempt_id,
        )
        return CaseComputationInput(
            case=scheduled,
            settings_data=run_settings.model_dump(),
            dataset_overrides=self._extra_datasets,
            method_overrides=self._extra_methods or None,
            model_overrides=self._extra_models or None,
            metric_overrides=self._extra_metrics or None,
        )

    def _make_hamilton_executor(self, run_settings: Settings) -> HamiltonLayerExecutor:
        """Parent-local executor assembled from the platform's registries/overrides."""
        dataset_registry = DatasetRegistry()
        for name, info in self._extra_datasets.items():
            dataset_registry.register(name, info)
        method_classes: dict[str, type[BaseMethod]] = dict(MethodRegistry.get_all_methods())
        method_classes.update(self._extra_methods)
        return HamiltonLayerExecutor(
            settings=run_settings,
            artifact_store=LocalArtifactStore(run_settings.dataflow.artifact_root),
            dataset_registry=dataset_registry,
            method_classes=method_classes,
            model_overrides=self._extra_models or None,
            metric_overrides=self._extra_metrics or None,
            lifecycle_repository=self._lifecycle_repository(run_settings),
        )

    def _apply_tracker_effects(
        self,
        run_tracker: ExperimentTracker,
        run_settings: Settings,
        case: ExperimentCase,
        result: ExperimentResult,
    ) -> None:
        """Apply tracker side effects exactly once for a finished Hamilton case."""
        run_name = result.run_id or case.effective_attempt_id
        config: dict[str, Any] = {
            "dataset": case.dataset,
            "method": case.method,
            "model": case.model,
            "seed": case.seed,
            "mode": case.mode,
            "cv_folds": case.cv_folds,
            # Effective failure policy for this run (ADR 0017): the
            # run-scoped snapshot, not the cached global settings value.
            "failure_policy": run_settings.execution.failure_policy.value,
            # Runtime provenance (ADR 0010): actual Intel/sklearnex state,
            # not just the config flag — records which numeric backend
            # produced the scores.
            "runtime": runtime_info(),
            # Resolved evaluation settings are retained with tracker output;
            # sandbox profile/degradation is never inferred from the host.
            "evaluation": run_settings.evaluation.model_dump(mode="json"),
            "sandbox_profile": run_settings.evaluation.sandbox_profile.value,
            "sandbox_degraded": run_settings.evaluation.sandbox_profile.value
            == "degraded_development",
        }
        if result.evaluation_protocol is not None:
            # Evaluation protocol provenance (ADR 0018, plan 23 PR 4):
            # compatibility-protocol results are selection-biased by
            # construction and must stay labelable downstream.
            config["evaluation_protocol"] = result.evaluation_protocol
            config["selection_biased"] = result.evaluation_protocol == "compatibility"
        run_tracker.init_run(run_name=run_name, config=config)
        try:
            if result.error is None:
                metrics = {
                    "cv_score": float(result.cv_score or 0.0),
                    "gain": float(result.gain or 0.0),
                    "baseline_score": float(result.baseline_score or 0.0),
                }
                # Directional headline fields (ADR 0018 decisions 6-7): the
                # directional gain and its paired Student-t interval. Logged
                # only when present so legacy rows keep the key set stable;
                # bounds are never coerced through ``float(x or 0.0)`` — the
                # 0.0-vs-``None`` distinction matters.
                if result.directional_gain is not None:
                    metrics["directional_gain"] = result.directional_gain
                if result.gain_lower_bound is not None:
                    metrics["gain_lower_bound"] = result.gain_lower_bound
                if result.gain_upper_bound is not None:
                    metrics["gain_upper_bound"] = result.gain_upper_bound
                run_tracker.log_metrics(metrics)
        finally:
            run_tracker.finish()

    @staticmethod
    def _result_to_dict(result: ExperimentResult) -> dict[str, Any]:
        """Serialize an ``ExperimentResult`` including nested ``StageExecution`` rows.

        ``dataclasses.asdict`` recurses the dataclass but leaves Pydantic
        ``StageExecution`` instances untouched, so stages are converted to plain
        dicts explicitly via ``model_dump``. ``state`` is the additive resolved
        terminal state (ADR 0017): explicit ``cancelled`` survives verbatim;
        otherwise derived from ``error``/``failure`` presence.
        """
        data = asdict(result)
        data["state"] = result.resolved_state.value
        data["stages"] = [stage.model_dump(mode="json") for stage in result.stages]
        data["failure"] = result.failure.model_dump(mode="json") if result.failure else None
        return data

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
