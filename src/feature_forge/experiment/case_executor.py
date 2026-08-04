"""Deep execution module for experiment case computation."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from feature_forge.config import Settings, TrackerConfig
from feature_forge.contracts.artifacts import ManifestRef
from feature_forge.contracts.orchestration import EffectiveResourcePlan, ResumePolicy
from feature_forge.contracts.stages import FailureClass, Layer, RunState
from feature_forge.data import DatasetRegistry
from feature_forge.evaluation.holdout import resolve_partition
from feature_forge.evaluation.metrics import MetricRegistry, get_metric_direction
from feature_forge.evaluation.model_factory import ModelRegistry
from feature_forge.exceptions import EvaluationError
from feature_forge.experiment.context import CaseExecutionContext, resolve_case_context
from feature_forge.experiment.execution import (
    ExperimentCase,
    ExperimentResult,
    failed_result,
    failure_record,
)
from feature_forge.experiment.factory import create_tracker_from_config
from feature_forge.experiment.resources import safer_resource_plan, settings_for_plan, thread_limits
from feature_forge.experiment.resume import (
    ResumeExecutionError,
    build_resume_plan,
    execute_resume_plan,
)
from feature_forge.experiment.tracker import ExperimentTracker
from feature_forge.methods import BaseMethod, MethodRegistry
from feature_forge.observability.structlog_config import get_logger
from feature_forge.storage.local import LocalArtifactStore

logger = get_logger(__name__)

LayerExecutor = Callable[[ExperimentCase, Layer, Mapping[Layer, ManifestRef]], ManifestRef]


@dataclass(frozen=True)
class CaseComputationInput:
    """Serializable input for one case computation."""

    case: ExperimentCase
    settings_data: dict[str, Any]
    dataset_overrides: dict[str, dict[str, Any]] | None = None
    method_overrides: dict[str, type[BaseMethod]] | None = None
    model_overrides: dict[str, Any] | None = None
    metric_overrides: dict[str, Any] | None = None
    tracker_config: dict[str, Any] | None = None
    resource_plan: dict[str, Any] | None = None
    artifact_root: str | None = None
    resume_policy: dict[str, Any] | None = None
    layer_fingerprints: dict[str, str] | None = None
    layer_executor: LayerExecutor | None = None


@dataclass
class CaseComputation:
    """Pure computation for one experiment case."""

    all_methods: dict[str, type[BaseMethod]]

    def compute(self, payload: CaseComputationInput) -> ExperimentResult:
        case = payload.case
        run_settings = Settings(**payload.settings_data)
        plan = (
            EffectiveResourcePlan.model_validate(payload.resource_plan)
            if payload.resource_plan is not None
            else None
        )
        if plan is not None:
            run_settings = settings_for_plan(run_settings, plan)
        if case.artifact_policy != "legacy":
            try:
                return self._compute_layered(payload, plan)
            except ResumeExecutionError as exc:
                result = failed_result(
                    payload,
                    exc.cause,
                    stage=f"resume.{exc.failed_layer.value}",
                )
                result.execution_plan = {"resume": exc.plan.model_dump(mode="json")}
                result.resource_plan = plan.model_dump(mode="json") if plan is not None else None
                return result
            except Exception as exc:
                return failed_result(payload, exc, stage="resume")
        dataset_registry = DatasetRegistry()
        for ds_name, ds_info in (payload.dataset_overrides or {}).items():
            dataset_registry.register(ds_name, ds_info)

        try:
            data = dataset_registry.load(case.dataset)
            try:
                raw_registry_metadata = dataset_registry.info(case.dataset)
            except KeyError:
                # Custom/ephemeral loaders may provide authoritative metadata only in load().
                raw_registry_metadata = {}
            registry_metadata = (
                raw_registry_metadata if isinstance(raw_registry_metadata, dict) else {}
            )
            context = resolve_case_context(
                case=case,
                base_settings=run_settings,
                loaded_dataset=data,
                registry_metadata=registry_metadata,
            )
            target_col = data.get("target")
            if target_col is None:
                raise EvaluationError(f"Dataset '{case.dataset}' has no target column")
            train_df = data.get("train")
            if train_df is None or train_df.empty:
                raise EvaluationError(f"Dataset '{case.dataset}' has no training data")

            y = train_df[target_col]
            X = train_df.drop(columns=[target_col])

            method_cls = self.all_methods.get(case.method)
            if method_cls is None:
                raise EvaluationError(f"Method '{case.method}' not found")

            method = self._construct_method(method_cls, context)
            method.name = case.method

            # Leakage-safe evaluation: carve a discovery subset (seen by
            # feature generation + selection) disjoint from an evaluation
            # subset (used only for the reported CV score). When the holdout
            # is disabled or the data is too small, the whole frame is used
            # for both — preserving the legacy behavior exactly.
            partition = resolve_partition(
                len(X),
                fraction=context.evaluation.evaluation_holdout_fraction,
                cv_folds=context.cv_folds,
                stratified=(
                    context.evaluation.evaluation_holdout_stratified
                    and context.task == "classification"
                ),
                seed=context.seed,
                y_for_stratify=y if context.task == "classification" else None,
            )
            if partition.holdout_active:
                X_discovery = X.iloc[partition.discovery_idx].reset_index(drop=True)
                y_discovery = y.iloc[partition.discovery_idx].reset_index(drop=True)
                X_eval = X.iloc[partition.evaluation_idx].reset_index(drop=True)
                y_eval = y.iloc[partition.evaluation_idx].reset_index(drop=True)
                n_discovery = len(X_discovery)
                n_evaluation = len(X_eval)
            else:
                if context.evaluation.evaluation_holdout_fraction > 0.0:
                    logger.warning(
                        "discovery_holdout_skipped",
                        dataset=case.dataset,
                        n_rows=len(X),
                        cv_folds=context.cv_folds,
                        fraction=context.evaluation.evaluation_holdout_fraction,
                    )
                X_discovery = X
                y_discovery = y
                X_eval = X
                y_eval = y
                n_discovery = len(X)
                n_evaluation = None

            # fit_transform() runs feature generation + supervised selection on
            # the discovery rows only; transform() then applies the discovered
            # code to the held-out evaluation rows for the reported score.
            X_discovery_transformed = method.fit_transform(X_discovery, y_discovery)
            accepted_output_columns = [
                str(column)
                for column in X_discovery_transformed.columns
                if column not in X_discovery.columns
            ]
            metadata_names = {
                str(item["name"])
                for item in method.feature_metadata
                if isinstance(item, dict) and item.get("name")
            }
            num_candidates = max(len(metadata_names), len(accepted_output_columns))
            not_materialized = max(num_candidates - len(accepted_output_columns), 0)

            X_eval_transformed = (
                method.transform(X_eval)
                if accepted_output_columns and partition.holdout_active
                else X_discovery_transformed
            )
            baseline_score = context.evaluator.evaluate_baseline(
                X_eval, y_eval, model_name=case.model
            )
            gain = context.evaluator.evaluate_feature(
                X_eval,
                y_eval,
                X_eval_transformed,
                baseline_score=baseline_score,
                model_name=case.model,
            )

            return ExperimentResult(
                dataset=case.dataset,
                method=case.method,
                model=case.model,
                seed=case.seed,
                cv_score=baseline_score + gain,
                gain=gain,
                baseline_score=baseline_score,
                num_features_generated=len(accepted_output_columns),
                num_candidate_features=num_candidates,
                num_executed_features=len(accepted_output_columns),
                num_accepted_features=len(accepted_output_columns),
                num_accepted_output_columns=len(accepted_output_columns),
                feature_failure_counts=(
                    {"not_materialized": not_materialized} if not_materialized else {}
                ),
                run_id=context.run_id,
                case_fingerprint=context.case_fingerprint,
                state=RunState.SUCCEEDED.value,
                resource_plan=plan.model_dump(mode="json") if plan is not None else None,
                n_discovery_rows=n_discovery,
                n_evaluation_rows=n_evaluation,
                metric=context.metric,
                metric_direction=get_metric_direction(context.metric).value,
            )
        except (EvaluationError, ValueError, KeyError, ImportError) as exc:
            return failed_result(payload, exc)
        except Exception as exc:
            logger.exception(
                "unexpected_experiment_error", dataset=case.dataset, method=case.method
            )
            return failed_result(payload, exc)

    @staticmethod
    def _compute_layered(
        payload: CaseComputationInput,
        resource_plan: EffectiveResourcePlan | None,
    ) -> ExperimentResult:
        """Execute a fully resolved durable layer plan instead of the legacy case path."""
        case = payload.case
        if payload.artifact_root is None:
            raise ValueError("non-legacy artifact policy requires artifact_root")
        if payload.layer_fingerprints is None:
            raise ValueError("non-legacy artifact policy requires resolved layer fingerprints")
        expected_layers = {layer.value for layer in Layer}
        if set(payload.layer_fingerprints) != expected_layers:
            raise ValueError(
                "non-legacy artifact policy requires exactly all four layer fingerprints"
            )
        if payload.layer_executor is None:
            raise ValueError("non-legacy artifact policy requires a layer executor")
        layer_executor = payload.layer_executor
        fingerprints = {Layer(name): value for name, value in payload.layer_fingerprints.items()}
        store = LocalArtifactStore(payload.artifact_root)
        policy = ResumePolicy.model_validate(payload.resume_policy or {"enabled": True})
        plan = build_resume_plan(
            store=store,
            run_id=case.effective_run_id,
            expected_fingerprints=fingerprints,
            policy=policy,
        )
        completed = execute_resume_plan(
            store=store,
            plan=plan,
            execute_stage=lambda layer, chain: layer_executor(case, layer, chain),
        )
        refs = {
            decision.layer: decision.manifest_ref
            for decision in completed.decisions
            if decision.manifest_ref is not None
        }
        try:
            silver_ref = refs[Layer.SILVER]
            gold_ref = refs[Layer.GOLD]
            platinum_ref = refs[Layer.PLATINUM]
        except KeyError as exc:
            raise ValueError(
                "completed layer plan is missing Silver/Gold/Platinum evidence"
            ) from exc
        from feature_forge.dataflows.gold import load_gold_package
        from feature_forge.dataflows.platinum import (
            experiment_result_from_package,
            load_platinum_package,
        )
        from feature_forge.dataflows.silver import load_silver_package

        silver = load_silver_package(store, silver_ref)
        gold = load_gold_package(store, gold_ref)
        platinum = load_platinum_package(store, platinum_ref)
        result = ExperimentResult(
            **experiment_result_from_package(
                platinum,
                silver=silver,
                gold=gold,
                manifest_uri=f"04_platinum/runs/{platinum_ref.run_id}/manifest.json",
            )
        )
        result.resource_plan = (
            resource_plan.model_dump(mode="json") if resource_plan is not None else None
        )
        result.execution_plan = {"resume": completed.model_dump(mode="json")}
        return result

    @staticmethod
    def _construct_method(
        method_cls: type[BaseMethod],
        context: CaseExecutionContext,
    ) -> BaseMethod:
        """Construct a method adapter, preferring an explicit factory.

        Built-in methods expose ``from_run_context`` so the case context
        (LLM client, evaluator, sandbox, settings, mode) is injected
        explicitly — no constructor-name reflection, no silently-dropped
        parameters. Third-party entry-point methods that do not implement
        ``from_run_context`` fall back to the legacy ``inspect.signature``
        path (``_method_kwargs``), which preserves the plugin story.
        """
        factory = getattr(method_cls, "from_run_context", None)
        if callable(factory):
            # Bound classmethod; cast because getattr() erases the type.
            method: BaseMethod = factory(context)
            return method
        return method_cls(**CaseComputation._method_kwargs(method_cls, context))

    @staticmethod
    def _method_kwargs(
        method_cls: type[BaseMethod],
        context: CaseExecutionContext,
    ) -> dict[str, Any]:
        """Legacy reflection-based kwargs for methods without ``from_run_context``.

        Used only as a fallback for third-party entry-point plugins that have
        not yet adopted ``from_run_context``. Built-in methods no longer go
        through this path. The historical gaps (no ``llm_client`` injection,
        ``mode`` silently dropped for MALMAS) are resolved by ``from_run_context``
        overrides on each built-in adapter.
        """
        parameters = inspect.signature(method_cls.__init__).parameters
        kwargs: dict[str, Any] = {}
        if "mode" in parameters and context.case.mode is not None:
            kwargs["mode"] = context.case.mode
        if "artifact_config" in parameters:
            kwargs["artifact_config"] = None
        if "settings" in parameters:
            kwargs["settings"] = context.settings
        if "config" in parameters:
            kwargs["config"] = context.settings
        if "evaluator" in parameters:
            kwargs["evaluator"] = context.evaluator
        if "metric" in parameters:
            kwargs["metric"] = context.metric
        if "n_jobs" in parameters:
            kwargs["n_jobs"] = context.evaluation.max_cv_workers or 1
        return kwargs


class ExperimentCaseExecutor:
    """Executor module that owns behavior and tracker side effects."""

    def __init__(
        self,
        settings: Settings,
        tracker: ExperimentTracker,
        extra_methods: dict[str, type[BaseMethod]] | None = None,
        extra_datasets: dict[str, dict[str, Any]] | None = None,
        extra_models: dict[str, Any] | None = None,
        extra_metrics: dict[str, Any] | None = None,
        resource_plan: EffectiveResourcePlan | None = None,
        artifact_root: str | None = None,
        resume_policy: ResumePolicy | None = None,
        layer_fingerprints: dict[str, dict[str, str]] | None = None,
        layer_executor: LayerExecutor | None = None,
        tracker_failure_policy: str = "optional",
    ) -> None:
        self.settings = settings
        self.tracker = tracker
        self.extra_methods = extra_methods or {}
        self.extra_datasets = extra_datasets or {}
        self.extra_models = extra_models or {}
        self.extra_metrics = extra_metrics or {}
        self.resource_plan = resource_plan
        self.artifact_root = artifact_root
        self.resume_policy = resume_policy
        self.layer_fingerprints = layer_fingerprints or {}
        self.layer_executor = layer_executor
        self.tracker_failure_policy = tracker_failure_policy

        self._all_methods = dict(MethodRegistry.get_all_methods())
        self._all_methods.update(self.extra_methods)
        for name, fn in self.extra_models.items():
            ModelRegistry.register(name, fn)
        for name, fn in self.extra_metrics.items():
            MetricRegistry.register(name, fn)

    def _payload_for(self, case: ExperimentCase) -> CaseComputationInput:
        run_id = case.effective_run_id
        return CaseComputationInput(
            case=case,
            settings_data=self.settings.model_dump(),
            dataset_overrides=self.extra_datasets,
            method_overrides=self.extra_methods,
            model_overrides=self.extra_models,
            metric_overrides=self.extra_metrics,
            resource_plan=(
                self.resource_plan.model_dump(mode="json")
                if self.resource_plan is not None
                else None
            ),
            artifact_root=self.artifact_root,
            resume_policy=(
                self.resume_policy.model_dump(mode="json")
                if self.resume_policy is not None
                else None
            ),
            layer_fingerprints=self.layer_fingerprints.get(run_id),
            layer_executor=self.layer_executor,
        )

    def execute(self, case: ExperimentCase) -> ExperimentResult:
        run_name = case.effective_run_id
        initialized = False
        tracking_failure = None
        try:
            try:
                self.tracker.init_run(run_name=run_name, config=_tracker_config(case))
                initialized = True
            except Exception as exc:
                tracking_failure = failure_record(exc, stage="tracker.init")
            result = CaseComputation(all_methods=self._all_methods).compute(self._payload_for(case))
            if initialized and result.error is None:
                metrics = {
                    "cv_score": float(result.cv_score or 0.0),
                    "gain": float(result.gain or 0.0),
                    "baseline_score": float(result.baseline_score or 0.0),
                }
                if result.n_evaluation_rows is not None:
                    metrics["n_evaluation_rows"] = float(result.n_evaluation_rows)
                    metrics["n_discovery_rows"] = float(result.n_discovery_rows or 0.0)
                try:
                    self.tracker.log_metrics(metrics)
                    if result.manifest_uri is not None:
                        self.tracker.log_artifact_reference(result.manifest_uri)
                except Exception as exc:
                    tracking_failure = failure_record(exc, stage="tracker.log")
        finally:
            if initialized:
                try:
                    self.tracker.finish()
                except Exception as exc:
                    logger.exception("tracker_finish_failed", run_name=run_name)
                    tracking_failure = failure_record(exc, stage="tracker.finish")
        if tracking_failure is not None and result.error is None:
            required = self.tracker_failure_policy == "required"
            result.state = RunState.FAILED.value if required else RunState.PARTIAL.value
            result.failure = tracking_failure.model_dump(mode="json")
            if required:
                result.error = tracking_failure.message
        return result


def run_case(payload: CaseComputationInput) -> ExperimentResult:
    """Spawn-safe worker with worker-local registration, tracking, and retry."""
    all_methods = dict(MethodRegistry.get_all_methods())
    all_methods.update(payload.method_overrides or {})
    for name, factory in (payload.model_overrides or {}).items():
        ModelRegistry.register(name, factory)
    for name, metric in (payload.metric_overrides or {}).items():
        MetricRegistry.register(name, metric)

    plan = (
        EffectiveResourcePlan.model_validate(payload.resource_plan)
        if payload.resource_plan is not None
        else None
    )
    attempt = 1
    while True:
        tracker_config = (
            payload.tracker_config or Settings(**payload.settings_data).tracker.model_dump()
        )
        tracker = create_tracker_from_config(TrackerConfig.model_validate(tracker_config))
        executor = ExperimentCaseExecutor(
            settings=Settings(**payload.settings_data),
            tracker=tracker,
            extra_methods=payload.method_overrides,
            extra_datasets=payload.dataset_overrides,
            extra_models=payload.model_overrides,
            extra_metrics=payload.metric_overrides,
            resource_plan=plan,
            artifact_root=payload.artifact_root,
            resume_policy=(
                ResumePolicy.model_validate(payload.resume_policy)
                if payload.resume_policy is not None
                else None
            ),
            layer_fingerprints=(
                {payload.case.effective_run_id: payload.layer_fingerprints}
                if payload.layer_fingerprints is not None
                else None
            ),
            layer_executor=payload.layer_executor,
            tracker_failure_policy=TrackerConfig.model_validate(tracker_config).failure_policy,
        )
        executor._all_methods = all_methods
        if plan is None:
            result = executor.execute(payload.case)
        else:
            with thread_limits(plan):
                result = executor.execute(payload.case)
        result.attempt = attempt
        result.resource_plan = plan.model_dump(mode="json") if plan is not None else None
        if result.failure is not None:
            result.failure["attempt"] = attempt
        failure_class = result.failure.get("failure_class") if result.failure is not None else None
        if plan is None or result.error is None or attempt >= plan.max_attempts:
            return result
        if failure_class == FailureClass.RESOURCE.value:
            safer = safer_resource_plan(plan)
            if safer is None:
                return result
            plan = safer
        elif failure_class != FailureClass.TRANSIENT.value:
            return result
        delay = min(plan.backoff_base_seconds * (2 ** (attempt - 1)), plan.backoff_max_seconds)
        time.sleep(delay)
        attempt += 1


def _tracker_config(case: ExperimentCase) -> dict[str, Any]:
    return {
        "dataset": case.dataset,
        "method": case.method,
        "model": case.model,
        "seed": case.seed,
        "mode": case.mode,
        "cv_folds": case.cv_folds,
    }
