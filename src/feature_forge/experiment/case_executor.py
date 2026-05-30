"""Deep execution module for experiment case computation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from feature_forge.config import Settings
from feature_forge.data import DatasetRegistry
from feature_forge.evaluation import CVEvaluator, ModelFactory
from feature_forge.evaluation.metrics import MetricRegistry
from feature_forge.evaluation.model_factory import ModelRegistry
from feature_forge.exceptions import EvaluationError
from feature_forge.experiment.execution import ExperimentCase, ExperimentResult
from feature_forge.experiment.tracker import ExperimentTracker
from feature_forge.methods import BaseMethod, MethodRegistry
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaseComputationInput:
    """Serializable input for one case computation."""

    case: ExperimentCase
    settings_data: dict[str, Any]
    dataset_overrides: dict[str, dict[str, Any]] | None = None


@dataclass
class CaseComputation:
    """Pure computation for one experiment case."""

    all_methods: dict[str, type[BaseMethod]]

    def compute(self, payload: CaseComputationInput) -> ExperimentResult:
        case = payload.case
        run_settings = Settings(**payload.settings_data)
        if case.cv_folds is not None:
            run_settings.evaluation.cv_folds = case.cv_folds
        run_settings.random_state = case.seed

        model_factory = ModelFactory(random_state=case.seed)
        cv_evaluator = CVEvaluator(config=run_settings, model_factory=model_factory)
        dataset_registry = DatasetRegistry()
        for ds_name, ds_info in (payload.dataset_overrides or {}).items():
            dataset_registry.register(ds_name, ds_info)

        try:
            data = dataset_registry.load(case.dataset)
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

            method_kwargs: dict[str, Any] = {}
            sig = inspect.signature(method_cls.__init__)
            if "mode" in sig.parameters and case.mode is not None:
                method_kwargs["mode"] = case.mode
            if "artifact_config" in sig.parameters:
                method_kwargs["artifact_config"] = None
            method = method_cls(**method_kwargs)
            method.name = case.method

            method.fit(X, y)
            X_transformed = method.transform(X)
            baseline_score = cv_evaluator.evaluate_baseline(X, y, model_name=case.model)
            gain = cv_evaluator.evaluate_feature(
                X,
                y,
                X_transformed,
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
                num_features_generated=len(method.generated_scripts),
            )
        except (EvaluationError, ValueError, KeyError, ImportError) as exc:
            return ExperimentResult(
                dataset=case.dataset,
                method=case.method,
                model=case.model,
                seed=case.seed,
                error=str(exc),
            )
        except Exception as exc:
            logger.exception(
                "unexpected_experiment_error", dataset=case.dataset, method=case.method
            )
            return ExperimentResult(
                dataset=case.dataset,
                method=case.method,
                model=case.model,
                seed=case.seed,
                error=str(exc),
            )


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
    ) -> None:
        self.settings = settings
        self.tracker = tracker
        self.extra_methods = extra_methods or {}
        self.extra_datasets = extra_datasets or {}
        self.extra_models = extra_models or {}
        self.extra_metrics = extra_metrics or {}

        self._all_methods = dict(MethodRegistry.get_all_methods())
        self._all_methods.update(self.extra_methods)
        for name, fn in self.extra_models.items():
            ModelRegistry.register(name, fn)
        for name, fn in self.extra_metrics.items():
            MetricRegistry.register(name, fn)

    def _payload_for(self, case: ExperimentCase) -> CaseComputationInput:
        return CaseComputationInput(
            case=case,
            settings_data=self.settings.model_dump(),
            dataset_overrides=self.extra_datasets,
        )

    def execute(self, case: ExperimentCase) -> ExperimentResult:
        run_name = case.run_id or f"run_{case.dataset}_{case.method}_{case.model}_{case.seed}"
        self.tracker.init_run(
            run_name=run_name,
            config={
                "dataset": case.dataset,
                "method": case.method,
                "model": case.model,
                "seed": case.seed,
                "mode": case.mode,
                "cv_folds": case.cv_folds,
            },
        )
        try:
            result = CaseComputation(all_methods=self._all_methods).compute(self._payload_for(case))
            if result.error is None:
                metrics = {
                    "cv_score": float(result.cv_score or 0.0),
                    "gain": float(result.gain or 0.0),
                    "baseline_score": float(result.baseline_score or 0.0),
                }
                self.tracker.log_metrics(metrics)
            return result
        finally:
            self.tracker.finish()


def run_case(payload: CaseComputationInput) -> ExperimentResult:
    """Top-level worker for process-pool serialization seams."""
    all_methods = dict(MethodRegistry.get_all_methods())
    return CaseComputation(all_methods=all_methods).compute(payload)
