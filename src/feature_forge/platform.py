"""Unified experimental platform for feature_forge.

Provides a one-liner API for running method comparison experiments.
Wraps DatasetRegistry, MethodRegistry, CVEvaluator, ModelFactory,
ExperimentCaseExecutor, and Reporter.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.config import Settings, get_settings
from feature_forge.data import DatasetRegistry
from feature_forge.evaluation import MetricRegistry, ModelRegistry
from feature_forge.experiment import ExperimentTracker, NoOpTracker, Reporter
from feature_forge.experiment.case_executor import (
    CaseComputationInput,
    ExperimentCaseExecutor,
    run_case,
)
from feature_forge.experiment.execution import (
    ExperimentCase,
    ProcessPoolExecutionAdapter,
    SequentialExecutionAdapter,
)
from feature_forge.methods import BaseMethod, MethodRegistry
from feature_forge.observability.structlog_config import get_logger

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
    ) -> list[dict[str, Any]]:
        """Execute a method comparison experiment.

        Args:
            datasets: Dataset names to evaluate on.
            methods: Method names to compare.
            models: Model names for CV evaluation (default: ``['xgboost']``).
            mode: Method-specific mode (e.g. ``'single_shot'``, ``'iterative'``).
            cv_folds: Override CV folds for this run.
            seeds: Random seeds (default: ``[42]``).
            tracker: Optional experiment tracker (default: ``NoOpTracker``).
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
                            )
                        )

        run_settings = self._get_settings()
        run_tracker = tracker or NoOpTracker(project="feature-forge-platform")
        executor = ExperimentCaseExecutor(
            settings=run_settings,
            tracker=run_tracker,
            extra_methods=self._extra_methods,
            extra_datasets=self._extra_datasets,
            extra_models=self._extra_models,
            extra_metrics=self._extra_metrics,
        )

        if parallel:
            if self._extra_methods:
                raise ValueError(
                    "parallel=True currently supports only registry-discovered methods; "
                    "instance-local extra methods are not serializable across process seam."
                )
            pool_backend = ProcessPoolExecutionAdapter(max_workers=max_workers)
            payloads = [
                CaseComputationInput(
                    case=c,
                    settings_data=run_settings.model_dump(),
                    dataset_overrides=self._extra_datasets,
                )
                for c in cases
            ]
            backend_results = pool_backend.run(payloads, run_case, progress=progress)
        else:
            seq_backend = SequentialExecutionAdapter()
            backend_results = seq_backend.run(cases, executor.execute, progress=progress)

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
