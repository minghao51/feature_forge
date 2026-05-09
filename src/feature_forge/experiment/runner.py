"""Experiment runner for executing experiment matrices."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from feature_forge.experiment.tracker import ExperimentTracker
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class ExperimentRunner:
    """Execute experiment configurations sequentially or in parallel.

    Usage:
        runner = ExperimentRunner(tracker=tracker)
        results = runner.run(matrix.generate(), experiment_fn)
    """

    def __init__(
        self,
        tracker: ExperimentTracker | None = None,
        max_workers: int = 1,
    ) -> None:
        self.tracker = tracker
        self.max_workers = max_workers

    def run(
        self,
        configs: list[dict[str, Any]],
        experiment_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run all experiment configurations.

        Args:
            configs: List of parameter dictionaries.
            experiment_fn: Function that takes a config dict and returns results.

        Returns:
            List of result dictionaries.
        """
        results: list[dict[str, Any]] = []
        for i, config in enumerate(configs):
            run_name = f"run_{i}_{config.get('dataset', 'unknown')}"
            logger.info("experiment_run_start", run_name=run_name, config_keys=list(config.keys()))
            if self.tracker is not None:
                self.tracker.init_run(
                    run_name=run_name,
                    config=config,
                )
            try:
                result = experiment_fn(config)
                if self.tracker is not None:
                    self.tracker.log_metrics(
                        {k: v for k, v in result.items() if isinstance(v, float)}
                    )
                results.append({**config, **result})
                logger.info("experiment_run_complete", run_name=run_name)
            except Exception as exc:
                logger.error("experiment_run_error", run_name=run_name, error=str(exc))
                results.append({**config, "error": str(exc)})
            finally:
                if self.tracker is not None:
                    self.tracker.finish()
        return results

    def run_parallel(
        self,
        configs: list[dict[str, Any]],
        experiment_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run experiments in parallel using process pool.

        Note: experiment_fn must be pickleable.
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(experiment_fn, cfg) for cfg in configs]
            results = []
            for cfg, future in zip(configs, futures, strict=False):
                try:
                    result = future.result()
                    results.append({**cfg, **result})
                except Exception as exc:
                    results.append({**cfg, "error": str(exc)})
            return results
