"""Execution backends for experiment cases."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

from tqdm import tqdm

from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExperimentCase:
    """Serializable case payload for one experiment run."""

    dataset: str
    method: str
    model: str
    seed: int
    mode: str | None = None
    cv_folds: int | None = None
    run_id: str | None = None


@dataclass
class ExperimentResult:
    """Normalized case result payload."""

    dataset: str
    method: str
    model: str
    seed: int
    cv_score: float | None = None
    gain: float | None = None
    baseline_score: float | None = None
    num_features_generated: int | None = None
    error: str | None = None


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ExecutionBackend:
    """Interface for case execution backends."""

    def run(
        self,
        cases: list[InputT],
        worker: Callable[[InputT], OutputT],
        progress: bool = True,
    ) -> list[OutputT]:
        raise NotImplementedError


class SequentialExecutionAdapter(ExecutionBackend):
    """Sequential backend."""

    def run(
        self,
        cases: list[InputT],
        worker: Callable[[InputT], OutputT],
        progress: bool = True,
    ) -> list[OutputT]:
        results: list[OutputT] = []
        iterator = tqdm(cases, total=len(cases), desc="Experiments") if progress else cases
        for case in iterator:
            results.append(worker(case))
        return results


class ProcessPoolExecutionAdapter(ExecutionBackend):
    """Process-pool backend using a top-level worker."""

    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max_workers

    def run(
        self,
        cases: list[InputT],
        worker: Callable[[InputT], OutputT],
        progress: bool = True,
    ) -> list[OutputT]:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(worker, case) for case in cases]
            results: list[OutputT] = []
            iterator = (
                tqdm(
                    futures,
                    total=len(futures),
                    desc="Experiments (parallel)",
                )
                if progress
                else futures
            )
            for fut in iterator:
                try:
                    results.append(fut.result())
                except Exception as exc:
                    logger.error("parallel_case_failed", error=str(exc))
                    raise RuntimeError(f"Parallel case failed: {exc}") from exc
            return results
