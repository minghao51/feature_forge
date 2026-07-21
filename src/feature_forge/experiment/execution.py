"""Execution backends for experiment cases."""

from __future__ import annotations

import gc
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.reduction import ForkingPickler
from pickle import PicklingError
from typing import Any, TypeVar, cast

from tqdm import tqdm

from feature_forge.contracts.orchestration import FailureRecord
from feature_forge.contracts.stages import FailureClass, RunState
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.exceptions import (
    CodeExecutionError,
    DatasetError,
    EvaluationError,
    LLMError,
    SandboxTimeoutError,
    SandboxValidationError,
    TrackingError,
    TransientNetworkError,
)
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)

_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|token)(\s*[:=]\s*)([^\s,;]+)"
)
_PROVIDER_KEY_PATTERN = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b")
_SERIALIZATION_ERRORS = (PicklingError, TypeError, AttributeError, ValueError, RecursionError)


def _serialization_failure(value: object) -> BaseException | None:
    """Capture arbitrary reducer failures only within the serialization boundary."""
    try:
        ForkingPickler.dumps(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except _SERIALIZATION_ERRORS as exc:
        return exc
    except BaseException as exc:
        return exc
    return None


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
    metric: str | None = None
    execution_profile: ExecutionProfile = ExecutionProfile.DEVELOPMENT
    artifact_policy: str = "legacy"


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
    run_id: str | None = None
    case_fingerprint: str | None = None
    state: str | None = None
    manifest_uri: str | None = None
    uncertainty: dict[str, float] | None = None
    num_candidate_features: int | None = None
    num_executed_features: int | None = None
    num_accepted_features: int | None = None
    num_accepted_output_columns: int | None = None
    feature_failure_counts: dict[str, int] | None = None
    silver_fingerprint: str | None = None
    gold_fingerprint: str | None = None
    platinum_fingerprint: str | None = None
    platinum_manifest_uri: str | None = None
    directional_gain: float | None = None
    selection_profile: str | None = None
    metric: str | None = None
    metric_direction: str | None = None
    failure: dict[str, Any] | None = None
    attempt: int = 1
    resource_plan: dict[str, Any] | None = None
    execution_plan: dict[str, Any] | None = None


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ExecutionBackend(ABC):
    """Interface for case execution backends."""

    @abstractmethod
    def run(
        self,
        cases: list[InputT],
        worker: Callable[[InputT], OutputT],
        progress: bool = True,
        fail_fast: bool = False,
        on_start: Callable[[InputT], None] | None = None,
    ) -> list[OutputT]:
        """Execute ``worker`` over ``cases`` and return outputs in order."""


class SequentialExecutionAdapter(ExecutionBackend):
    """Sequential backend."""

    def run(
        self,
        cases: list[InputT],
        worker: Callable[[InputT], OutputT],
        progress: bool = True,
        fail_fast: bool = False,
        on_start: Callable[[InputT], None] | None = None,
    ) -> list[OutputT]:
        results: list[OutputT] = []
        iterator = tqdm(cases, total=len(cases), desc="Experiments") if progress else cases
        failed = False
        for case in iterator:
            if failed:
                results.append(cast(OutputT, cancelled_result(case, "cancelled by fail_fast")))
                continue
            try:
                if on_start is not None:
                    on_start(case)
                result = worker(case)
            except BaseException as exc:
                result = cast(OutputT, failed_result(case, exc))
            results.append(result)
            failed = fail_fast and result_failed(result)
        return results


class ProcessPoolExecutionAdapter(ExecutionBackend):
    """Process-pool backend using a top-level worker.

    The default start method is platform-aware: ``fork`` on Linux (near-instant
    worker startup via copy-on-write of the parent's already-imported modules),
    ``spawn`` elsewhere (fork is deprecated on macOS, unavailable on Windows).
    Callers can override via ``mp_start_method``.
    """

    def __init__(
        self,
        max_workers: int = 1,
        *,
        mp_start_method: str | None = None,
    ) -> None:
        self.max_workers = max_workers
        if mp_start_method is None:
            mp_start_method = "fork" if sys.platform.startswith("linux") else "spawn"
        self.mp_start_method = mp_start_method

    def run(
        self,
        cases: list[InputT],
        worker: Callable[[InputT], OutputT],
        progress: bool = True,
        fail_fast: bool = False,
        on_start: Callable[[InputT], None] | None = None,
    ) -> list[OutputT]:
        if not cases:
            return []
        worker_serialization_failure = _serialization_failure(worker)
        if worker_serialization_failure is not None:
            exc = worker_serialization_failure
            first = cast(OutputT, failed_result(cases[0], exc, stage="serialization"))
            if fail_fast:
                return [
                    first,
                    *[
                        cast(OutputT, cancelled_result(case, "cancelled by fail_fast"))
                        for case in cases[1:]
                    ],
                ]
            return [
                first,
                *[
                    cast(OutputT, failed_result(case, exc, stage="serialization"))
                    for case in cases[1:]
                ],
            ]

        results: list[OutputT | None] = [None] * len(cases)
        next_index = 0
        stopped = False
        context = get_context(self.mp_start_method)
        progress_bar = tqdm(total=len(cases), desc="Experiments (parallel)") if progress else None
        try:
            executor = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=context)
        except OSError as exc:
            # The runner cannot allocate a process pool right now. On Linux this
            # surfaces as OSError [Errno 12] Cannot allocate memory when the
            # ProcessPoolExecutor's internal call/result Queues try to create
            # SemLocks under /dev/shm. Fall back to in-process sequential
            # execution so callers still get correct results — they just lose
            # parallelism. This is also the correct behavior for environments
            # that ban multiprocessing entirely.
            logger.warning(
                "process_pool_unavailable_fallback_sequential",
                error=str(exc),
                max_workers=self.max_workers,
            )
            return self._run_sequential(cases, worker, progress=progress, fail_fast=fail_fast, on_start=on_start)
        pending: dict[Future[OutputT], int] = {}
        try:
            while next_index < len(cases) and len(pending) < self.max_workers:
                next_index, submission_failed = self._submit(
                    cases, worker, executor, pending, results, next_index, on_start
                )
                if fail_fast and submission_failed:
                    stopped = True
                    break
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    try:
                        result = future.result()
                    except BaseException as exc:
                        logger.error("parallel_case_failed", index=index, error=str(exc))
                        result = cast(OutputT, failed_result(cases[index], exc))
                    results[index] = result
                    if progress_bar is not None:
                        progress_bar.update(1)
                    if fail_fast and result_failed(result):
                        stopped = True
                while not stopped and next_index < len(cases) and len(pending) < self.max_workers:
                    next_index, submission_failed = self._submit(
                        cases, worker, executor, pending, results, next_index, on_start
                    )
                    if fail_fast and submission_failed:
                        stopped = True
            if stopped:
                for index in range(next_index, len(cases)):
                    results[index] = cast(
                        OutputT, cancelled_result(cases[index], "cancelled by fail_fast")
                    )
                    if progress_bar is not None:
                        progress_bar.update(1)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            if progress_bar is not None:
                progress_bar.close()
            # ProcessPoolExecutor's internal call/result queues allocate
            # SemLocks from /dev/shm that are only freed when the executor
            # object is garbage-collected. On long test suites the cumulative
            # leaked semaphores exhaust /dev/shm and the next
            # ProcessPoolExecutor(...) fails with OSError [Errno 12] Cannot
            # allocate memory at SemLock.__init__. Drop the local ref and
            # force a collection so the SemLocks are released before return.
            del executor
            gc.collect()
        return [cast(OutputT, result) for result in results]

    @staticmethod
    def _run_sequential(
        cases: list[InputT],
        worker: Callable[[InputT], OutputT],
        *,
        progress: bool,
        fail_fast: bool,
        on_start: Callable[[InputT], None] | None,
    ) -> list[OutputT]:
        """In-process fallback used when ProcessPoolExecutor cannot start.

        Mirrors :class:`SequentialExecutionAdapter.run` but additionally
        performs the same per-case serialization check that the process-pool
        path does, so unpickleable cases are reported as ``stage="serialization"``
        failures rather than silently succeeding in-process.
        """
        results: list[OutputT] = []
        iterator = tqdm(cases, total=len(cases), desc="Experiments") if progress else cases
        failed = False
        for case in iterator:
            if failed:
                results.append(cast(OutputT, cancelled_result(case, "cancelled by fail_fast")))
                continue
            serialization_failure = _serialization_failure(case)
            if serialization_failure is not None:
                results.append(
                    cast(
                        OutputT,
                        failed_result(case, serialization_failure, stage="serialization"),
                    )
                )
                failed = fail_fast
                continue
            try:
                if on_start is not None:
                    on_start(case)
                result = worker(case)
            except BaseException as exc:
                result = cast(OutputT, failed_result(case, exc))
            results.append(result)
            failed = fail_fast and result_failed(result)
        return results

    @staticmethod
    def _submit(
        cases: list[InputT],
        worker: Callable[[InputT], OutputT],
        executor: ProcessPoolExecutor,
        pending: dict[Future[OutputT], int],
        results: list[OutputT | None],
        index: int,
        on_start: Callable[[InputT], None] | None,
    ) -> tuple[int, bool]:
        case = cases[index]
        serialization_failure = _serialization_failure(case)
        if serialization_failure is not None:
            results[index] = cast(
                OutputT,
                failed_result(case, serialization_failure, stage="serialization"),
            )
            return index + 1, True
        try:
            future = executor.submit(worker, case)
            pending[future] = index
        except (BrokenProcessPool, RuntimeError) as exc:
            results[index] = cast(OutputT, failed_result(case, exc, stage="scheduler.submit"))
            return index + 1, True
        if on_start is not None:
            on_start(case)
        return index + 1, False


def result_failed(result: object) -> bool:
    """Return whether an arbitrary normalized result is terminally unsuccessful."""
    if isinstance(result, ExperimentResult):
        return result.error is not None or result.state in {
            RunState.FAILED.value,
            RunState.CANCELLED.value,
        }
    return bool(getattr(result, "error", None))


def classify_failure(exc: BaseException) -> FailureClass:
    """Classify failures conservatively for retry and reporting."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return FailureClass.CANCELLED
    if isinstance(exc, SandboxValidationError):
        return FailureClass.POLICY
    if isinstance(exc, (SandboxTimeoutError, MemoryError, TimeoutError, BrokenProcessPool)):
        return FailureClass.RESOURCE
    if isinstance(exc, (TrackingError, TransientNetworkError)):
        return FailureClass.TRANSIENT
    if isinstance(exc, LLMError):
        from feature_forge.llm.retry import is_transient_llm_error

        return FailureClass.TRANSIENT if is_transient_llm_error(exc) else FailureClass.DETERMINISTIC
    if isinstance(exc, (DatasetError, EvaluationError, CodeExecutionError, ValueError, KeyError)):
        return FailureClass.DETERMINISTIC
    return FailureClass.DETERMINISTIC


def failure_record(exc: BaseException, *, stage: str = "case", attempt: int = 1) -> FailureRecord:
    """Build a bounded, pickle-safe exception record without traceback locals."""
    category = classify_failure(exc)
    causes: list[str] = []
    cause = exc.__cause__
    while cause is not None and len(causes) < 8:
        causes.append(type(cause).__name__)
        cause = cause.__cause__
    message = str(exc).strip() or type(exc).__name__
    message = _SECRET_PATTERN.sub(r"\1\2[REDACTED]", message)
    message = _PROVIDER_KEY_PATTERN.sub("[REDACTED]", message)
    return FailureRecord(
        failure_class=category,
        error_type=type(exc).__name__,
        message=message,
        stage=stage,
        retryable=category is FailureClass.TRANSIENT,
        attempt=attempt,
        cause_types=causes,
    )


def _case(value: object) -> object:
    return getattr(value, "case", value)


def failed_result(value: object, exc: BaseException, *, stage: str = "case") -> ExperimentResult:
    """Normalize an exception to the legacy-compatible terminal result shape."""
    case = _case(value)
    failure = failure_record(exc, stage=stage)
    return ExperimentResult(
        dataset=str(getattr(case, "dataset", "unknown")),
        method=str(getattr(case, "method", "unknown")),
        model=str(getattr(case, "model", "unknown")),
        seed=int(getattr(case, "seed", 0)),
        error=failure.message,
        run_id=getattr(case, "run_id", None),
        state=RunState.FAILED.value,
        failure=failure.model_dump(mode="json"),
    )


def cancelled_result(value: object, reason: str) -> ExperimentResult:
    """Return a terminal cancellation without running the case."""
    case = _case(value)
    failure = FailureRecord(
        failure_class=FailureClass.CANCELLED,
        error_type="CancelledError",
        message=reason,
        stage="scheduler",
    )
    return ExperimentResult(
        dataset=str(getattr(case, "dataset", "unknown")),
        method=str(getattr(case, "method", "unknown")),
        model=str(getattr(case, "model", "unknown")),
        seed=int(getattr(case, "seed", 0)),
        error=reason,
        run_id=getattr(case, "run_id", None),
        state=RunState.CANCELLED.value,
        failure=failure.model_dump(mode="json"),
    )
