"""Execution backends for experiment cases."""

from __future__ import annotations

import multiprocessing
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import uuid4

from feature_forge.contracts.orchestration import FailureRecord
from feature_forge.contracts.stages import FailureClass, RunState

if TYPE_CHECKING:
    from feature_forge.contracts.orchestration import StageExecution
    from feature_forge.methods import BaseMethod

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
    case_key: str | None = None
    attempt_id: str | None = None

    @property
    def effective_attempt_id(self) -> str:
        """Return the allocated attempt identifier or the compatibility run ID."""
        return (
            self.attempt_id
            or self.run_id
            or f"run_{self.dataset}_{self.method}_{self.model}_{self.seed}"
        )


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
    stages: list[StageExecution] = field(default_factory=list)
    failure: FailureRecord | None = None
    # Explicit terminal case state (ADR 0017). ``None`` keeps existing
    # constructors back-compatible; the scheduler derives the state when
    # unset. CANCELLED is only ever set explicitly, for cases that never ran
    # (or ran before the stop condition) — it is never derived.
    state: RunState | None = None

    @property
    def resolved_state(self) -> RunState:
        """Terminal case state, derived when not set explicitly.

        Derivation rule (ADR 0017): a result without an explicit ``state`` is
        ``FAILED`` when an error string or failure record is present, and
        ``SUCCEEDED`` otherwise. Serialization must always surface the
        resolved value so the public row is explicit.
        """
        if self.state is not None:
            return self.state
        if self.error is not None or self.failure is not None:
            return RunState.FAILED
        return RunState.SUCCEEDED


@dataclass(frozen=True)
class CaseComputationInput:
    """Serializable input for one case computation.

    Deliberately carries no runtime handles: no cancellation token, no failure
    policy, no provider, and no live service ever crosses the process seam
    (ADR 0017). The token is parent-process-only and checked at case
    boundaries; the failure policy is resolved by the parent scheduler.
    """

    case: ExperimentCase
    settings_data: dict[str, Any]
    dataset_overrides: dict[str, dict[str, Any]] | None = None
    method_overrides: dict[str, type[BaseMethod]] | None = None
    model_overrides: dict[str, Any] | None = None
    metric_overrides: dict[str, Any] | None = None


CANCELLED_ERROR_TYPE = "CaseCancelled"

# Stop reason recorded for cases proven not to have started after a
# ``KeyboardInterrupt`` (plan 22 §2.6). Shared by both schedulers so the
# sequential and process journals use one vocabulary.
KEYBOARD_INTERRUPT_STOP_REASON = "keyboard_interrupt"


def cancelled_failure_record(
    *,
    case_fingerprint: str | None = None,
    attempt: int = 1,
    reason: str = "operator_request",
) -> FailureRecord:
    """Build the typed cancellation record for a case that never ran.

    Carries only the cancellation category, the stable non-sensitive error
    type, and the parent-allocated case/attempt identity — never exception
    text, prompts, feature values, inputs, or secrets (ADR 0017, plan 22 §3).
    """
    identity = f" for case '{case_fingerprint}'" if case_fingerprint else ""
    return FailureRecord(
        failure_class=FailureClass.CANCELLED,
        error_type=CANCELLED_ERROR_TYPE,
        message=f"Case did not run: cancelled before start ({reason}){identity}.",
        stage="case",
        retryable=False,
        attempt=attempt,
    )


class CancellationToken:
    """Parent-process cooperative cancellation signal (ADR 0017).

    Thread-safe and idempotent: the first ``cancel`` reason wins and later
    calls cannot overwrite it. Cancellation is cooperative — the scheduler
    checks the token at case boundaries only; a case already in flight is
    allowed to finish safely. The token is never serialized into
    :class:`CaseComputationInput` and never crosses the process seam; workers
    observe cancellation only through results returned by the parent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: str | None = None

    @classmethod
    def cancelled(cls, reason: str = "operator_request") -> CancellationToken:
        """Return an already-cancelled token (e.g. for pre-cancelled runs)."""
        token = cls()
        token.cancel(reason)
        return token

    def cancel(self, reason: str = "operator_request") -> None:
        """Request cancellation; idempotent with first-reason-wins."""
        with self._lock:
            if self._reason is None:
                self._reason = reason

    @property
    def reason(self) -> str | None:
        """The winning cancellation reason, or ``None`` if not cancelled."""
        with self._lock:
            return self._reason

    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        with self._lock:
            return self._reason is not None


def allocate_attempt_id(case_key: str, *, now: datetime | None = None) -> str:
    """Allocate an identifier-safe, unique attempt namespace."""
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    safe_key = "".join(character if character.isalnum() else "-" for character in case_key)
    return f"{safe_key[:16] or 'case'}-{timestamp}-{uuid4().hex[:8]}"


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
    ) -> list[OutputT]:
        """Execute ``worker`` over ``cases`` and return outputs in order."""


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
    """Process-pool backend using a top-level worker with bounded submission.

    ``mp_context`` selects the multiprocessing start method (see plan 21
    §16: sequential/process parity must hold under both the Linux default
    context and an explicit ``spawn`` context). ``None`` keeps the
    interpreter's platform default, so existing callers are unaffected.

    Submission is bounded (plan 22 §2.5, ADR 0017): at most ``max_workers``
    futures are in flight and the window is refilled only while the stop
    conditions permit. With ``token`` and ``stop_on_result`` unset (the
    default call), every case executes exactly as before.
    """

    def __init__(
        self,
        max_workers: int = 1,
        mp_context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self.max_workers = max_workers
        self.mp_context = mp_context

    def run(
        self,
        cases: list[InputT],
        worker: Callable[[InputT], OutputT],
        progress: bool = True,
        *,
        stop_on_result: Callable[[OutputT], bool] | None = None,
        stop_reason: str = "operator_request",
        token: CancellationToken | None = None,
        cancelled_result: Callable[[InputT, str], OutputT] | None = None,
    ) -> list[OutputT]:
        """Execute ``worker`` over ``cases`` and return outputs in input order.

        Stop conditions are evaluated parent-side (plan 22 §2.5):

        - ``stop_on_result`` is consulted on each completed output; a truthy
          verdict stops new submissions and drains the tail with
          ``stop_reason`` as the cancellation reason.
        - ``token`` is checked before each new submission and between
          completions; its winning reason drains the tail.

        After a stop condition nothing further is submitted. Futures that
        have not started are cancelled with ``Future.cancel()`` and, together
        with the never-submitted tail, converted to typed cancelled outputs
        via ``cancelled_result(input, reason)``. Already-running workers
        finish and keep their real outputs — this is bounded cooperative
        cancellation, not hard termination. Original order and cardinality
        are preserved regardless of completion order, and the executor is
        shut down and joined before returning.

        ``KeyboardInterrupt`` is never swallowed (plan 22 §2.6): submission
        stops, pending futures are cancelled, the pool is joined,
        proven-not-started cases are journaled through ``cancelled_result``
        (when provided), and the interrupt is re-raised to the caller.
        """
        if (stop_on_result is not None or token is not None) and cancelled_result is None:
            raise ValueError(
                "cancelled_result is required when stop_on_result or token is provided"
            )
        if not cases:
            return []

        outputs: list[OutputT | None] = [None] * len(cases)
        not_started: dict[int, str] = {}
        in_flight: dict[Future[OutputT], int] = {}
        next_index = 0
        stop: str | None = None
        executor = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=self.mp_context)
        bar = tqdm(total=len(cases), desc="Experiments (parallel)") if progress else None
        try:
            while in_flight or next_index < len(cases):
                # Bounded refill: at most ``max_workers`` futures in flight,
                # refilled only while the stop conditions permit.
                while (
                    stop is None
                    and next_index < len(cases)
                    and len(in_flight) < self.max_workers
                    and (token is None or not token.is_cancelled())
                ):
                    future = executor.submit(worker, cases[next_index])
                    in_flight[future] = next_index
                    next_index += 1
                if stop is None and token is not None and token.is_cancelled():
                    stop = token.reason or "operator_request"
                if stop is not None:
                    # Freeze the never-submitted tail, then cancel submitted
                    # futures that have not started; running workers finish.
                    not_started.update(dict.fromkeys(range(next_index, len(cases)), stop))
                    for pending, index in list(in_flight.items()):
                        if pending.cancel():
                            del in_flight[pending]
                            not_started[index] = stop
                    if not in_flight:
                        break
                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    index = in_flight.pop(future)
                    try:
                        completed = future.result()
                    except Exception as exc:
                        logger.error("parallel_case_failed", error=str(exc))
                        raise RuntimeError(f"Parallel case failed: {exc}") from exc
                    outputs[index] = completed
                    if bar is not None:
                        bar.update(1)
                    if stop is None and stop_on_result is not None and stop_on_result(completed):
                        stop = stop_reason
        except KeyboardInterrupt:
            # plan 22 §2.6: stop submission, cancel pending futures, close and
            # join the pool, journal cancellation for the cases proven not to
            # have started, then re-raise. A case that already started keeps
            # its own lifecycle evidence and is never relabelled cancelled.
            executor.shutdown(wait=True, cancel_futures=True)
            for pending, index in in_flight.items():
                if pending.cancel() or pending.cancelled():
                    not_started[index] = KEYBOARD_INTERRUPT_STOP_REASON
            not_started.update(
                dict.fromkeys(range(next_index, len(cases)), KEYBOARD_INTERRUPT_STOP_REASON)
            )
            if cancelled_result is not None:
                for index in sorted(not_started):
                    cancelled_result(cases[index], not_started[index])
            raise
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            if bar is not None:
                bar.close()
        executor.shutdown(wait=True, cancel_futures=True)
        if cancelled_result is not None:
            for index, reason in sorted(not_started.items()):
                outputs[index] = cancelled_result(cases[index], reason)
                if bar is not None:
                    bar.update(1)
        return cast("list[OutputT]", outputs)
