"""Bounded process scheduling and interruption hygiene (plan 22 PR 3, ADR 0017).

Covers the process runtime slice of the failure/cancellation contract: the
bounded submission window (at most ``max_workers`` futures in flight,
refilled only while the stop conditions permit), parent-side stop conditions
(``stop_on_result`` verdict plus ``CancellationToken``), pending-future
cancellation with typed cancelled tails, order/cardinality preservation,
KeyboardInterrupt re-raise with redacted not-started journals, executor
shutdown without worker leaks under the Linux default and explicit ``spawn``
contexts, and ``CaseComputationInput`` serialization guarantees.

Real-pool tests use top-level pickleable workers, fresh temporary roots,
``multiprocessing.Manager`` events for deterministic synchronization (no
timing sleeps), and bounded waits everywhere. The pending-future
cancellation test (plan 22 §6 test 6) drives the adapter with a scripted
executor handing out real ``concurrent.futures.Future`` objects: with a
bounded window the executor's feeder thread marks futures running before a
worker can lag, so a parent-observable queued-not-started future is only
deterministically constructible at the logic level.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import pickle
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from feature_forge import ExperimentalPlatform
from feature_forge import platform as platform_module
from feature_forge.contracts import CaseExecutionPlan, Layer
from feature_forge.contracts.orchestration import FailureRecord
from feature_forge.contracts.stages import FailureClass, RunState
from feature_forge.experiment import execution as execution_module
from feature_forge.experiment.execution import (
    KEYBOARD_INTERRUPT_STOP_REASON,
    CancellationToken,
    CaseComputationInput,
    ExperimentCase,
    ExperimentResult,
    ProcessPoolExecutionAdapter,
    SequentialExecutionAdapter,
    cancelled_failure_record,
)
from feature_forge.experiment.lifecycle import LocalRunRepository

# The Linux-default ``fork`` context is a mandated plan-22 test dimension.
# Under pytest (coverage plugin) the parent is multi-threaded, so Python 3.12+
# emits a DeprecationWarning per fork(); expected environmental noise here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*multi-threaded.*fork:DeprecationWarning"
)

# Every worker-side wait is bounded; tests never hang on a lost gate.
GATE_TIMEOUT_SECONDS = 30.0
_JOIN_TIMEOUT_SECONDS = 60.0

_PASS = "pass"
_FAIL = "fail"
_BLOCK = "block"
_WAIT_STARTED_FAIL = "wait_started_fail"
_RAISE_INTERRUPT = "raise_interrupt"


# ── Top-level pickleable workers and helpers ───────────────────


@dataclass
class _GateSpec:
    """Picklable worker input carrying optional manager-event gates."""

    case: ExperimentCase
    mode: str = _PASS
    started: Any = None  # multiprocessing.Manager().Event() proxy
    release: Any = None  # multiprocessing.Manager().Event() proxy


def _succeeded(case: ExperimentCase) -> ExperimentResult:
    return ExperimentResult(
        dataset=case.dataset,
        method=case.method,
        model=case.model,
        seed=case.seed,
        cv_score=0.9,
        gain=0.1,
        baseline_score=0.8,
        num_features_generated=2,
    )


def _failed(case: ExperimentCase, message: str = "terminal boom") -> ExperimentResult:
    return ExperimentResult(
        dataset=case.dataset,
        method=case.method,
        model=case.model,
        seed=case.seed,
        error=message,
        failure=FailureRecord(
            failure_class=FailureClass.DETERMINISTIC,
            error_type="RuntimeError",
            message=message,
            stage="case",
        ),
    )


def _cancelled_row(case: ExperimentCase, reason: str) -> ExperimentResult:
    failure = cancelled_failure_record(case_fingerprint=f"casekey-{case.method}", reason=reason)
    return ExperimentResult(
        dataset=case.dataset,
        method=case.method,
        model=case.model,
        seed=case.seed,
        run_id=f"attempt-{case.method}",
        case_fingerprint=f"casekey-{case.method}",
        error=failure.message,
        failure=failure,
        state=RunState.CANCELLED,
    )


def _plain_worker(case: ExperimentCase) -> ExperimentResult:
    """Top-level worker required for process-pool serialization."""
    return _succeeded(case)


def _mixed_worker(case: ExperimentCase) -> ExperimentResult:
    """Succeeds everywhere except ``m2``, which fails terminally."""
    return _failed(case) if case.method == "m2" else _succeeded(case)


def _gate_worker(spec: _GateSpec) -> ExperimentResult:
    """Top-level worker with deterministic event gates (no timing sleeps)."""
    case = spec.case
    if spec.mode == _RAISE_INTERRUPT:
        raise KeyboardInterrupt("worker interrupted")
    if spec.mode == _WAIT_STARTED_FAIL:
        started = True
        if spec.started is not None:
            started = bool(spec.started.wait(GATE_TIMEOUT_SECONDS))
        if not started:
            return _failed(case, "gate timeout: peer never started")
        return _failed(case)
    if spec.mode == _BLOCK:
        if spec.started is not None:
            spec.started.set()
        if spec.release is not None and not spec.release.wait(GATE_TIMEOUT_SECONDS):
            return _failed(case, "gate timeout")
    if spec.mode == _FAIL:
        return _failed(case)
    return _succeeded(case)


def _cases(count: int) -> list[ExperimentCase]:
    return [
        ExperimentCase(dataset="d", method=f"m{index}", model="random_forest", seed=42)
        for index in range(1, count + 1)
    ]


def _active_children() -> set[Any]:
    return set(multiprocessing.active_children())


class _ScriptedPoolExecutor:
    """Deterministic ``ProcessPoolExecutor`` stand-in with scripted futures.

    Hands out real ``concurrent.futures.Future`` objects so the adapter's
    ``concurrent.futures.wait`` bookkeeping stays genuine; only the pool is
    simulated. Class-level state is reset by each test (single-threaded).
    """

    futures: ClassVar[list[Future[Any]]] = []
    instances: ClassVar[list[_ScriptedPoolExecutor]] = []

    def __init__(self, max_workers: int | None = None, mp_context: Any = None) -> None:
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.submit_count = 0
        self.shutdown_calls: list[tuple[bool, bool]] = []
        type(self).instances.append(self)

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        self.submit_count += 1
        return type(self).futures.pop(0)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class _RunningFuture(Future[ExperimentResult]):
    """A future already picked up by a worker: cancel fails, then it finishes."""

    def __init__(self, result: ExperimentResult) -> None:
        super().__init__()
        self._result_value = result
        self.cancel_attempts = 0
        self.set_running_or_notify_cancel()

    def cancel(self) -> bool:
        self.cancel_attempts += 1
        if not self.done():
            self.set_result(self._result_value)
        return False


class _InterruptingExecutor:
    """HamiltonLayerExecutor stand-in raising KeyboardInterrupt on ``m2``."""

    def __init__(self) -> None:
        self.planned: list[str] = []
        self.executed: list[str] = []

    def plan_case(self, case: ExperimentCase) -> CaseExecutionPlan:
        self.planned.append(case.method)
        return CaseExecutionPlan(
            case_key=f"casekey-{case.method}",
            attempt_id=f"attempt-{case.method}",
            dataset=case.dataset,
            method=case.method,
            model=case.model,
            unresolved_layers=list(Layer),
        )

    def execute_case(self, case: ExperimentCase) -> ExperimentResult:
        self.executed.append(case.method)
        if case.method == "m2":
            raise KeyboardInterrupt("simulated interrupt during case")
        result = _succeeded(case)
        result.run_id = case.attempt_id
        result.case_fingerprint = case.case_key
        return result


# ── Continue policy parity across adapters (plan 22 §6 test 1) ──


class TestContinuePolicyAdapterParity:
    def test_success_failure_success_execute_in_both_adapters(self) -> None:
        """Default (no stop hooks) calls keep today's semantics: all cases run."""
        cases = _cases(3)

        sequential = SequentialExecutionAdapter().run(cases, _mixed_worker, progress=False)
        process = ProcessPoolExecutionAdapter(max_workers=2).run(
            cases, _mixed_worker, progress=False
        )

        expected = [("m1", RunState.SUCCEEDED), ("m2", RunState.FAILED), ("m3", RunState.SUCCEEDED)]
        assert [(row.method, row.resolved_state) for row in sequential] == expected
        assert [(row.method, row.resolved_state) for row in process] == expected
        assert [row.cv_score for row in process] == [0.9, None, 0.9]


# ── Bounded fail-fast window (plan 22 §6 test 5) ────────────────


class TestFailFastBoundedWindow:
    @pytest.mark.parametrize("context_name", [None, "spawn"], ids=["linux-default", "spawn"])
    def test_bounded_in_flight_completes_then_cancelled_tail(
        self, context_name: str | None
    ) -> None:
        """One in-flight success completes after the failure; the tail cancels.

        Deterministic choreography: ``m1`` blocks on a manager event while
        ``m2`` waits for ``m1``'s start signal before failing, so at stop time
        ``m1`` is provably running (cancel must fail) and ``m3..m5`` are never
        submitted. The ``stop_on_result`` callback releases ``m1``.
        """
        children_before = _active_children()
        manager = multiprocessing.Manager()
        try:
            started = manager.Event()
            release = manager.Event()
            specs = [
                _GateSpec(case=_cases(5)[0], mode=_BLOCK, started=started, release=release),
                _GateSpec(case=_cases(5)[1], mode=_WAIT_STARTED_FAIL, started=started),
                *(_GateSpec(case=case) for case in _cases(5)[2:]),
            ]
            drained: list[tuple[str, str]] = []

            def factory(spec: _GateSpec, reason: str) -> ExperimentResult:
                drained.append((spec.case.method, reason))
                return _cancelled_row(spec.case, reason)

            def stop_on_result(result: ExperimentResult) -> bool:
                if result.resolved_state is RunState.FAILED:
                    release.set()
                    return True
                return False

            context = multiprocessing.get_context(context_name) if context_name else None
            adapter = ProcessPoolExecutionAdapter(max_workers=2, mp_context=context)
            results = adapter.run(
                specs,
                _gate_worker,
                progress=False,
                stop_on_result=stop_on_result,
                stop_reason="fail_fast",
                cancelled_result=factory,
            )
        finally:
            manager.shutdown()

        # Exact order and cardinality: one row per requested case, matrix order.
        assert [(row.method, row.resolved_state) for row in results] == [
            ("m1", RunState.SUCCEEDED),
            ("m2", RunState.FAILED),
            ("m3", RunState.CANCELLED),
            ("m4", RunState.CANCELLED),
            ("m5", RunState.CANCELLED),
        ]
        # The in-flight case kept its REAL result, completing out of order.
        assert results[0].cv_score == pytest.approx(0.9)
        assert results[1].error == "terminal boom"
        for row in results[2:]:
            assert row.state is RunState.CANCELLED
            assert row.stages == []
            assert row.cv_score is None
            assert row.error is not None
            assert row.failure is not None
            assert row.failure.failure_class is FailureClass.CANCELLED
            assert "(fail_fast)" in row.error
        assert drained == [("m3", "fail_fast"), ("m4", "fail_fast"), ("m5", "fail_fast")]
        # No worker leak after the fail-fast run.
        assert _active_children() == children_before


# ── Pending-future cancellation (plan 22 §6 test 6) ─────────────


class TestPendingFutureCancellation:
    def test_queued_future_cancelled_before_start_becomes_typed_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Queued-not-started work cancels cleanly; running work stays real.

        Deterministic scripted futures: ``f1`` fails (stop trigger), ``f2`` is
        running (cancel attempt fails, then it finishes), ``f3`` is queued
        (cancel succeeds). ``m4`` is never submitted.
        """
        cases = _cases(4)
        completed = Future[ExperimentResult]()
        completed.set_result(_failed(cases[0]))
        running = _RunningFuture(_succeeded(cases[1]))
        queued = Future[ExperimentResult]()
        _ScriptedPoolExecutor.futures = [completed, running, queued]
        _ScriptedPoolExecutor.instances = []
        monkeypatch.setattr(execution_module, "ProcessPoolExecutor", _ScriptedPoolExecutor)

        drained: list[tuple[str, str]] = []

        def factory(case: ExperimentCase, reason: str) -> ExperimentResult:
            drained.append((case.method, reason))
            return _cancelled_row(case, reason)

        adapter = ProcessPoolExecutionAdapter(max_workers=3)
        results = adapter.run(
            cases,
            _plain_worker,
            progress=False,
            stop_on_result=lambda result: result.resolved_state is RunState.FAILED,
            stop_reason="fail_fast",
            cancelled_result=factory,
        )

        executor = _ScriptedPoolExecutor.instances[0]
        assert executor.submit_count == 3  # m4 never submitted after the stop
        assert executor.shutdown_calls == [(True, True)]
        # The queued future was cancelled before start; the running one was not.
        assert queued.cancelled() is True
        assert running.cancel_attempts == 1
        assert running.cancelled() is False
        assert [(row.method, row.resolved_state) for row in results] == [
            ("m1", RunState.FAILED),
            ("m2", RunState.SUCCEEDED),
            ("m3", RunState.CANCELLED),
            ("m4", RunState.CANCELLED),
        ]
        assert results[2].cv_score is None
        assert results[2].stages == []
        assert drained == [("m3", "fail_fast"), ("m4", "fail_fast")]

    def test_pre_cancelled_token_submits_nothing_and_drains_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ScriptedPoolExecutor.futures = []
        _ScriptedPoolExecutor.instances = []
        monkeypatch.setattr(execution_module, "ProcessPoolExecutor", _ScriptedPoolExecutor)

        drained: list[tuple[str, str]] = []

        def factory(case: ExperimentCase, reason: str) -> ExperimentResult:
            drained.append((case.method, reason))
            return _cancelled_row(case, reason)

        adapter = ProcessPoolExecutionAdapter(max_workers=2)
        results = adapter.run(
            _cases(3),
            _plain_worker,
            progress=False,
            token=CancellationToken.cancelled("stop_early"),
            cancelled_result=factory,
        )

        executor = _ScriptedPoolExecutor.instances[0]
        assert executor.submit_count == 0
        assert executor.shutdown_calls == [(True, True)]
        assert [(row.method, row.resolved_state) for row in results] == [
            ("m1", RunState.CANCELLED),
            ("m2", RunState.CANCELLED),
            ("m3", RunState.CANCELLED),
        ]
        assert all(row.error is not None and "(stop_early)" in row.error for row in results)
        assert drained == [("m1", "stop_early"), ("m2", "stop_early"), ("m3", "stop_early")]

    def test_stop_conditions_require_cancelled_result_factory(self) -> None:
        with pytest.raises(ValueError, match="cancelled_result"):
            ProcessPoolExecutionAdapter(max_workers=1).run(
                _cases(1),
                _plain_worker,
                progress=False,
                token=CancellationToken(),
            )


# ── Token cancellation mid-run (plan 22 §6 test 4, process path) ──


class TestTokenMidRunCancellation:
    @pytest.mark.parametrize("context_name", [None, "spawn"], ids=["linux-default", "spawn"])
    def test_token_cancelled_during_case_finishes_it_then_cancels_tail(
        self, context_name: str | None
    ) -> None:
        """A token cancelled while ``m1`` runs lets it finish truthfully.

        Deterministic choreography (plan 22 §6 test 4, process path): a
        controller thread waits for ``m1``'s started gate, cancels the
        token, then releases the gate — so cancellation provably arrives
        mid-case. ``m1`` keeps its real success; ``m2``/``m3`` never
        submit and drain as typed cancelled rows with the token's reason.
        """
        children_before = _active_children()
        manager = multiprocessing.Manager()
        token = CancellationToken()
        try:
            started = manager.Event()
            release = manager.Event()
            specs = [
                _GateSpec(case=_cases(3)[0], mode=_BLOCK, started=started, release=release),
                *(_GateSpec(case=case) for case in _cases(3)[1:]),
            ]
            drained: list[tuple[str, str]] = []

            def factory(spec: _GateSpec, reason: str) -> ExperimentResult:
                drained.append((spec.case.method, reason))
                return _cancelled_row(spec.case, reason)

            def controller() -> None:
                if started.wait(GATE_TIMEOUT_SECONDS):
                    token.cancel("operator_request")
                    release.set()

            thread = threading.Thread(target=controller, daemon=True)
            thread.start()
            context = multiprocessing.get_context(context_name) if context_name else None
            adapter = ProcessPoolExecutionAdapter(max_workers=1, mp_context=context)
            results = adapter.run(
                specs,
                _gate_worker,
                progress=False,
                token=token,
                cancelled_result=factory,
            )
            thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
        finally:
            manager.shutdown()

        assert [(row.method, row.resolved_state) for row in results] == [
            ("m1", RunState.SUCCEEDED),
            ("m2", RunState.CANCELLED),
            ("m3", RunState.CANCELLED),
        ]
        # The in-flight case kept its real result despite the cancellation.
        assert results[0].cv_score == pytest.approx(0.9)
        for row in results[1:]:
            assert row.state is RunState.CANCELLED
            assert row.stages == []
            assert row.cv_score is None
            assert row.failure is not None
            assert row.failure.failure_class is FailureClass.CANCELLED
            assert "(operator_request)" in (row.error or "")
        assert drained == [
            ("m2", "operator_request"),
            ("m3", "operator_request"),
        ]
        # No worker leak after the token-cancelled run.
        assert _active_children() == children_before


# ── KeyboardInterrupt hygiene (plan 22 §6 test 7) ────────────────


class TestKeyboardInterruptHygiene:
    def test_interrupt_during_collection_reraises_journals_and_joins_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KI while collecting: cleanup runs, not-started tail journals, KI re-raised.

        Deterministic injection: the first ``wait()`` call — the canonical
        SIGINT point — raises after both fast workers were submitted, so
        nothing blocks. ``m3``/``m4`` are provably never submitted.
        ``m1`` always starts; ``m2`` has usually started too, but under heavy
        load the pool may not have picked it up yet — in that case the
        interrupt handler's cancel succeeds and it is truthfully journaled
        as not-started, so its fate is asserted as a two-way complement.
        """
        real_wait = concurrent.futures.wait

        def interrupting_wait(fs: Any, **kwargs: Any) -> Any:
            if not interrupting_wait.called:  # type: ignore[attr-defined]
                interrupting_wait.called = True  # type: ignore[attr-defined]
                raise KeyboardInterrupt("injected interrupt during wait")
            return real_wait(fs, **kwargs)

        interrupting_wait.called = False  # type: ignore[attr-defined]
        monkeypatch.setattr(execution_module, "wait", interrupting_wait)

        children_before = _active_children()
        drained: list[tuple[str, str]] = []

        def factory(case: ExperimentCase, reason: str) -> ExperimentResult:
            drained.append((case.method, reason))
            return _cancelled_row(case, reason)

        with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
            ProcessPoolExecutionAdapter(max_workers=2).run(
                _cases(4),
                _plain_worker,
                progress=False,
                cancelled_result=factory,
            )

        # Only never-started cases journal. m3/m4 were never submitted; m2
        # either started (keeps its own lifecycle evidence) or was still
        # queued and is cancelled-before-start — both outcomes are truthful.
        assert [item for item in drained if item[0] in {"m3", "m4"}] == [
            ("m3", KEYBOARD_INTERRUPT_STOP_REASON),
            ("m4", KEYBOARD_INTERRUPT_STOP_REASON),
        ]
        assert all(reason == KEYBOARD_INTERRUPT_STOP_REASON for _, reason in drained)
        assert {method for method, _ in drained} <= {"m2", "m3", "m4"}
        # No worker leak after the interrupted run.
        assert _active_children() == children_before

    def test_worker_raised_interrupt_propagates_as_keyboard_interrupt(self) -> None:
        """A worker's KeyboardInterrupt reaches the caller as KI, not RuntimeError."""
        children_before = _active_children()
        cases = _cases(2)
        specs = [_GateSpec(case=cases[0]), _GateSpec(case=cases[1], mode=_RAISE_INTERRUPT)]
        drained: list[tuple[str, str]] = []

        def factory(spec: _GateSpec, reason: str) -> ExperimentResult:
            drained.append((spec.case.method, reason))
            return _cancelled_row(spec.case, reason)

        with pytest.raises(KeyboardInterrupt, match="worker interrupted"):
            ProcessPoolExecutionAdapter(max_workers=2).run(
                specs,
                _gate_worker,
                progress=False,
                cancelled_result=factory,
            )

        # Both cases were in flight when the interrupt surfaced: neither is
        # proven not-started, so nothing journals.
        assert drained == []
        assert _active_children() == children_before

    def test_sequential_interrupt_journals_remaining_and_reraises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sequential path: journal the never-started tail, then re-raise KI."""
        executor = _InterruptingExecutor()
        monkeypatch.setattr(platform_module, "HamiltonLayerExecutor", lambda **_: executor)
        artifact_root = tmp_path / "artifacts"
        platform = ExperimentalPlatform(
            config={
                "tracker": {"backend": "none"},
                "dataflow": {
                    "artifact_root": artifact_root,
                    "cache": {"path": tmp_path / "cache"},
                },
            }
        )

        with pytest.raises(KeyboardInterrupt, match="simulated interrupt"):
            platform.run(
                datasets=["d"],
                methods=["m1", "m2", "m3"],
                progress=False,
            )

        # m2 started (interrupted in flight) and is left alone, not cancelled.
        assert executor.executed == ["m1", "m2"]
        repository = LocalRunRepository(artifact_root / "control" / "lifecycle")
        runs_root = artifact_root / "control" / "lifecycle" / "runs"
        assert {path.name for path in runs_root.iterdir()} == {"attempt-m3"}
        events = repository.load_events("attempt-m3")
        assert len(events) == 1
        assert events[0].event_type == "case_cancelled"
        assert events[0].failure is not None
        assert f"({KEYBOARD_INTERRUPT_STOP_REASON})" in events[0].failure.message


# ── Platform process wiring: lifecycle, tracker, identity ──────


class TestPlatformProcessLifecycleAndTracker:
    """Plan 22 §6 tests 8 (never-started slice) and 9 on the process path."""

    @staticmethod
    def _planning_executor() -> MagicMock:
        executor = MagicMock()
        executor.plan_case.side_effect = lambda case: CaseExecutionPlan(
            case_key=f"casekey-{case.method}",
            attempt_id=f"attempt-{case.method}",
            dataset=case.dataset,
            method=case.method,
            model=case.model,
            unresolved_layers=list(Layer),
        )
        return executor

    def test_pre_cancelled_process_run_journals_and_skips_tracker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = self._planning_executor()
        monkeypatch.setattr(platform_module, "HamiltonLayerExecutor", lambda **_: executor)
        artifact_root = tmp_path / "artifacts"
        tracker = MagicMock()
        platform = ExperimentalPlatform(
            config={
                "tracker": {"backend": "none"},
                "dataflow": {
                    "artifact_root": artifact_root,
                    "cache": {"path": tmp_path / "cache"},
                },
            }
        )
        children_before = _active_children()

        results = platform.run(
            datasets=["d"],
            methods=["m1", "m2", "m3"],
            progress=False,
            parallel=True,
            max_workers=2,
            tracker=tracker,
            cancellation_token=CancellationToken.cancelled("stop_early"),
        )

        # Payload identity is reused: no extra planning calls for the drain.
        assert executor.plan_case.call_count == 3
        assert [row["state"] for row in results] == ["cancelled"] * 3
        assert [row["run_id"] for row in results] == [
            "attempt-m1",
            "attempt-m2",
            "attempt-m3",
        ]
        # Exactly-once tracker rule: unstarted cases initialize no tracker run.
        tracker.init_run.assert_not_called()
        tracker.finish.assert_not_called()

        # One redacted case_cancelled event per unstarted process case.
        lifecycle_root = artifact_root / "control" / "lifecycle"
        repository = LocalRunRepository(lifecycle_root)
        runs_root = lifecycle_root / "runs"
        assert {path.name for path in runs_root.iterdir()} == {
            "attempt-m1",
            "attempt-m2",
            "attempt-m3",
        }
        for row in results:
            events = repository.load_events(row["run_id"])
            assert len(events) == 1
            event = events[0]
            assert event.event_type == "case_cancelled"
            assert event.state is RunState.CANCELLED
            assert event.run_id == row["run_id"]
            assert event.case_id == row["case_fingerprint"]
            assert event.fingerprint == row["case_fingerprint"]
            assert event.failure is not None
            assert event.failure.failure_class is FailureClass.CANCELLED
            assert event.failure.error_type == "CaseCancelled"
            # Allowed fields only: no manifest payload, stage, or free text.
            assert event.manifest_ref is None
            assert event.stage is None
            assert event.details == {}
            text = (runs_root / row["run_id"] / "events.jsonl").read_text(encoding="utf-8")
            assert "api_key" not in text
            assert "prompt" not in text.lower()
        # A pre-cancelled token spawns zero worker processes.
        assert _active_children() == children_before


# ── Payload serialization guard (plan 22 §6 test 11) ────────────


class TestPayloadSerialization:
    def test_platform_payloads_pickle_round_trip_without_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataclasses import fields as dataclass_fields

        executor = TestPlatformProcessLifecycleAndTracker._planning_executor()
        monkeypatch.setattr(platform_module, "HamiltonLayerExecutor", lambda **_: executor)
        captured: list[CaseComputationInput] = []
        pool = MagicMock()

        def capture_run(
            payloads: list[CaseComputationInput], worker: Any, progress: bool = True, **_: Any
        ) -> list[ExperimentResult]:
            captured.extend(payloads)
            return [_succeeded(payload.case) for payload in payloads]

        pool.run.side_effect = capture_run
        monkeypatch.setattr(platform_module, "ProcessPoolExecutionAdapter", lambda **_: pool)
        platform = ExperimentalPlatform(
            config={
                "tracker": {"backend": "none"},
                "dataflow": {
                    "artifact_root": tmp_path / "artifacts",
                    "cache": {"path": tmp_path / "cache"},
                },
            }
        )

        results = platform.run(
            datasets=["d"],
            methods=["m1", "m2"],
            progress=False,
            parallel=True,
        )

        assert [row["state"] for row in results] == ["succeeded", "succeeded"]
        assert len(captured) == 2
        assert {field.name for field in dataclass_fields(CaseComputationInput)} == {
            "case",
            "settings_data",
            "dataset_overrides",
            "method_overrides",
            "model_overrides",
            "metric_overrides",
        }
        for payload in captured:
            clone = pickle.loads(pickle.dumps(payload))
            assert clone == payload
            assert not any("token" in name or "cancel" in name for name in vars(clone).keys())
            assert clone.case.case_key is not None
            assert clone.case.attempt_id is not None


# ── Worker-leak sweep (plan 22 §6) ──────────────────────────────


class TestWorkerLeakSweep:
    def test_normal_run_joins_pool_and_preserves_order(self) -> None:
        """Default call: every case executes, rows come back in matrix order."""
        children_before = _active_children()

        results = ProcessPoolExecutionAdapter(max_workers=2).run(
            _cases(4), _plain_worker, progress=False
        )

        assert [row.method for row in results] == ["m1", "m2", "m3", "m4"]
        assert all(row.resolved_state is RunState.SUCCEEDED for row in results)
        assert _active_children() == children_before
