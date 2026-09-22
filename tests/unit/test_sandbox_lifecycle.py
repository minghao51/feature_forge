"""Focused bounded-lifecycle tests for the sandbox worker."""

from __future__ import annotations

import asyncio
import ctypes
import glob
import multiprocessing as mp
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import feature_forge.evaluation.sandbox as sandbox_module
from feature_forge.config import SandboxProfile
from feature_forge.evaluation.sandbox import (
    SandboxedExecutor,
    _WorkerHandle,
    active_worker_handle_count,
)
from feature_forge.exceptions import CodeExecutionError, SandboxTimeoutError

_HANGING_CODE = """
def generate_features(df):
    while True:
        pass
"""
_SUCCESS_CODE = """
def generate_features(df):
    return df.assign(double=df.x * 2)
"""


def _sandbox_temp_files() -> set[str]:
    tmp = Path(tempfile.gettempdir())
    return set(glob.glob(str(tmp / "feature_forge_input_*")) + glob.glob(str(tmp / "ff_sandbox_*")))


def _assert_no_temp_leak(before: set[str], settle: float = 3.0) -> None:
    """Assert no sandbox temp files remain, tolerating concurrent runs.

    The scan sees the whole shared /tmp: a *concurrent* pytest process on the
    same host legitimately creates the pinned prefixes while our own worker
    files are already gone (seen during multi-version parallel validation).
    A genuine leak persists, so poll until the difference is empty or the
    bounded settle window expires; only then fail.
    """
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        if _sandbox_temp_files() <= before:
            return
        time.sleep(0.1)
    leaked = _sandbox_temp_files() - before
    assert not leaked, f"sandbox temp files leaked: {sorted(leaked)}"


def _executor(timeout: float = 0.25) -> SandboxedExecutor:
    return SandboxedExecutor(
        timeout_seconds=timeout,
        profile=SandboxProfile.DEGRADED_DEVELOPMENT,
    )


def _child_pids() -> set[int]:
    return {child.pid for child in mp.active_children() if child.pid is not None}


def _worker_dies_without_posting(*_args: object, **_kwargs: object) -> None:
    """Spawn-picklable worker target that exits without posting a response."""
    return None


def _worker_exits_without_posting(*_args: object, **_kwargs: object) -> None:
    """Spawn-picklable worker target that dies immediately with a known code."""
    os._exit(7)


def _closed_write_end_conn() -> Any:
    """Read end of a one-way pipe whose write end is already closed (EOF).

    Pipe-appropriate replacement for the former queue double: the real worker
    death signal is now an EOF on the parent's read end, so the double is a
    genuine ``Connection`` (``poll`` reports readability, ``recv`` raises
    ``EOFError``) rather than a hand-rolled stub.
    """
    ctx = mp.get_context("spawn")
    read_end, write_end = ctx.Pipe(duplex=False)
    write_end.close()
    return read_end


def test_worker_death_without_response_surfaces_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker dying before posting surfaces a typed sandbox error, never EOFError.

    End-to-end guard for a worker that dies before it can post (see
    _worker_log_event): the child exits without sending, so ``execute()``
    must report through the sandbox error hierarchy the pipeline already
    handles — never a raw EOFError that bypasses it. Whether the read
    observes the closed write-end as EOFError (→ CodeExecutionError) or
    races the deadline (→ SandboxTimeoutError) is host timing under the short
    0.25s timeout here; both are typed and both satisfy the contract. (A
    spawn child that fails to import this test module also lands in the
    timeout branch — still typed; the deterministic closed-write-end pin
    below covers the exact EOF scenario on both read paths, and
    ``test_worker_death_yields_eof_not_timeout`` pins it end-to-end without
    the boot race.)
    """
    monkeypatch.setattr(sandbox_module, "_sandbox_worker_main", _worker_dies_without_posting)
    before_files = _sandbox_temp_files()
    before_children = _child_pids()
    before_handles = active_worker_handle_count()
    with pytest.raises((CodeExecutionError, SandboxTimeoutError)):
        _executor().execute(_SUCCESS_CODE, pd.DataFrame({"x": [1.0]}))
    _assert_no_temp_leak(before_files)
    assert _child_pids() <= before_children
    assert active_worker_handle_count() == before_handles


def test_worker_death_yields_eof_not_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real worker that exits without sending is typed worker death, not a stall.

    With the one-way pipe the parent holds no write handle, so the child's
    exit closes the last writer and the parent's ``poll`` reports readability
    immediately: EOF maps to the typed worker-death error deterministically
    instead of waiting out the deadline and reporting a stall.  The generous
    deadline removes the spawn-boot race, so the EOF branch is the only
    reachable one.
    """
    monkeypatch.setattr(sandbox_module, "_sandbox_worker_main", _worker_dies_without_posting)
    before_files = _sandbox_temp_files()
    before_children = _child_pids()
    before_handles = active_worker_handle_count()
    started = time.monotonic()
    with pytest.raises(CodeExecutionError, match="worker died before reporting a result"):
        _executor(60.0).execute(_SUCCESS_CODE, pd.DataFrame({"x": [1.0]}))
    # EOF arrived from worker death; the 60s deadline was never consumed.
    assert time.monotonic() - started < 30.0
    _assert_no_temp_leak(before_files)
    assert _child_pids() <= before_children
    assert active_worker_handle_count() == before_handles


def test_worker_death_message_includes_abnormal_exit_code() -> None:
    """The typed worker-death message carries an observable abnormal exit code.

    Pins requirement 4's exit-code half deterministically: the process is
    joined before the message is built, so ``_worker_death_message`` (which
    the EOF branch feeds) reports the reaped code rather than racing process
    teardown.
    """
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=_worker_exits_without_posting, daemon=True)
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 7
    response_conn = _closed_write_end_conn()
    try:
        handle = _WorkerHandle(process, response_conn, "", "")
        message = sandbox_module.SandboxedExecutor._worker_death_message(handle)
    finally:
        response_conn.close()
        process.close()
    assert message == "sandbox worker died before reporting a result (worker exit_code=7)"


def test_closed_write_end_response_surfaces_code_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EOF branch maps to CodeExecutionError on both pipe-read paths.

    Deterministic pin for the exact scenario: the pipe write-end closes with
    no item (worker killed mid-flight), so ``poll`` reports readability and
    ``recv`` raises EOFError, which must surface as CodeExecutionError, not
    the raw error.
    """
    monkeypatch.setattr(sandbox_module, "_sandbox_worker_main", _worker_dies_without_posting)
    executor = _executor(5.0)
    response_conn = _closed_write_end_conn()
    try:
        handle = _WorkerHandle(cast("mp.Process", None), response_conn, "", "")
        with pytest.raises(CodeExecutionError, match="worker died before reporting a result"):
            executor._wait_for_response(handle, time.monotonic() + 5)
        with pytest.raises(CodeExecutionError, match="worker died before reporting a result"):
            asyncio.run(executor._poll_response(handle, time.monotonic() + 5))
    finally:
        response_conn.close()


def test_sync_timeout_is_bounded_and_clean() -> None:
    before_files = _sandbox_temp_files()
    before_children = _child_pids()
    before_threads = {thread.ident for thread in threading.enumerate()}
    before_handles = active_worker_handle_count()
    started = time.monotonic()
    with pytest.raises(SandboxTimeoutError):
        _executor().execute(_HANGING_CODE, pd.DataFrame({"x": [1.0]}))
    assert time.monotonic() - started < 2.0
    _assert_no_temp_leak(before_files)
    assert _child_pids() <= before_children
    assert {thread.ident for thread in threading.enumerate()} <= before_threads
    assert active_worker_handle_count() == before_handles


@pytest.mark.asyncio
async def test_async_timeout_is_bounded_and_clean() -> None:
    before_files = _sandbox_temp_files()
    before_children = _child_pids()
    before_threads = {thread.ident for thread in threading.enumerate()}
    before_handles = active_worker_handle_count()
    started = time.monotonic()
    with pytest.raises(SandboxTimeoutError):
        await _executor().execute_async(_HANGING_CODE, pd.DataFrame({"x": [1.0]}))
    assert time.monotonic() - started < 2.0
    await asyncio.sleep(0.05)
    _assert_no_temp_leak(before_files)
    assert _child_pids() <= before_children
    assert {thread.ident for thread in threading.enumerate()} <= before_threads
    assert active_worker_handle_count() == before_handles


@pytest.mark.asyncio
async def test_async_cancellation_cleans_worker_resources() -> None:
    before_files = _sandbox_temp_files()
    before_children = _child_pids()
    before_threads = {thread.ident for thread in threading.enumerate()}
    before_handles = active_worker_handle_count()
    task = asyncio.create_task(
        _executor(5.0).execute_async(_HANGING_CODE, pd.DataFrame({"x": [1.0]}))
    )
    await asyncio.sleep(0.15)  # lets Process.start() and the worker run
    assert not task.done()
    assert active_worker_handle_count() == before_handles + 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)

    _assert_no_temp_leak(before_files)
    assert _child_pids() <= before_children
    assert {thread.ident for thread in threading.enumerate()} <= before_threads
    # The registry drop proves the queue/process handle cleanup path ran.
    assert active_worker_handle_count() == before_handles


@pytest.mark.asyncio
async def test_async_process_start_runs_on_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = mp.get_context("spawn")
    process_type = context.Process
    event_loop_thread = threading.get_ident()
    observed: list[int] = []
    original_start = process_type.start

    def start(process: object, *args: object, **kwargs: object) -> None:
        observed.append(threading.get_ident())
        original_start(process, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(process_type, "start", start)
    result = await _executor(5.0).execute_async(
        _SUCCESS_CODE, pd.DataFrame({"x": [2.0]}), source="lifecycle-test"
    )
    assert result["double"].tolist() == [4.0]
    assert observed == [event_loop_thread]


@pytest.mark.asyncio
async def test_sync_and_async_success_results_match() -> None:
    frame = pd.DataFrame({"x": [1.0, 3.0]})
    sync_result = _executor(5.0).execute(_SUCCESS_CODE, frame)
    async_result = await _executor(5.0).execute_async(_SUCCESS_CODE, frame)
    pd.testing.assert_frame_equal(sync_result, async_result)


class TestBoundedPhaseDiagnostics:
    """Slice B step 1: cheap monotonic phase timing that survives a stall."""

    def test_execution_diagnostics_names_last_completed_phase(self) -> None:
        diagnostics = sandbox_module._ExecutionDiagnostics()
        diagnostics.mark("validate")
        diagnostics.mark("serialize")
        summary = diagnostics.summary()
        assert summary["last_parent_phase"] == "serialize"
        assert {
            "phase_parent_validate_ms",
            "phase_parent_serialize_ms",
            "parent_phase_elapsed_ms",
            "elapsed_ms",
        } <= set(summary)
        assert "last completed parent phase=serialize" in diagnostics.describe()

    def test_worker_progress_reports_not_started_and_in_progress_phase(self) -> None:
        ctx = mp.get_context("spawn")
        progress = sandbox_module._WorkerProgress(
            phase=ctx.Value(ctypes.c_int, sandbox_module._PHASE_NOT_STARTED, lock=False),
            stamps=ctx.RawArray(ctypes.c_double, len(sandbox_module._WORKER_PHASES)),
        )
        assert progress.fields() == {"worker_phase": "not_started"}
        progress.enter("containment_setup")
        progress.enter("input_load")
        progress.enter("landlock")
        progress.enter("code_exec")
        fields = progress.fields()
        assert fields["worker_phase"] == "code_exec"
        assert fields["worker_phase_elapsed_ms"] >= 0
        assert fields["phase_worker_containment_setup_ms"] >= 0
        assert fields["phase_worker_input_load_ms"] >= 0
        assert fields["phase_worker_landlock_ms"] >= 0
        # Bootstrap precedes the worker's entry stamp, so the marker never
        # derives it; the worker reports it against the parent's spawn stamp.
        assert "phase_worker_bootstrap_ms" not in fields

    def test_timeout_error_carries_last_completed_phase(self) -> None:
        diagnostics = sandbox_module._ExecutionDiagnostics()
        diagnostics.mark("spawn")
        error = _executor(5.0)._timeout_error(diagnostics)
        assert isinstance(error, SandboxTimeoutError)
        assert "last completed parent phase=spawn" in str(error)

    def test_pipe_send_and_response_wait_phases_are_recorded(self) -> None:
        """The pipe swap keeps step-1 timing: real send + poll/recv window.

        ``phase_parent_response_wait_ms`` brackets the deadline-aware
        poll/recv window.  ``phase_worker_result_send_ms`` is derived from the
        worker's shared marker because it cannot travel inside the very
        message it measures; on a host where the marker is unavailable that
        field degrades away by design, so it is asserted only when present.
        """
        executor = _executor(5.0)
        diagnostics = sandbox_module._ExecutionDiagnostics()
        result = executor._execute_in_worker(
            _SUCCESS_CODE,
            pd.DataFrame({"x": [1.0]}),
            deadline=time.monotonic() + 5.0,
            diagnostics=diagnostics,
        )
        assert result["double"].tolist() == [2.0]
        assert "phase_parent_response_wait_ms" in diagnostics.phases_ms
        if diagnostics.worker is not None:
            assert "phase_worker_result_send_ms" in diagnostics.reported_phases_ms
            assert diagnostics.reported_phases_ms["phase_worker_result_send_ms"] >= 0.0

    def test_timeout_message_reports_phase_and_worker_progress(self) -> None:
        with pytest.raises(SandboxTimeoutError) as excinfo:
            _executor(0.25).execute(_HANGING_CODE, pd.DataFrame({"x": [1.0]}))
        message = str(excinfo.value)
        assert "last completed parent phase=" in message
        assert "completed[" in message
        # The marker fields are intentionally not asserted here: shared-memory
        # creation can legitimately fail on a host (the executor degrades
        # cleanly by design), so the pure message fallback is pinned by
        # ``test_timeout_error_carries_last_completed_phase`` and the
        # zero-marker path by ``test_timeout_without_worker_marker_degrades``.

    def test_timeout_without_worker_marker_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A host where shared-memory creation fails still times out cleanly.

        Pins the degraded fallback: no marker attached, so the timeout message
        carries only the parent-side phase trail and the error stays exactly
        ``SandboxTimeoutError`` (no diagnostics wrapper or machinery leak).
        """
        monkeypatch.setattr(
            sandbox_module.SandboxedExecutor,
            "_new_worker_progress",
            staticmethod(lambda ctx: None),
        )
        with pytest.raises(SandboxTimeoutError) as excinfo:
            _executor(0.25).execute(_HANGING_CODE, pd.DataFrame({"x": [1.0]}))
        message = str(excinfo.value)
        assert type(excinfo.value) is SandboxTimeoutError
        assert "last completed parent phase=" in message
        assert "worker phase=" not in message
