"""Focused bounded-lifecycle tests for the sandbox worker."""

from __future__ import annotations

import asyncio
import glob
import multiprocessing as mp
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

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


class _ClosedWriteEndQueue:
    """Queue double for a worker whose write-end closed with no item posted."""

    def get(self, timeout: float | None = None) -> object:
        raise EOFError

    def get_nowait(self) -> object:
        raise EOFError


def test_worker_death_without_response_surfaces_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker dying before posting surfaces a typed sandbox error, never EOFError.

    End-to-end guard for the RLIMIT_AS feeder-thread failure mode (see
    _worker_log_event): the child exits without posting, so ``execute()``
    must report through the sandbox error hierarchy the pipeline already
    handles — never a raw EOFError that bypasses it. Whether the queue read
    observes the closed write-end as EOFError (→ CodeExecutionError) or
    races the deadline (→ SandboxTimeoutError) is host timing; both are
    typed and both satisfy the contract. (A spawn child that fails to import
    this test module also lands in the timeout branch — still typed; the
    deterministic closed-write-end pin below covers the exact EOF scenario
    on both read paths.)
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


def test_closed_write_end_response_surfaces_code_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EOF branch maps to CodeExecutionError on both queue-read paths.

    Deterministic pin for the exact fix-batch scenario: the queue write-end
    closes with no item (worker killed mid-flight), so the read raises
    EOFError and must surface as CodeExecutionError, not the raw error.
    """
    monkeypatch.setattr(sandbox_module, "_sandbox_worker_main", _worker_dies_without_posting)
    executor = _executor(5.0)
    handle = _WorkerHandle(cast("mp.Process", None), _ClosedWriteEndQueue(), "", "")
    with pytest.raises(CodeExecutionError, match="worker died before reporting a result"):
        executor._wait_for_response(handle, time.monotonic() + 5)
    with pytest.raises(CodeExecutionError, match="worker died before reporting a result"):
        asyncio.run(executor._poll_response(handle, time.monotonic() + 5))


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
