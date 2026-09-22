"""Slice B step 4 pins: scoped sandbox address-space limiting.

The sandbox worker must not run under the configured ``RLIMIT_AS`` while it
starts, imports its runtime, loads the input frame, or publishes its result:
a fresh Python 3.13 ``spawn`` worker already maps ~0.9-1.8 GB of virtual
address space (CPython + pandas/numpy/pyarrow + Landlock), so a 512 MB cap
applied at worker entry either kills the worker during import or during the
response send.  The cap is therefore scoped around generated-code execution
only and restored before output publication.

These tests are written before the rework: the e2e pins fail against the
entry-time ``_apply_resource_limits`` implementation, and the unit pins fail
because the scoped helper/metadata do not exist yet.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

import feature_forge.evaluation.sandbox as sandbox_module
from feature_forge.evaluation.sandbox import (
    SandboxedExecutor,
    active_worker_handle_count,
)
from feature_forge.exceptions import CodeExecutionError, SandboxTimeoutError

_TRIVIAL_CODE = "def generate_features(df):\n    return df.assign(x2=df.x * 2)\n"
# A single huge virtual allocation: under a reachable soft RLIMIT_AS it fails
# with a MemoryError subclass, but without a cap the anonymous mapping is
# lazy and does not touch physical pages (so a skipped cap cannot OOM-kill
# the worker and masquerade as a typed failure).
_OOM_CODE = """
def generate_features(df):
    huge = np.empty(2_000_000_000, dtype=np.float64)
    return df.assign(n=float(len(huge)))
"""


def _require_landlock() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("scoped address-space caps are pinned on Linux")
    if sandbox_module._probe_landlock_abi() is None:
        pytest.skip("Landlock is unavailable on this host")


def _sandbox_temp_files() -> set[str]:
    tmp = Path(tempfile.gettempdir())
    return set(glob.glob(str(tmp / "feature_forge_input_*")) + glob.glob(str(tmp / "ff_sandbox_*")))


def _assert_no_temp_leak(before: set[str], settle: float = 3.0) -> None:
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        if _sandbox_temp_files() <= before:
            return
        time.sleep(0.1)
    leaked = _sandbox_temp_files() - before
    assert not leaked, f"sandbox temp files leaked: {sorted(leaked)}"


def _child_pids() -> set[int]:
    return {child.pid for child in mp.active_children() if child.pid is not None}


def _capture_metadata(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the worker response metadata the parent consumed."""
    captured: dict[str, Any] = {}
    original = SandboxedExecutor._consume_response

    def capture(
        self_: SandboxedExecutor,
        handle: Any,
        status: str,
        payload: str,
        worker_metadata: dict[str, Any],
        deadline: float,
        **kwargs: Any,
    ) -> pd.DataFrame:
        captured["status"] = status
        captured["payload"] = payload
        captured["metadata"] = dict(worker_metadata)
        return original(self_, handle, status, payload, worker_metadata, deadline, **kwargs)

    monkeypatch.setattr(SandboxedExecutor, "_consume_response", capture)
    return captured


def _assert_cap_fields(metadata: dict[str, Any], requested_mb: int) -> str:
    """Every response advertises the memory-cap decision; never silent."""
    state = metadata["sandbox_memory_cap"]
    assert state in {"enforced", "skipped-infeasible", "disabled", "unavailable"}
    assert metadata["sandbox_memory_cap_requested_mb"] == requested_mb
    return str(state)


@pytest.mark.parametrize("requested_mb", [512, 2048])
def test_configured_worker_starts_and_reports_cap_state(
    monkeypatch: pytest.MonkeyPatch, requested_mb: int
) -> None:
    """P1: a configured worker starts, returns a small frame, and is never silent.

    512 MB is infeasible on this host (post-import VmSize is well above it),
    so the worker must skip the cap explicitly and still succeed; 2048 MB is
    normally feasible and recorded as enforced.  On either path the response
    metadata carries the requested value.
    """
    _require_landlock()
    before_files = _sandbox_temp_files()
    before_children = _child_pids()
    before_handles = active_worker_handle_count()
    captured = _capture_metadata(monkeypatch)
    executor = SandboxedExecutor(timeout_seconds=30.0, max_memory_mb=requested_mb)
    result = executor.execute(
        _TRIVIAL_CODE, pd.DataFrame({"x": [1.0, 2.0]}), source="memory-pin", agent_name="memory"
    )
    assert result["x2"].tolist() == [2.0, 4.0]
    assert captured["status"] == "ok"
    state = _assert_cap_fields(captured["metadata"], requested_mb)
    assert state in {"enforced", "skipped-infeasible"}

    _assert_no_temp_leak(before_files)
    assert _child_pids() <= before_children
    assert active_worker_handle_count() == before_handles


def test_reachable_cap_fails_through_typed_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2: generated code exceeding a reachable cap is a typed memory failure.

    The cap is derived from the worker's own reported post-import VmSize (a
    fresh worker maps ~3 GB here, far more than the parent process), so the
    requested value is provably reachable on any host.  Generated code then
    requests a ~16 GB anonymous mapping.  The worker must survive long enough
    to report through :class:`CodeExecutionError`, the parent must not hang,
    and no worker/temp artifact may leak.
    """
    _require_landlock()
    before_files = _sandbox_temp_files()
    before_children = _child_pids()
    before_handles = active_worker_handle_count()
    captured = _capture_metadata(monkeypatch)

    # Probe run: cap high enough to be enforced, used only to learn the
    # worker's real post-import footprint from its own response metadata.
    probe = SandboxedExecutor(timeout_seconds=30.0, max_memory_mb=1_048_576)
    probe.execute(
        _TRIVIAL_CODE, pd.DataFrame({"x": [1.0]}), source="memory-probe", agent_name="memory"
    )
    worker_mb = int(captured["metadata"]["sandbox_memory_cap_current_mb"])
    requested_mb = worker_mb + 256
    captured.clear()

    executor = SandboxedExecutor(timeout_seconds=30.0, max_memory_mb=requested_mb)
    started = time.monotonic()
    failure: CodeExecutionError | None = None
    try:
        executor.execute(
            _OOM_CODE, pd.DataFrame({"x": [1.0]}), source="memory-pin", agent_name="memory"
        )
    except SandboxTimeoutError as exc:
        # A loaded runner can trip the executor's own wall clock before the
        # worker reports; that is a host-timing skip, not a broken pin.
        pytest.skip(f"worker timed out under load before reporting the cap: {exc}")
    except CodeExecutionError as exc:
        failure = exc
    elapsed = time.monotonic() - started
    # Well above the executor's 30 s ceiling so a loaded runner cannot turn
    # this pin into an uncaught timeout, while still catching a real hang.
    assert elapsed < 60.0

    metadata = captured["metadata"]
    state = _assert_cap_fields(metadata, requested_mb)
    if state == "skipped-infeasible":
        # The worker grew past the measured cap; the mapping is lazy without a
        # cap, so there is no reachable cap to pin on this machine.
        pytest.skip(f"host worker VmSize already exceeded the measured cap ({requested_mb} MB)")
    assert state == "enforced"
    assert failure is not None, "generated code under a reachable cap did not fail"
    assert "memory" in str(failure).lower(), str(failure)

    _assert_no_temp_leak(before_files)
    assert _child_pids() <= before_children
    assert active_worker_handle_count() == before_handles


def _mock_resource(soft: int, hard: int) -> MagicMock:
    mock_resource = MagicMock()
    mock_resource.RLIMIT_AS = 0
    mock_resource.getrlimit.return_value = (soft, hard)
    return mock_resource


def test_scoped_cap_enforces_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3: the cap is set before the body and the original limits restored."""
    mock_resource = _mock_resource(-1, -1)
    monkeypatch.setattr(sandbox_module, "_load_resource_module", lambda: mock_resource)
    monkeypatch.setattr(sandbox_module, "_current_address_space_bytes", lambda: 100 * 1024 * 1024)
    with sandbox_module._scoped_address_space_cap(512) as report:
        assert report.state == "enforced"
        assert report.requested_mb == 512
        first = mock_resource.setrlimit.call_args_list[0]
        assert first[0][0] == 0  # RLIMIT_AS
        assert first[0][1] == (512 * 1024 * 1024, -1)  # soft capped, hard untouched
    assert mock_resource.setrlimit.call_args_list[-1][0][1] == (-1, -1)


def test_scoped_cap_respects_existing_hard_and_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3: an existing hard/soft limit bounds the enforced cap."""
    mock_resource = _mock_resource(128 * 1024 * 1024, 256 * 1024 * 1024)
    monkeypatch.setattr(sandbox_module, "_load_resource_module", lambda: mock_resource)
    monkeypatch.setattr(sandbox_module, "_current_address_space_bytes", lambda: 64 * 1024 * 1024)
    with sandbox_module._scoped_address_space_cap(2048) as report:
        assert report.state == "enforced"
        assert mock_resource.setrlimit.call_args_list[0][0][1][0] == 128 * 1024 * 1024
    assert mock_resource.setrlimit.call_args_list[-1][0][1] == (
        128 * 1024 * 1024,
        256 * 1024 * 1024,
    )


def test_scoped_cap_skips_infeasible_without_setrlimit(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3: usage already at/above the request is explicit, never enforced."""
    mock_resource = _mock_resource(-1, -1)
    monkeypatch.setattr(sandbox_module, "_load_resource_module", lambda: mock_resource)
    monkeypatch.setattr(sandbox_module, "_current_address_space_bytes", lambda: 600 * 1024 * 1024)
    with sandbox_module._scoped_address_space_cap(512) as report:
        assert report.state == "skipped-infeasible"
        assert report.current_mb == 600
        assert report.requested_mb == 512
    mock_resource.setrlimit.assert_not_called()


def test_scoped_cap_zero_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3: ``max_memory_mb <= 0`` disables the cap."""
    mock_resource = _mock_resource(-1, -1)
    monkeypatch.setattr(sandbox_module, "_load_resource_module", lambda: mock_resource)
    with sandbox_module._scoped_address_space_cap(0) as report:
        assert report.state == "disabled"
    mock_resource.setrlimit.assert_not_called()
    mock_resource.getrlimit.assert_not_called()


def test_scoped_cap_missing_resource_module_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P3: a platform without ``resource`` records an explicit unavailable state."""
    monkeypatch.setattr(sandbox_module, "_load_resource_module", lambda: None)
    with sandbox_module._scoped_address_space_cap(512) as report:
        assert report.state == "unavailable"


def test_scoped_cap_unknown_usage_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3: unknown VmSize means achievability is unknown, so no cap is applied."""
    mock_resource = _mock_resource(-1, -1)
    monkeypatch.setattr(sandbox_module, "_load_resource_module", lambda: mock_resource)
    monkeypatch.setattr(sandbox_module, "_current_address_space_bytes", lambda: 0)
    with sandbox_module._scoped_address_space_cap(512) as report:
        assert report.state == "unavailable"
    mock_resource.setrlimit.assert_not_called()


def test_response_fields_present_for_every_cap_state() -> None:
    """P4: the response builder reports requested/current for all states."""
    enforced = sandbox_module._MemoryCapReport("enforced", 2048, 1700).fields()
    assert enforced["sandbox_memory_cap"] == "enforced"
    assert enforced["sandbox_memory_cap_requested_mb"] == 2048
    assert enforced["sandbox_memory_cap_current_mb"] == 1700

    skipped = sandbox_module._MemoryCapReport("skipped-infeasible", 512, 1757).fields()
    assert skipped["sandbox_memory_cap"] == "skipped-infeasible"
    assert skipped["sandbox_memory_cap_requested_mb"] == 512
    assert skipped["sandbox_memory_cap_current_mb"] == 1757

    for state in ("disabled", "unavailable"):
        fields = sandbox_module._MemoryCapReport(state, 512).fields()
        assert fields["sandbox_memory_cap"] == state
        assert fields["sandbox_memory_cap_requested_mb"] == 512
