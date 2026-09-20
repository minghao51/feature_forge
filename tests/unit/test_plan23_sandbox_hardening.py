"""Regression tests encoding findings from docs/plan/23_evaluation_integrity_security_hardening.md §1.

Finding 1 (Critical): AST validation only blocked pandas file-I/O attribute names
(``BLOCKED_IO_ATTRS``), so NumPy file APIs such as ``np.fromfile`` passed
``SandboxedExecutor._parse_and_validate``, and the worker process keeps the
host user's filesystem permissions. The tests here pin the fixed behavior:
every generated-code path that can read or write host files is rejected with
:class:`SandboxValidationError` (plan 23 PR 5, ADR 0019 decision 4), and the
strict profile additionally contains file reads at the Landlock kernel
boundary.

Finding 9 (Medium): ``methods/malmas/pipeline/core._exec_sandbox`` wrapped
``asyncio.to_thread(sandbox.execute, ...)`` in ``wait_for``; cancelling the
outer future cannot stop the running thread. One test pins the bounded
worker-process lifecycle; the others pin the fixed cancellation contract:
``_exec_sandbox`` awaits ``SandboxedExecutor.execute_async`` directly, so task
cancellation reaches the sandbox coroutine and cleanup leaves no worker
process, thread, queue, or temporary file (plan 23 PR 5, ADR 0019 decision 5).
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from feature_forge.evaluation import sandbox as sandbox_module
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import SandboxTimeoutError, SandboxValidationError
from feature_forge.methods.malmas.pipeline.core import _exec_sandbox

_SANDBOX_TMP_PREFIXES = ("ff_sandbox_", "feature_forge_input_")


def _sandbox_tmp_names() -> set[str]:
    """Names of current sandbox temp artifacts in the system temp directory.

    The scan sees the whole shared temp dir: a concurrent pytest run on the
    same host could plant ``ff_sandbox_*`` files inside the window. The leak
    assertions therefore settle (see ``_assert_no_sandbox_temp_leak``).
    """
    tmp = Path(tempfile.gettempdir())
    return {p.name for p in tmp.iterdir() if p.name.startswith(_SANDBOX_TMP_PREFIXES)}


def _assert_no_sandbox_temp_leak(before: set[str], settle: float = 3.0) -> None:
    """Assert no sandbox temp files remain, tolerating concurrent runs.

    A concurrent pytest process on the same host legitimately creates these
    prefixes while this test's own artifacts are already cleaned (observed
    during multi-version parallel validation); a genuine leak persists, so
    poll the bounded settle window before failing the pin.
    """
    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        if not (_sandbox_tmp_names() - before):
            return
        time.sleep(0.1)
    leaked = _sandbox_tmp_names() - before
    assert not leaked, f"sandbox temp files leaked: {sorted(leaked)}"


def _wait_for_new_children_exit(
    before_pids: set[int], timeout: float = 10.0
) -> list[mp.process.BaseProcess]:
    """Bounded wait for worker processes started during the sandbox call.

    ``mp.active_children()`` is process-global and also reports joblib/Loky
    workers left alive by earlier tests, so the pin must scope itself to
    children whose pid did not exist before the call. Reaping a terminated
    spawn child can lag under load; poll instead of asserting immediately so
    the pin stays deterministic without weakening what it pins.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = [child for child in mp.active_children() if child.pid not in before_pids]
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(0.05)


def _gen_code(body: str) -> str:
    """Wrap ``body`` statements (already indented-friendly) in generate_features(df)."""
    lines = "\n".join(f"    {line}" for line in body.splitlines())
    return f"def generate_features(df):\n{lines}\n    return df\n"


def _require_landlock() -> None:
    """Skip strict-profile lifecycle pins on hosts without Landlock.

    These tests build ``SandboxedExecutor`` with the default strict profile,
    so ``_ensure_strict_available`` raises ``SandboxContainmentError``
    pre-launch on macOS/Windows and the ``SandboxTimeoutError`` pin would
    fail rather than skip (plan 23 §8 keeps strict qualification Linux-only;
    same guard pattern as ``test_sandbox_containment._require_landlock``).
    """
    if not sys.platform.startswith("linux"):
        pytest.skip("Landlock containment is Linux-specific (plan 23 §8)")
    if sandbox_module._probe_landlock_abi() is None:
        pytest.skip("Landlock is unavailable on this host (plan 23 §8)")


class _BlockingSandbox:
    """Duck-typed sandbox whose ``execute_async`` blocks until cancelled.

    2026-09-18 fixture adjustment (same change as the marker removal, plan 23
    PR 5): the pre-fix fixture blocked inside ``execute`` on a
    ``threading.Event`` because the pinned finding was ``asyncio.to_thread``
    hiding an unstoppable worker thread. The fixed ``_exec_sandbox`` awaits
    ``execute_async`` directly on the event loop, so the double now blocks
    inside the coroutine; the pinned intent is unchanged — cancelling the
    outer ``wait_for`` must stop the sandbox work promptly and leave no
    lingering executor thread.
    """

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release: asyncio.Event | None = None
        self.cancelled = False

    async def execute_async(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
    ) -> pd.DataFrame:
        self.release = asyncio.Event()
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return df


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.0, 2.0, 3.0]})


class TestFinding1NumpyAndPandasFileApis:
    """Finding 1: generated code must not reach host-file APIs, numpy included."""

    @pytest.mark.parametrize(
        ("body", "api_id"),
        [
            ("arr = np.fromfile('/etc/hostname', dtype=np.uint8)", "fromfile"),
            ("obj = np.load('/etc/hostname', allow_pickle=False)", "load"),
            ("np.save('/tmp/ff_leak.npy', df)", "save"),
            ("np.savez('/tmp/ff_leak.npz', df)", "savez"),
            ("mm = np.memmap('/etc/hostname', dtype=np.uint8)", "memmap"),
            ("arr = np.loadtxt('/etc/hostname')", "loadtxt"),
            ("arr = np.genfromtxt('/etc/hostname')", "genfromtxt"),
        ],
    )
    def test_ast_blocks_numpy_file_apis(self, api_id: str, body: str) -> None:
        code = "import numpy as np\n\n\n" + _gen_code(body)
        executor = SandboxedExecutor()
        with pytest.raises(SandboxValidationError):
            executor._parse_and_validate(code)

    def test_ast_blocks_aliased_numpy_file_api(self) -> None:
        code = "import numpy as np\nimport numpy as n2\n\n\n" + _gen_code(
            "arr = n2.fromfile('/etc/hostname', dtype=np.uint8)"
        )
        executor = SandboxedExecutor()
        with pytest.raises(SandboxValidationError):
            executor._parse_and_validate(code)

    @pytest.mark.parametrize(
        ("body", "api_id"),
        [
            ("extra = pd.read_feather('/tmp/ff_leak.feather')", "read_feather"),
            ("extra = pd.read_orc('/tmp/ff_leak.orc')", "read_orc"),
            ("extra = pd.read_stata('/tmp/ff_leak.dta')", "read_stata"),
            ("extra = pd.read_xml('/tmp/ff_leak.xml')", "read_xml"),
            ("df.to_hdf('/tmp/ff_leak.h5', key='df')", "to_hdf"),
            ("df.to_feather('/tmp/ff_leak.feather')", "to_feather"),
        ],
    )
    def test_ast_blocks_pandas_equivalent_io(self, api_id: str, body: str) -> None:
        code = "import numpy as np\nimport pandas as pd\n\n\n" + _gen_code(body)
        executor = SandboxedExecutor()
        with pytest.raises(SandboxValidationError):
            executor._parse_and_validate(code)

    """
    Strict-profile execution cannot exfiltrate an external sentinel (plan §7.11).

    AST validation rejects ``np.fromfile`` up front; the Landlock boundary in
    tests/unit/test_sandbox_containment.py independently proves the kernel
    denial for paths that static policy might miss.
    """

    def test_execute_cannot_read_external_sentinel(self, small_df: pd.DataFrame) -> None:
        payload = os.urandom(64)
        fd, sentinel_path = tempfile.mkstemp(prefix="ff_plan23_sentinel_")
        try:
            with os.fdopen(fd, "wb"):
                pass
            with open(sentinel_path, "wb") as fh:
                fh.write(payload)
            code = "import numpy as np\n\n\n" + _gen_code(
                f"arr = np.fromfile({sentinel_path!r}, dtype=np.uint8)\n"
                "result = df.assign(leak=float(arr.sum()))"
            )
            executor = SandboxedExecutor(timeout_seconds=10.0)
            with pytest.raises(SandboxValidationError):
                executor.execute(code, small_df, source="plan23", agent_name="plan23")
        finally:
            with contextlib.suppress(OSError):
                os.unlink(sentinel_path)


class TestFinding9BoundedLifecycle:
    """Finding 9: sandbox lifecycle must be bounded and cancellable."""

    def test_timeout_cleans_up_worker_and_temp_files(self, small_df: pd.DataFrame) -> None:
        """Regression pin (expected to pass today): timeout terminates the worker."""
        _require_landlock()
        code = "def generate_features(df):\n    while True:\n        pass\n    return df\n"
        before = _sandbox_tmp_names()
        before_children = {child.pid for child in mp.active_children() if child.pid is not None}
        executor = SandboxedExecutor(timeout_seconds=1.0)
        with pytest.raises(SandboxTimeoutError):
            executor.execute(code, small_df, source="plan23_pin", agent_name="plan23_pin")
        remaining = _wait_for_new_children_exit(before_children)
        assert remaining == [], f"sandbox worker processes still alive: {remaining}"
        _assert_no_sandbox_temp_leak(before)

    def test_exec_sandbox_cancellation_stops_sandbox_work(self, small_df: pd.DataFrame) -> None:
        fake = _BlockingSandbox()
        # Duck-typed stand-in for SandboxedExecutor to exercise the _exec_sandbox seam.
        sandbox = cast("SandboxedExecutor", fake)
        started_at = time.monotonic()

        async def scenario() -> None:
            # Observe from inside the running loop: asyncio.run() joins the
            # default executor on shutdown, hiding post-cancellation state.
            await asyncio.wait_for(
                _exec_sandbox(
                    sandbox,
                    "agent_a",
                    _gen_code("return df"),
                    small_df,
                    "plan23",
                    0.2,
                ),
                timeout=0.5,
            )

        try:
            with pytest.raises(TimeoutError):
                asyncio.run(scenario())
            assert fake.started.is_set(), "execute_async never started"
            assert fake.cancelled, "asyncio cancellation never reached the sandbox coroutine"
            assert time.monotonic() - started_at < 1.0, (
                "event loop stayed blocked past the cancellation deadline"
            )
        finally:
            if fake.release is not None:
                fake.release.set()

    def test_async_sandbox_entry_point_is_cancellable_and_bounded(
        self, small_df: pd.DataFrame
    ) -> None:
        _require_landlock()

        async def scenario() -> None:
            executor = SandboxedExecutor(timeout_seconds=1.0)
            await asyncio.wait_for(
                executor.execute_async(
                    "def generate_features(df):\n    while True:\n        pass\n",
                    small_df,
                    source="plan23",
                    agent_name="plan23",
                ),
                timeout=3.0,
            )

        before = _sandbox_tmp_names()
        # Scope to children started during the call: mp.active_children() is
        # process-global and also reports joblib/Loky workers left by earlier
        # tests in a full-suite run (same approach as the lifecycle pin above).
        before_children = {child.pid for child in mp.active_children() if child.pid is not None}
        with pytest.raises(SandboxTimeoutError):
            asyncio.run(scenario())
        remaining = _wait_for_new_children_exit(before_children)
        assert remaining == [], f"sandbox worker processes still alive: {remaining}"
        _assert_no_sandbox_temp_leak(before)
