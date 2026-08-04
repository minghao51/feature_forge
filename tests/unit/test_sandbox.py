"""Tests for sandboxed executor — edge cases and uncovered paths."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from feature_forge.evaluation.sandbox import (
    SandboxedExecutor,
    _current_vmsize_bytes,
    _get_worker_context,
    _memory_limit_ctx,
    _read_ipc_frame,
    _to_parquet_safe,
    _write_ipc_frame,
)
from feature_forge.exceptions import CodeExecutionError, SandboxValidationError


class TestToParquetSafe:
    """Cover _to_parquet_safe edge cases (lines 34-69)."""

    def test_categorical_interval_dtype(self):
        intervals = pd.IntervalIndex.from_breaks([0, 5, 10])
        cats = pd.Categorical(intervals)
        df = pd.DataFrame({"x": cats})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == float

    def test_categorical_with_str_conversion(self):
        cats = pd.Categorical(["a", "b", "c"])
        df = pd.DataFrame({"x": cats})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == object

    def test_object_column_numeric_coercion(self):
        df = pd.DataFrame({"x": ["1", "2", "3"]})
        result = _to_parquet_safe(df)
        assert np.issubdtype(result["x"].dtype, np.number)

    def test_object_column_all_strings(self):
        df = pd.DataFrame({"x": ["hello", "world"]})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == object

    def test_extension_dtype_passthrough(self):
        s = pd.Series([1, 2, 3], dtype=pd.Int64Dtype())
        df = pd.DataFrame({"x": s})
        result = _to_parquet_safe(df)
        assert "x" in result.columns

    def test_non_numeric_bool_preserved(self):
        df = pd.DataFrame({"x": [True, False, True]})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == bool

    def test_mixed_object_column(self):
        df = pd.DataFrame({"x": [1, "two", 3.0]})
        result = _to_parquet_safe(df)
        assert np.issubdtype(result["x"].dtype, np.number) or result["x"].dtype == object

    def test_series_with_only_nan_after_numeric(self):
        df = pd.DataFrame({"x": ["nan", "nan", "nan"]})
        result = _to_parquet_safe(df)
        assert pd.api.types.is_string_dtype(result["x"].dtype) or result["x"].isna().all()

    def test_numeric_column_passthrough(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == float


class TestIpcFrameFallback:
    """The sandbox transports DataFrames parent<->worker via parquet.

    On some CI runners pyarrow._parquet.so fails to mmap inside a spawn
    worker even though the parent's parquet write succeeded. The IPC helpers
    must therefore write a pickle sidecar up front and fall back to it on
    read failure. These tests pin that contract.
    """

    def test_write_produces_parquet_and_pickle_sidecar(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        # _write_ipc_frame uses its own tempfile; patch the prefix so we can
        # locate and clean up the artifacts deterministically.
        with patch("feature_forge.evaluation.sandbox.tempfile.NamedTemporaryFile") as mock:
            parquet_path = str(tmp_path / "frame.parquet")
            open(parquet_path, "wb").close()
            mock.return_value.__enter__.return_value.name = parquet_path
            mock.return_value.name = parquet_path
            primary = _write_ipc_frame(df, prefix="test_")
        assert primary == parquet_path
        assert os.path.exists(parquet_path)
        assert os.path.exists(parquet_path + ".pkl")

    def test_read_falls_back_to_pickle_sidecar_when_parquet_fails(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        primary = _write_ipc_frame(df, prefix=str(tmp_path / "frame_")[: -len("frame_")])
        # Simulate the CI failure: parquet read raises (e.g. mmap denied),
        # but the pickle sidecar is present and must be used.
        with patch(
            "feature_forge.evaluation.sandbox.pd.read_parquet",
            side_effect=OSError("failed to map segment from shared object"),
        ):
            result = _read_ipc_frame(primary)
        pd.testing.assert_frame_equal(result, df)

    def test_read_raises_when_parquet_fails_and_no_sidecar(self, tmp_path):
        parquet_path = str(tmp_path / "lone.parquet")
        open(parquet_path, "wb").close()
        with patch(
            "feature_forge.evaluation.sandbox.pd.read_parquet",
            side_effect=OSError("failed to map segment"),
        ):
            with pytest.raises(OSError, match="failed to map segment"):
                _read_ipc_frame(parquet_path)


class TestWorkerContextSelection:
    """``_get_worker_context`` picks fork on Linux, spawn elsewhere.

    fork gives near-instant worker startup on Linux (copy-on-write shares the
    parent's already-imported modules), while spawn is the portable fallback
    for macOS (fork is deprecated there) and Windows (fork unavailable).
    """

    def test_linux_uses_fork(self):
        with patch("feature_forge.evaluation.sandbox.sys") as mock_sys:
            mock_sys.platform = "linux"
            ctx = _get_worker_context()
        assert ctx.get_start_method() == "fork"

    def test_macos_uses_spawn(self):
        with patch("feature_forge.evaluation.sandbox.sys") as mock_sys:
            mock_sys.platform = "darwin"
            ctx = _get_worker_context()
        assert ctx.get_start_method() == "spawn"

    def test_windows_uses_spawn(self):
        with patch("feature_forge.evaluation.sandbox.sys") as mock_sys:
            mock_sys.platform = "win32"
            ctx = _get_worker_context()
        assert ctx.get_start_method() == "spawn"

    def test_current_platform_returns_a_valid_context(self):
        # Whatever platform we're on, the helper must return a usable context
        # whose Process/Pipe/Queue constructors work.
        ctx = _get_worker_context()
        assert ctx.get_start_method() in {"fork", "spawn", "forkserver"}


class TestMemoryLimitCtx:
    """``_memory_limit_ctx`` scopes RLIMIT_AS around user-code execution only.

    On Linux RLIMIT_AS caps the process's *total* virtual address space,
    which on a fresh Python 3.13 spawn worker already exceeds 1GB once
    pandas/numpy/pyarrow are imported. A cap below that level turns every
    subsequent allocation into MemoryError. These tests pin the two
    guardrails that keep the cap from bricking the worker:
    1. Skip the cap when the target is at or below the current VmSize.
    2. Back off via a 32MB canary when the cap is otherwise unreachable.
    """

    def test_skips_when_target_at_or_below_current_vmsize(self):
        # Force the helper to report a VmSize larger than the target so the
        # context manager must skip the cap entirely.
        with patch(
            "feature_forge.evaluation.sandbox._current_vmsize_bytes",
            return_value=2 * 1024 * 1024 * 1024,  # 2GB
        ):
            import sys

            mock_resource = MagicMock()
            mock_resource.RLIMIT_AS = 0
            old = sys.modules.get("resource")
            sys.modules["resource"] = mock_resource
            try:
                with _memory_limit_ctx(max_memory_mb=512):
                    # No setrlimit call should have happened — the cap was
                    # skipped because target (512MB) ≤ current VmSize (2GB).
                    mock_resource.setrlimit.assert_not_called()
            finally:
                if old is not None:
                    sys.modules["resource"] = old
                else:
                    del sys.modules["resource"]

    def test_canary_backs_off_when_allocation_fails(self):
        # Simulate a cap that the canary cannot satisfy. The context manager
        # must restore the original limit and yield control without applying
        # the (broken) cap.
        import sys

        mock_resource = MagicMock()
        mock_resource.RLIMIT_AS = 0
        # Hard limit and soft limit high enough that the cap arithmetic does
        # not itself skip; the failure path we want exercised is the canary.
        mock_resource.getrlimit.return_value = (
            4 * 1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
        )
        old = sys.modules.get("resource")
        sys.modules["resource"] = mock_resource
        try:
            with (
                patch(
                    "feature_forge.evaluation.sandbox._current_vmsize_bytes",
                    return_value=0,  # unknown — don't skip via the VmSize path
                ),
                patch(
                    "feature_forge.evaluation.sandbox.bytearray",
                    side_effect=MemoryError,
                ),
            ):
                with _memory_limit_ctx(max_memory_mb=512):
                    # The cap was applied, the canary failed, and the
                    # original limit was restored — so we expect at least
                    # three setrlimit calls (apply, restore-on-canary-fail).
                    assert mock_resource.setrlimit.call_count >= 2
        finally:
            if old is not None:
                sys.modules["resource"] = old
            else:
                del sys.modules["resource"]

    def test_zero_target_yields_without_touching_limits(self):
        import sys

        mock_resource = MagicMock()
        mock_resource.RLIMIT_AS = 0
        old = sys.modules.get("resource")
        sys.modules["resource"] = mock_resource
        try:
            with _memory_limit_ctx(max_memory_mb=0):
                mock_resource.setrlimit.assert_not_called()
        finally:
            if old is not None:
                sys.modules["resource"] = old
            else:
                del sys.modules["resource"]

    def test_vmsize_helper_returns_nonnegative(self):
        # On Linux this reads /proc/self/status; elsewhere it returns 0.
        # Either way it must be a non-negative int and not raise.
        result = _current_vmsize_bytes()
        assert isinstance(result, int)
        assert result >= 0


class TestSandboxExecutorEdgeCases:
    """Cover sandbox executor edge paths."""

    def test_worker_non_dataframe_return(self):
        executor = SandboxedExecutor(timeout_seconds=2.0)
        code = """
def generate_features(df):
    return [1, 2, 3]
"""
        with pytest.raises(CodeExecutionError, match="must return a DataFrame"):
            executor.execute(code, pd.DataFrame({"a": [1, 2, 3]}))

    def test_forbidden_name_ref_expression(self):
        executor = SandboxedExecutor()
        code = "__import__\ndef generate_features(df): return df"
        with pytest.raises(SandboxValidationError, match="Forbidden name reference"):
            executor.execute(code, pd.DataFrame())

    def test_forbidden_import_from(self):
        executor = SandboxedExecutor()
        code = "from os import path\ndef generate_features(df): return df"
        with pytest.raises(SandboxValidationError, match="Import from not allowed"):
            executor.execute(code, pd.DataFrame())

    def test_forbidden_name_ref(self):
        executor = SandboxedExecutor()
        code = "x = open\ndef generate_features(df): return df"
        with pytest.raises(SandboxValidationError, match="Forbidden name reference"):
            executor.execute(code, pd.DataFrame())

    def test_execute_invalid_syntax_during_parse(self):
        executor = SandboxedExecutor()
        with pytest.raises(CodeExecutionError, match="Invalid syntax"):
            executor.execute("def generate_features(df", pd.DataFrame())

    def test_blocked_file_io_api_usage(self):
        executor = SandboxedExecutor()
        code = """
def generate_features(df):
    out = df.copy()
    out.to_csv("x.csv")
    return out
"""
        with pytest.raises(SandboxValidationError, match="Blocked file I/O API usage"):
            executor.execute(code, pd.DataFrame({"a": [1, 2, 3]}))

    def test_blocked_network_api_usage(self):
        executor = SandboxedExecutor()
        code = """
def generate_features(df):
    df.create_connection()
    return df
"""
        with pytest.raises(SandboxValidationError, match="Blocked network API usage"):
            executor.execute(code, pd.DataFrame({"a": [1, 2, 3]}))

    def test_missing_generate_features_in_worker(self):
        executor = SandboxedExecutor(timeout_seconds=2.0)
        code = "x = 42"
        with pytest.raises(CodeExecutionError, match="must define"):
            executor.execute(code, pd.DataFrame({"a": [1]}))

    def test_temp_file_cleanup(self):
        executor = SandboxedExecutor(timeout_seconds=2.0)
        code = """
def generate_features(df):
    result = df.copy()
    result['x'] = df['a'] * 2
    return result
"""
        with patch("os.unlink") as mock_unlink:
            df = pd.DataFrame({"a": [1, 2, 3]})
            executor.execute(code, df)
            assert mock_unlink.call_count >= 2


class TestParseAndValidate:
    """Cover parse_and_validate edge paths."""

    def test_blocks_regular_import_not_allowed(self):
        executor = SandboxedExecutor()
        code = "import os\ndef generate_features(df): return df"
        with pytest.raises(SandboxValidationError, match="Import not allowed"):
            executor.execute(code, pd.DataFrame())

    def test_allows_regular_import_allowed(self):
        executor = SandboxedExecutor()
        code = "import math\ndef generate_features(df): return df"
        result = executor.execute(code, pd.DataFrame({"a": [1.0]}))
        assert isinstance(result, pd.DataFrame)

    def test_blocks_dunder_attribute_access(self):
        executor = SandboxedExecutor()
        code = """
def generate_features(df):
    df.__class__
    return df
"""
        with pytest.raises(SandboxValidationError, match="Forbidden dunder attribute"):
            executor.execute(code, pd.DataFrame({"a": [1]}))

    def test_allows_import_from_allowed(self):
        executor = SandboxedExecutor()
        code = "from math import sqrt\ndef generate_features(df): return df"
        result = executor.execute(code, pd.DataFrame({"a": [1.0]}))
        assert isinstance(result, pd.DataFrame)


class TestExecuteFullPath:
    """Cover the full execute() success path with metadata assertions."""

    def test_success_path_returns_features(self):
        executor = SandboxedExecutor(timeout_seconds=5.0)
        code = """
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['double'] = df['a'] * 2
    return result
"""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = executor.execute(code, df, source="test", agent_name="test_agent")
        assert list(result["double"]) == [2.0, 4.0, 6.0]
        assert len(result) == 3

    def test_success_path_preserves_index(self):
        executor = SandboxedExecutor(timeout_seconds=5.0)
        code = """
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['sum'] = df['a'] + df['b']
    return result
"""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]}, index=[10, 20])
        result = executor.execute(code, df)
        assert list(result.index) == [10, 20]
        assert list(result["sum"]) == [11.0, 22.0]

    def test_parquet_output_valid(self, tmp_path):
        executor = SandboxedExecutor(timeout_seconds=5.0)
        code = """
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['x'] = df.iloc[:, 0] * 3
    return result
"""
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        result = executor.execute(code, df)
        parquet_path = str(tmp_path / "output.parquet")
        result.to_parquet(parquet_path)
        loaded = pd.read_parquet(parquet_path)
        assert list(loaded["x"]) == [3.0, 6.0, 9.0]
