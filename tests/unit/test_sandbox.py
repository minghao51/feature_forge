"""Tests for sandboxed executor — edge cases and uncovered paths."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from feature_forge.evaluation import sandbox as sandbox_module
from feature_forge.evaluation.sandbox import (
    _MAX_RESPONSE_PAYLOAD_CHARS,
    SandboxedExecutor,
    _scoped_address_space_cap,
    _to_parquet_safe,
)
from feature_forge.exceptions import CodeExecutionError, SandboxValidationError


class TestToParquetSafe:
    """Cover _to_parquet_safe edge cases (lines 34-69)."""

    def test_categorical_interval_dtype(self) -> None:
        intervals = pd.IntervalIndex.from_breaks([0, 5, 10])
        cats = pd.Categorical(intervals)
        df = pd.DataFrame({"x": cats})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == float

    def test_categorical_with_str_conversion(self) -> None:
        cats = pd.Categorical(["a", "b", "c"])
        df = pd.DataFrame({"x": cats})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == object

    def test_object_column_numeric_coercion(self) -> None:
        df = pd.DataFrame({"x": ["1", "2", "3"]})
        result = _to_parquet_safe(df)
        # _to_parquet_safe only produces numpy dtypes here, but pandas-stubs
        # types Series.dtype as a numpy/extension union that cannot be narrowed.
        assert np.issubdtype(cast("np.dtype[Any]", result["x"].dtype), np.number)

    def test_object_column_all_strings(self) -> None:
        df = pd.DataFrame({"x": ["hello", "world"]})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == object

    def test_extension_dtype_passthrough(self) -> None:
        s = pd.Series([1, 2, 3], dtype=pd.Int64Dtype())
        df = pd.DataFrame({"x": s})
        result = _to_parquet_safe(df)
        assert "x" in result.columns

    def test_non_numeric_bool_preserved(self) -> None:
        df = pd.DataFrame({"x": [True, False, True]})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == bool

    def test_mixed_object_column(self) -> None:
        df = pd.DataFrame({"x": [1, "two", 3.0]})
        result = _to_parquet_safe(df)
        # See test_object_column_numeric_coercion for the cast rationale.
        assert (
            np.issubdtype(cast("np.dtype[Any]", result["x"].dtype), np.number)
            or result["x"].dtype == object
        )

    def test_series_with_only_nan_after_numeric(self) -> None:
        df = pd.DataFrame({"x": ["nan", "nan", "nan"]})
        result = _to_parquet_safe(df)
        assert pd.api.types.is_string_dtype(result["x"].dtype) or result["x"].isna().all()

    def test_numeric_column_passthrough(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = _to_parquet_safe(df)
        assert result["x"].dtype == float


class TestScopedAddressSpaceCap:
    """Pin the scoped RLIMIT_AS contract with a mocked ``resource`` module.

    Slice B step 4 replaces entry-time limiting: the cap is applied only
    around generated-code execution and the pre-exec limits are restored
    afterwards, so the worker's own import/input-load/publish mappings never
    run under it.
    """

    def _mock_resource(self, monkeypatch: pytest.MonkeyPatch, soft: int, hard: int) -> MagicMock:
        mock_resource = MagicMock()
        mock_resource.RLIMIT_AS = 0
        mock_resource.getrlimit.return_value = (soft, hard)
        monkeypatch.setattr(sandbox_module, "_load_resource_module", lambda: mock_resource)
        monkeypatch.setattr(
            sandbox_module, "_current_address_space_bytes", lambda: 64 * 1024 * 1024
        )
        return mock_resource

    def test_sets_only_soft_and_restores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resource = self._mock_resource(monkeypatch, soft=-1, hard=-1)
        with _scoped_address_space_cap(max_memory_mb=128) as report:
            assert report.state == "enforced"
            assert mock_resource.setrlimit.call_args_list[0][0][1] == (
                128 * 1024 * 1024,
                -1,
            )
        assert mock_resource.setrlimit.call_args_list[-1][0][1] == (-1, -1)

    def test_capped_by_hard_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resource = self._mock_resource(
            monkeypatch, soft=1024 * 1024 * 1024, hard=64 * 1024 * 1024
        )
        with _scoped_address_space_cap(max_memory_mb=512) as report:
            assert report.state == "enforced"
            args = mock_resource.setrlimit.call_args[0]
            assert args[1][0] == 64 * 1024 * 1024

    def test_capped_by_soft_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resource = self._mock_resource(
            monkeypatch, soft=32 * 1024 * 1024, hard=1024 * 1024 * 1024
        )
        with _scoped_address_space_cap(max_memory_mb=512) as report:
            assert report.state == "enforced"
            args = mock_resource.setrlimit.call_args[0]
            assert args[1][0] == 32 * 1024 * 1024

    def test_zero_memory_records_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resource = self._mock_resource(monkeypatch, soft=-1, hard=-1)
        with _scoped_address_space_cap(max_memory_mb=0) as report:
            assert report.state == "disabled"
        mock_resource.setrlimit.assert_not_called()

    def test_infeasible_cap_is_skipped_not_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resource = self._mock_resource(monkeypatch, soft=-1, hard=-1)
        monkeypatch.setattr(
            sandbox_module, "_current_address_space_bytes", lambda: 700 * 1024 * 1024
        )
        with _scoped_address_space_cap(max_memory_mb=512) as report:
            assert report.state == "skipped-infeasible"
            assert report.current_mb == 700
        mock_resource.setrlimit.assert_not_called()

    def test_real_rlimit_is_restored(self) -> None:
        """Real (unmocked) helper restores the host's limits afterwards.

        Runs in a subprocess so the real ``setrlimit`` cannot shrink the
        pytest process; this is the restore half of the scoped contract that
        keeps the response send outside the cap.  The cap is derived from the
        subprocess's own measured ``VmSize`` (plus 256 MB) so it is provably
        reachable and therefore actually exercises ``setrlimit`` + restore
        instead of always landing in the skipped-infeasible branch.
        """
        import subprocess
        import sys

        script = (
            "import resource, sys\n"
            "from feature_forge.evaluation.sandbox import (\n"
            "    _current_address_space_bytes,\n"
            "    _scoped_address_space_cap,\n"
            ")\n"
            "current = _current_address_space_bytes()\n"
            "if current <= 0:\n"
            "    sys.exit(0)\n"
            "current_mb = current // (1024 * 1024)\n"
            "before = resource.getrlimit(resource.RLIMIT_AS)\n"
            "with _scoped_address_space_cap(current_mb + 256) as report:\n"
            "    assert report.state == 'enforced', report.state\n"
            "after = resource.getrlimit(resource.RLIMIT_AS)\n"
            "assert after == before, (before, after)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestSandboxExecutorEdgeCases:
    """Cover sandbox executor edge paths."""

    def test_worker_non_dataframe_return(self) -> None:
        executor = SandboxedExecutor(timeout_seconds=5.0)
        code = """
def generate_features(df):
    return [1, 2, 3]
"""
        with pytest.raises(CodeExecutionError, match="must return a DataFrame"):
            executor.execute(code, pd.DataFrame({"a": [1, 2, 3]}))

    def test_forbidden_name_ref_expression(self) -> None:
        executor = SandboxedExecutor()
        code = "__import__\ndef generate_features(df): return df"
        with pytest.raises(SandboxValidationError, match="Forbidden name reference"):
            executor.execute(code, pd.DataFrame())

    def test_forbidden_import_from(self) -> None:
        executor = SandboxedExecutor()
        code = "from os import path\ndef generate_features(df): return df"
        with pytest.raises(SandboxValidationError, match="Import from not allowed"):
            executor.execute(code, pd.DataFrame())

    def test_forbidden_name_ref(self) -> None:
        executor = SandboxedExecutor()
        code = "x = open\ndef generate_features(df): return df"
        with pytest.raises(SandboxValidationError, match="Forbidden name reference"):
            executor.execute(code, pd.DataFrame())

    def test_execute_invalid_syntax_during_parse(self) -> None:
        executor = SandboxedExecutor()
        with pytest.raises(CodeExecutionError, match="Invalid syntax"):
            executor.execute("def generate_features(df", pd.DataFrame())

    def test_blocked_file_io_api_usage(self) -> None:
        executor = SandboxedExecutor()
        code = """
def generate_features(df):
    out = df.copy()
    out.to_csv("x.csv")
    return out
"""
        with pytest.raises(SandboxValidationError, match="Blocked file I/O API usage"):
            executor.execute(code, pd.DataFrame({"a": [1, 2, 3]}))

    def test_blocked_network_api_usage(self) -> None:
        executor = SandboxedExecutor()
        code = """
def generate_features(df):
    df.create_connection()
    return df
"""
        with pytest.raises(SandboxValidationError, match="Blocked network API usage"):
            executor.execute(code, pd.DataFrame({"a": [1, 2, 3]}))

    def test_missing_generate_features_in_worker(self) -> None:
        executor = SandboxedExecutor(timeout_seconds=5.0)
        code = "x = 42"
        with pytest.raises(CodeExecutionError, match="must define"):
            executor.execute(code, pd.DataFrame({"a": [1]}))

    def test_temp_file_cleanup(self) -> None:
        executor = SandboxedExecutor(timeout_seconds=5.0)
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

    def test_huge_exception_message_is_capped_before_send(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An enormous exception message is truncated at the worker, not in transit.

        Generated code can inflate exception text without bound
        (``"A" * 10_000_000``).  The worker must cap the payload at
        :data:`_MAX_RESPONSE_PAYLOAD_CHARS` before ``send`` so the parent's
        post-poll ``recv`` drains one bounded message; the typed error still
        arrives within the normal execution budget.
        """
        executor = SandboxedExecutor(timeout_seconds=10.0)
        code = """
def generate_features(df):
    raise ValueError("A" * 10_000_000)
"""
        received: dict[str, str] = {}
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
            received["payload"] = payload
            return original(self_, handle, status, payload, worker_metadata, deadline, **kwargs)

        monkeypatch.setattr(SandboxedExecutor, "_consume_response", capture)

        started = time.monotonic()
        with pytest.raises(CodeExecutionError, match="Feature generation execution failed"):
            executor.execute(code, pd.DataFrame({"a": [1.0]}))
        elapsed = time.monotonic() - started

        # The wire payload is capped at the source; the parent then truncates
        # it to 300 chars when building the typed message.  Allow for the
        # error prefix on top of the cap.
        prefix = "Feature generation execution failed: "
        assert len(received["payload"]) <= _MAX_RESPONSE_PAYLOAD_CHARS + len(prefix)
        assert elapsed < 10.0


class TestParseAndValidate:
    """Cover parse_and_validate edge paths."""

    def test_blocks_regular_import_not_allowed(self) -> None:
        executor = SandboxedExecutor()
        code = "import os\ndef generate_features(df): return df"
        with pytest.raises(SandboxValidationError, match="Import not allowed"):
            executor.execute(code, pd.DataFrame())

    def test_allows_regular_import_allowed(self) -> None:
        executor = SandboxedExecutor()
        code = "import math\ndef generate_features(df): return df"
        result = executor.execute(code, pd.DataFrame({"a": [1.0]}))
        assert isinstance(result, pd.DataFrame)

    def test_blocks_dunder_attribute_access(self) -> None:
        executor = SandboxedExecutor()
        code = """
def generate_features(df):
    df.__class__
    return df
"""
        with pytest.raises(SandboxValidationError, match="Forbidden dunder attribute"):
            executor.execute(code, pd.DataFrame({"a": [1]}))

    def test_allows_import_from_allowed(self) -> None:
        executor = SandboxedExecutor()
        code = "from math import sqrt\ndef generate_features(df): return df"
        result = executor.execute(code, pd.DataFrame({"a": [1.0]}))
        assert isinstance(result, pd.DataFrame)


class TestExecuteFullPath:
    """Cover the full execute() success path with metadata assertions."""

    def test_success_path_returns_features(self) -> None:
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

    def test_success_path_preserves_index(self) -> None:
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

    def test_parquet_output_valid(self, tmp_path: Path) -> None:
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
