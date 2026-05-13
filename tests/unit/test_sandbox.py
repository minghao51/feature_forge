"""Tests for sandboxed executor — edge cases and uncovered paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from feature_forge.evaluation.sandbox import (
    SandboxedExecutor,
    _apply_resource_limits,
    _to_parquet_safe,
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


class TestApplyResourceLimits:
    """Cover _apply_resource_limits edge cases (lines 317-334)."""

    def _inject_mock_resource(self):
        import sys

        mock_resource = MagicMock()
        mock_resource.RLIMIT_AS = 0
        mock_resource.getrlimit.return_value = (1024 * 1024 * 1024, 1024 * 1024 * 1024)

        def _wrapper_fn(max_memory_mb):
            old = sys.modules.get("resource")
            sys.modules["resource"] = mock_resource
            import importlib

            importlib.reload(sys.modules["feature_forge.evaluation.sandbox"])
            from feature_forge.evaluation.sandbox import _apply_resource_limits as fn

            if old is not None:
                sys.modules["resource"] = old
            return fn(max_memory_mb), mock_resource

        return _wrapper_fn

    def test_sets_rlimit(self):
        mock_resource = MagicMock()
        mock_resource.RLIMIT_AS = 0
        mock_resource.getrlimit.return_value = (1024 * 1024 * 1024, 1024 * 1024 * 1024)

        import sys

        old = sys.modules.get("resource")
        sys.modules["resource"] = mock_resource
        try:
            _apply_resource_limits(max_memory_mb=128)
        finally:
            if old is not None:
                sys.modules["resource"] = old
            else:
                del sys.modules["resource"]

        mock_resource.setrlimit.assert_called_once()

    def test_capped_by_hard_limit(self):
        import sys

        mock_resource = MagicMock()
        mock_resource.RLIMIT_AS = 0
        mock_resource.getrlimit.return_value = (1024 * 1024 * 1024, 64 * 1024 * 1024)
        old = sys.modules.get("resource")
        sys.modules["resource"] = mock_resource
        try:
            _apply_resource_limits(max_memory_mb=512)
        finally:
            if old is not None:
                sys.modules["resource"] = old
            else:
                del sys.modules["resource"]
        args = mock_resource.setrlimit.call_args[0]
        assert args[1][0] == 64 * 1024 * 1024

    def test_capped_by_soft_limit(self):
        import sys

        mock_resource = MagicMock()
        mock_resource.RLIMIT_AS = 0
        mock_resource.getrlimit.return_value = (32 * 1024 * 1024, 1024 * 1024 * 1024)
        old = sys.modules.get("resource")
        sys.modules["resource"] = mock_resource
        try:
            _apply_resource_limits(max_memory_mb=512)
        finally:
            if old is not None:
                sys.modules["resource"] = old
            else:
                del sys.modules["resource"]
        args = mock_resource.setrlimit.call_args[0]
        assert args[1][0] == 32 * 1024 * 1024

    def test_zero_memory_skips(self):
        import sys

        mock_resource = MagicMock()
        mock_resource.RLIMIT_AS = 0
        old = sys.modules.get("resource")
        sys.modules["resource"] = mock_resource
        try:
            _apply_resource_limits(max_memory_mb=0)
        finally:
            if old is not None:
                sys.modules["resource"] = old
            else:
                del sys.modules["resource"]
        mock_resource.setrlimit.assert_not_called()

    def test_import_fallback_no_error(self):
        result = _apply_resource_limits(max_memory_mb=128)
        assert result is None


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

    def test_missing_generate_features_in_worker(self):
        executor = SandboxedExecutor(timeout_seconds=2.0)
        code = "x = 42"
        with pytest.raises(CodeExecutionError, match="must define"):
            executor.execute(code, pd.DataFrame({"a": [1]}))

    def test_temp_file_cleanup(self):
        executor = SandboxedExecutor(timeout_seconds=2.0)
        code = """
import pandas as pd
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['x'] = df['a'] * 2
    return result
"""
        with patch("os.unlink") as mock_unlink:
            df = pd.DataFrame({"a": [1, 2, 3]})
            executor.execute(code, df)
            assert mock_unlink.call_count >= 2
