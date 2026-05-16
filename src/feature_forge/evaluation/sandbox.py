"""Sandboxed code execution for LLM-generated feature engineering code.

Security model:
1. AST static validation blocks dangerous syntax/name patterns.
2. Execution happens in a dedicated worker process (not in-process).
3. Worker has bounded timeout and optional memory limits.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import multiprocessing as mp
import os
import queue
import socket
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from feature_forge.evaluation.cv import CVEvaluator
import pandas as pd

from feature_forge.exceptions import (
    CodeExecutionError,
    SandboxTimeoutError,
    SandboxValidationError,
)
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


def _to_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame columns to parquet-serializable types.

    Handles Categorical columns with non-serializable values (e.g.,
    pd.Interval), complex objects, and other extension types that
    pyarrow cannot natively serialize.
    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        # Categorical columns may contain Interval or other complex types
        if isinstance(series.dtype, pd.CategoricalDtype):
            try:
                # Try to convert to numeric float first (works for interval midpoints)
                numeric = pd.to_numeric(
                    series.astype(str).str.extract(r"([-\d.]+)", expand=False), errors="coerce"
                )
                if numeric.notna().sum() > len(numeric) * 0.5:
                    df[col] = numeric.astype(float)
                else:
                    df[col] = series.astype(str)
            except Exception as exc:
                logger.debug(
                    "parquet_conversion_numeric_extract_failed", column=col, error=str(exc)
                )
                df[col] = series.astype(str)
        elif series.dtype == object:
            try:
                df[col] = pd.to_numeric(series, errors="coerce")
                if df[col].isna().all() and series.notna().any():
                    df[col] = series.astype(str)
            except Exception as exc:
                logger.debug("parquet_conversion_coerce_failed", column=col, error=str(exc))
                df[col] = series.astype(str)
        # Any unrecognized extension type → float or string
        elif not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            try:
                df[col] = series.astype(float)
            except (TypeError, ValueError):
                df[col] = series.astype(str)
    return df


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 5.0
    max_memory_mb: int = 512


class SandboxedExecutor:
    """AST-validated, process-isolated code execution."""

    FORBIDDEN_NAMES: ClassVar[set[str]] = {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "exit",
        "quit",
        "__builtins__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }
    FORBIDDEN_DUNDER_PREFIX: ClassVar[str] = "__"
    ALLOWED_BUILTINS: ClassVar[set[str]] = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "round",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "enumerate",
        "filter",
        "set",
        "frozenset",
        "isinstance",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "ZeroDivisionError",
        "OverflowError",
        "NotImplementedError",
        "StopIteration",
    }
    ALLOWED_IMPORTS: ClassVar[set[str]] = {"pandas", "numpy", "math"}
    BLOCKED_IO_ATTRS: ClassVar[set[str]] = {
        "read_csv",
        "read_parquet",
        "read_json",
        "read_pickle",
        "read_table",
        "read_excel",
        "read_hdf",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "to_csv",
        "to_parquet",
        "to_json",
        "to_pickle",
        "to_excel",
        "to_sql",
    }
    BLOCKED_NETWORK_ATTRS: ClassVar[set[str]] = {
        "urlopen",
        "request",
        "requests",
        "connect",
        "create_connection",
    }

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_memory_mb: int = 512,
    ) -> None:
        self.limits = SandboxLimits(timeout_seconds=timeout_seconds, max_memory_mb=max_memory_mb)

    @staticmethod
    def from_evaluator(evaluator: CVEvaluator | None) -> SandboxedExecutor:
        if evaluator is not None:
            cfg = evaluator.config.evaluation
            return SandboxedExecutor(
                timeout_seconds=cfg.sandbox_timeout_seconds,
                max_memory_mb=cfg.sandbox_max_memory_mb,
            )
        return SandboxedExecutor(timeout_seconds=5.0, max_memory_mb=512)

    def execute(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
    ) -> pd.DataFrame:
        """Execute feature generation code safely."""
        execute_t0 = time.perf_counter()
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        logger.info(
            "sandbox_execute_start",
            code_length=len(code),
            input_shape=df.shape,
            source=source,
            agent=agent_name,
            code_hash=code_hash,
        )
        tree = self._parse_and_validate(code)
        payload = ast.unparse(tree) if hasattr(ast, "unparse") else code
        result = self._execute_in_worker(payload, df, source=source, agent_name=agent_name)
        latency_ms = round((time.perf_counter() - execute_t0) * 1000, 1)
        logger.info(
            "sandbox_execute_complete",
            result_shape=result.shape,
            latency_ms=latency_ms,
            source=source,
            agent=agent_name,
            code_hash=code_hash,
        )
        return result

    def _execute_in_worker(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
    ) -> pd.DataFrame:
        ctx = mp.get_context("spawn")
        response_queue: mp.Queue[tuple[str, str]] = ctx.Queue(maxsize=1)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".parquet", delete=False, prefix="feature_forge_input_"
        ) as input_file:
            df.to_parquet(input_file.name)
            input_path = input_file.name
        proc = ctx.Process(
            target=_sandbox_worker_main,
            args=(code, input_path, self.limits.max_memory_mb, response_queue, source, agent_name),
            daemon=True,
        )
        proc.start()
        artifact_path = ""
        try:
            try:
                status, payload = response_queue.get(timeout=self.limits.timeout_seconds)
            except queue.Empty as exc:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)
                logger.error("sandbox_timeout", timeout_seconds=self.limits.timeout_seconds)
                raise SandboxTimeoutError(
                    f"Sandbox execution timed out after {self.limits.timeout_seconds:.1f}s"
                ) from exc

            if status == "ok":
                artifact_path = payload
                result = pd.read_parquet(artifact_path)
                if not isinstance(result, pd.DataFrame):
                    raise CodeExecutionError(
                        f"generate_features must return a DataFrame, got {type(result).__name__}"
                    )
                return result
            if status == "blocked":
                raise SandboxValidationError(payload)
            raise CodeExecutionError(payload)
        finally:
            proc.join(timeout=0.5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
            for path in (artifact_path, input_path):
                if path:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass

    def _parse_and_validate(self, code: str) -> ast.AST:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise CodeExecutionError(f"Invalid syntax: {exc}") from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in self.ALLOWED_IMPORTS:
                        logger.warning(
                            "sandbox_validation_blocked", reason=f"import_not_allowed: {alias.name}"
                        )
                        raise SandboxValidationError(f"Import not allowed: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in self.ALLOWED_IMPORTS:
                    logger.warning(
                        "sandbox_validation_blocked",
                        reason=f"import_from_not_allowed: {node.module}",
                    )
                    raise SandboxValidationError(f"Import from not allowed: {node.module}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.FORBIDDEN_NAMES:
                    logger.warning(
                        "sandbox_validation_blocked", reason=f"forbidden_function: {node.func.id}"
                    )
                    raise SandboxValidationError(f"Forbidden function call: {node.func.id}")
            elif isinstance(node, ast.Name):
                if node.id in self.FORBIDDEN_NAMES and isinstance(node.ctx, ast.Load):
                    logger.warning(
                        "sandbox_validation_blocked", reason=f"forbidden_name: {node.id}"
                    )
                    raise SandboxValidationError(f"Forbidden name reference: {node.id}")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith(self.FORBIDDEN_DUNDER_PREFIX):
                    logger.warning(
                        "sandbox_validation_blocked", reason=f"forbidden_dunder: {node.attr}"
                    )
                    raise SandboxValidationError(f"Forbidden dunder attribute access: {node.attr}")
                if node.attr in self.BLOCKED_IO_ATTRS:
                    logger.warning(
                        "sandbox_validation_blocked", reason=f"blocked_io_attr: {node.attr}"
                    )
                    raise SandboxValidationError(f"Blocked file I/O API usage: {node.attr}")
                if node.attr in self.BLOCKED_NETWORK_ATTRS:
                    logger.warning(
                        "sandbox_validation_blocked", reason=f"blocked_network_attr: {node.attr}"
                    )
                    raise SandboxValidationError(f"Blocked network API usage: {node.attr}")

        return tree


def _sandbox_worker_main(
    code: str,
    input_parquet_path: str,
    max_memory_mb: int,
    response_queue: mp.Queue[tuple[str, str]],
    source: str = "unknown",
    agent_name: str = "unknown",
) -> None:
    _apply_resource_limits(max_memory_mb=max_memory_mb)
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]

    def _blocked_network(*_args: Any, **_kwargs: Any) -> Any:
        logger.warning(
            "sandbox_runtime_blocked",
            category="network",
            reason="network_blocked",
            source=source,
            agent=agent_name,
            code_hash=code_hash,
        )
        raise PermissionError("Network access is blocked in sandbox runtime")

    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as exc:
        response_queue.put(("error", f"Failed to read input data: {exc}"))
        return

    socket.create_connection = _blocked_network
    socket.socket = _blocked_network  # type: ignore[assignment,misc]

    import builtins as _builtins

    def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
        root = name.split(".")[0]
        if root not in SandboxedExecutor.ALLOWED_IMPORTS:
            raise ImportError(f"Import not allowed: {name}")
        return _builtins.__import__(name, *args, **kwargs)

    safe_globals: dict[str, Any] = {
        "__builtins__": {
            name: getattr(builtins, name) for name in SandboxedExecutor.ALLOWED_BUILTINS
        },
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
    }
    import math

    safe_globals["__builtins__"]["__import__"] = _restricted_import
    safe_globals["math"] = math
    local_vars: dict[str, Any] = {}

    try:
        exec(compile(code, filename="<sandbox>", mode="exec"), safe_globals, local_vars)
        generate_features = local_vars.get("generate_features")
        if generate_features is None:
            response_queue.put(("error", "Code must define a 'generate_features(df)' function"))
            return
        result = generate_features(df)
        if not isinstance(result, pd.DataFrame):
            response_queue.put(
                ("error", f"generate_features must return a DataFrame, got {type(result).__name__}")
            )
            return
        # Convert non-serializable types (Interval, Categorical, object) to safe numeric/string
        result = _to_parquet_safe(result)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".parquet", delete=False, prefix="feature_forge_sandbox_"
        ) as temp_file:
            result.to_parquet(temp_file.name)
            response_queue.put(("ok", temp_file.name))
    except Exception as exc:  # pragma: no cover - subprocess path
        response_queue.put(("error", f"Feature generation execution failed: {exc}"))


def _apply_resource_limits(max_memory_mb: int) -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-Unix platforms
        return

    if max_memory_mb > 0:
        max_bytes = max_memory_mb * 1024 * 1024
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        if current_hard > 0:
            max_bytes = min(max_bytes, current_hard)
        if current_soft > 0 and max_bytes > current_soft:
            max_bytes = current_soft
        try:
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        except (OSError, ValueError):  # pragma: no cover - platform-specific
            # Best effort only; timeout still protects runaway execution.
            return
