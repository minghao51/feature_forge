"""Sandboxed code execution for LLM-generated feature engineering code.

Security model:
1. AST static validation blocks dangerous syntax/name patterns.
2. Execution happens in a dedicated worker process (not in-process).
3. Worker has bounded timeout and optional memory limits.
"""

from __future__ import annotations

import ast
import builtins
import multiprocessing as mp
import os
import queue
import tempfile
import time
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from feature_forge.exceptions import (
    CodeExecutionError,
    SandboxTimeoutError,
    SandboxValidationError,
)
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


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
        "hasattr",
        "type",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "RuntimeError",
        "ZeroDivisionError",
        "OverflowError",
        "Exception",
        "NotImplementedError",
        "StopIteration",
    }
    ALLOWED_IMPORTS: ClassVar[set[str]] = {"pandas", "numpy", "math"}

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_memory_mb: int = 512,
    ) -> None:
        self.limits = SandboxLimits(timeout_seconds=timeout_seconds, max_memory_mb=max_memory_mb)

    def execute(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """Execute feature generation code safely."""
        execute_t0 = time.perf_counter()
        logger.info("sandbox_execute_start", code_length=len(code), input_shape=df.shape)
        tree = self._parse_and_validate(code)
        payload = ast.unparse(tree) if hasattr(ast, "unparse") else code
        result = self._execute_in_worker(payload, df)
        latency_ms = round((time.perf_counter() - execute_t0) * 1000, 1)
        logger.info(
            "sandbox_execute_complete",
            result_shape=result.shape,
            latency_ms=latency_ms,
        )
        return result

    def _execute_in_worker(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        ctx = mp.get_context("spawn")
        response_queue: mp.Queue[tuple[str, str]] = ctx.Queue(maxsize=1)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".parquet", delete=False, prefix="feature_forge_input_"
        ) as input_file:
            df.to_parquet(input_file.name)
            input_path = input_file.name
        proc = ctx.Process(
            target=_sandbox_worker_main,
            args=(code, input_path, self.limits.max_memory_mb, response_queue),
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

        return tree


def _sandbox_worker_main(
    code: str,
    input_parquet_path: str,
    max_memory_mb: int,
    response_queue: mp.Queue[tuple[str, str]],
) -> None:
    _apply_resource_limits(max_memory_mb=max_memory_mb)
    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as exc:
        response_queue.put(("error", f"Failed to read input data: {exc}"))
        return

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
