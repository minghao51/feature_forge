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
import pickle
import socket
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from feature_forge.exceptions import (
    CodeExecutionError,
    SandboxTimeoutError,
    SandboxValidationError,
)
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)

BANNED_IMPORTS: set[str] = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "telnetlib",
    "pickle",
    "ctypes",
}
"""Canonical denylist of import roots blocked during code validation.

This is the single source of truth for the pre-execution AST check in
:mod:`feature_forge.methods.malmas.pipeline.codegen` and any other caller.
The runtime enforcement seam is :class:`SandboxedExecutor` (which uses the
stricter allowlist ``ALLOWED_IMPORTS`` = {pandas, numpy, math}).
"""


def _get_worker_context() -> Any:
    """Choose the multiprocessing start context for sandbox workers.

    Linux: ``fork``. The child inherits the parent's already-imported modules
    (CPython, pandas, numpy, pyarrow) via copy-on-write, so worker startup is
    near-instant (~1ms) instead of the 3-5s ``spawn`` takes to re-bootstrap a
    fresh interpreter and re-import the world. On a 921-test CI suite this is
    the difference between a 3-minute run and a 30+ minute run. Memory
    accumulation is also lower because forked children share read-only pages
    with the parent rather than each loading their own copy of the shared
    libraries.

    macOS / Windows / other: ``spawn``. ``fork`` is deprecated on macOS since
    Python 3.8 (unsafe with threads) and unavailable on Windows, so we fall
    back to the slower-but-portable ``spawn`` there.

    Security note: ``fork`` is safe here because (1) the sandbox already does
    AST-based static validation of the user code before execution, (2) the
    ``exec()`` call runs in a restricted globals namespace with a blocked
    ``__builtins__`` and import hook, and (3) the parent process does not hold
    any privileged locks at the point of forking. The process-level isolation
    (separate PID, RLIMIT_AS, timeout, blocked network/IO) is identical under
    both start methods.
    """
    if sys.platform.startswith("linux"):
        return mp.get_context("fork")
    return mp.get_context("spawn")


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
        "os",
        "sys",
        "signal",
        "subprocess",
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
        ctx = _get_worker_context()
        # Use a Pipe (two FDs, no kernel semaphores) rather than a Queue.
        # Each mp.Queue allocates 3 SemLocks from /dev/shm and spawns a feeder
        # thread. Across a long test suite on a CI runner whose semaphore/FD
        # budget is tight, Queue creation starts failing with
        # `OSError: [Errno 12] Cannot allocate memory` at SemLock.__init__.
        # The worker only ever sends one response message and exits, so a
        # single-shot pipe is both lighter and a closer fit.
        parent_conn, worker_conn = ctx.Pipe(duplex=False)
        input_path = _write_ipc_frame(df, prefix="feature_forge_input_")
        proc = ctx.Process(
            target=_sandbox_worker_main,
            args=(
                code,
                input_path,
                self.limits.max_memory_mb,
                worker_conn,
                source,
                agent_name,
            ),
            daemon=True,
        )
        # The parent never writes. Close the worker end in the parent only
        # AFTER proc.start() — under spawn the fd is transmitted via pickle
        # to the child, under fork the child inherits a copy at fork time.
        # Closing earlier would either invalidate the handle being pickled
        # (spawn) or leave a redundant writer end open in the parent (fork,
        # which would prevent the parent's recv() from seeing EOF when the
        # child exits).
        proc.start()
        worker_conn.close()
        artifact_path = ""
        try:
            # Pipe has no native recv-with-timeout, so poll with the configured
            # deadline. poll() returns True on data OR when the write end is
            # closed (worker exited) — in the latter case recv() raises EOFError.
            if not parent_conn.poll(self.limits.timeout_seconds):
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)
                logger.error("sandbox_timeout", timeout_seconds=self.limits.timeout_seconds)
                raise SandboxTimeoutError(
                    f"Sandbox execution timed out after {self.limits.timeout_seconds:.1f}s"
                )
            try:
                status, payload = parent_conn.recv()
            except EOFError as exc:
                # Worker died without sending anything — typically an interpreter
                # crash during exec() or a hard OOM kill. Treat like a timeout.
                logger.error(
                    "sandbox_worker_eof",
                    timeout_seconds=self.limits.timeout_seconds,
                    process_exitcode=proc.exitcode,
                )
                raise SandboxTimeoutError(
                    f"Sandbox worker exited without responding "
                    f"(exitcode={proc.exitcode}) after {self.limits.timeout_seconds:.1f}s"
                ) from exc

            # Bound the total wait so a worker that hangs after a partial
            # response still hits the configured timeout.
            deadline = time.monotonic() + self.limits.timeout_seconds
            remaining = max(0.0, deadline - time.monotonic())
            proc.join(timeout=remaining)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)

            if status == "ok":
                artifact_path = payload
                result = _read_ipc_frame(artifact_path)
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
            # Close both ends of the pipe so FDs are reclaimed immediately.
            # (worker_conn was already closed in the parent above; this is a
            # no-op there but covers the child-end lifecycle defensively.)
            for conn in (parent_conn, worker_conn):
                try:
                    conn.close()
                except (OSError, ValueError):
                    pass
            for path in (artifact_path, input_path):
                if not path:
                    continue
                # Each IPC frame may have a pickle sidecar (written by
                # _write_ipc_frame) that also needs cleanup.
                for candidate in (path, path + ".pkl"):
                    try:
                        os.unlink(candidate)
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


def _write_ipc_frame(df: pd.DataFrame, prefix: str) -> str:
    """Write ``df`` for cross-process transport and return the primary path.

    Always writes a pickle sidecar alongside the parquet frame when parquet
    succeeds, and writes only pickle when parquet fails. The reader
    (:func:`_read_ipc_frame`) prefers parquet and falls back to the sidecar.

    Rationale: on some CI runners (notably ubuntu-24.04 under memory
    pressure) ``pyarrow._parquet.so`` imports cleanly in the parent process
    but fails to mmap inside a ``multiprocessing.spawn`` worker. Because the
    parent's parquet write succeeds, a write-time-only fallback would leave
    the worker nothing to fall back to. Emitting both formats up front keeps
    the sandbox functional at the cost of one extra temp file.
    """
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".parquet", delete=False, prefix=prefix)
    tmp.close()
    path = tmp.name
    pickle_sidecar = path + ".pkl"
    try:
        df.to_parquet(path)
    except Exception as exc:
        logger.warning("sandbox_ipc_parquet_write_failed", error=str(exc), fallback="pickle")
        try:
            os.unlink(path)
        except OSError:
            pass
        with open(pickle_sidecar, "wb") as fh:
            pickle.dump(df, fh)
        return pickle_sidecar
    # Parquet write succeeded in the parent — also write a pickle sidecar so
    # a worker that cannot mmap pyarrow._parquet.so can still read the frame.
    try:
        with open(pickle_sidecar, "wb") as fh:
            pickle.dump(df, fh)
    except OSError as exc:
        logger.warning("sandbox_ipc_pickle_sidecar_write_failed", error=str(exc))
    return path


def _read_ipc_frame(path: str) -> pd.DataFrame:
    """Read a cross-process DataFrame written by :func:`_write_ipc_frame`.

    Dispatches on suffix and tolerates a parquet path whose engine fails at
    read time by retrying an adjacent ``.pkl`` sidecar when present.
    """
    if path.endswith(".pkl"):
        with open(path, "rb") as fh:
            return cast("pd.DataFrame", pickle.load(fh))  # trusted parent/worker path
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        pickle_sidecar = path + ".pkl"
        if os.path.exists(pickle_sidecar):
            logger.warning("sandbox_ipc_parquet_read_failed", error=str(exc), fallback="pickle")
            with open(pickle_sidecar, "rb") as fh:
                return cast("pd.DataFrame", pickle.load(fh))  # trusted parent/worker path
        raise


def _sandbox_worker_main(
    code: str,
    input_parquet_path: str,
    max_memory_mb: int,
    response_conn: mp.connection.Connection,
    source: str = "unknown",
    agent_name: str = "unknown",
) -> None:
    # NOTE: do NOT call _apply_resource_limits at the top of the spawn worker.
    # RLIMIT_AS caps the process's *virtual address space*, which on Linux must
    # also cover the CPython interpreter, pandas/numpy/pyarrow shared-library
    # mmaps, and thread stacks. A 512MB cap (the default) is below the cost of
    # merely importing pandas inside a fresh spawn worker on Python 3.13, so
    # applying it before imports causes MemoryError / RuntimeError("can't start
    # new thread") and the parent's timeout fires. We instead scope RLIMIT_AS
    # to wrap only the exec() of user code below.
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]

    def _respond(payload: tuple[str, str]) -> None:
        """Send one response over the pipe and close it. Best-effort: a broken
        pipe (parent gone) is treated as success since the parent's timeout
        already owns the failure path."""
        try:
            response_conn.send(payload)
        except (OSError, BrokenPipeError):
            pass
        finally:
            try:
                response_conn.close()
            except (OSError, ValueError):
                pass

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

    class _BlockedSocket:
        """Socket shim that blocks any runtime network usage in sandbox."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _blocked_network()

        def connect(self, *_args: Any, **_kwargs: Any) -> Any:
            return _blocked_network()

        def send(self, *_args: Any, **_kwargs: Any) -> Any:
            return _blocked_network()

        def sendall(self, *_args: Any, **_kwargs: Any) -> Any:
            return _blocked_network()

        def recv(self, *_args: Any, **_kwargs: Any) -> Any:
            return _blocked_network()

        def close(self) -> None:
            return None

    try:
        df = _read_ipc_frame(input_parquet_path)
    except Exception as exc:
        _respond(("error", f"Failed to read input data: {exc}"))
        return

    socket.create_connection = _blocked_network
    socket.socket = _BlockedSocket  # type: ignore[assignment,misc]

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
        with _memory_limit_ctx(max_memory_mb):
            exec(compile(code, filename="<sandbox>", mode="exec"), safe_globals, local_vars)
            generate_features = local_vars.get("generate_features")
            if generate_features is None:
                _respond(("error", "Code must define a 'generate_features(df)' function"))
                return
            result = generate_features(df)
        if not isinstance(result, pd.DataFrame):
            _respond(
                ("error", f"generate_features must return a DataFrame, got {type(result).__name__}")
            )
            return
        # Convert non-serializable types (Interval, Categorical, object) to safe numeric/string
        result = _to_parquet_safe(result)
        tmp_name = _write_ipc_frame(result, prefix="ff_sandbox_")
        _respond(("ok", tmp_name))
    except BaseException as exc:  # pragma: no cover - subprocess path
        _respond(("error", f"Feature generation execution failed: {exc}"))


def _apply_resource_limits(max_memory_mb: int) -> None:
    """Apply RLIMIT_AS for the calling process.

    Kept for backwards compatibility with tests that exercise the limit
    arithmetic directly. The sandbox worker itself uses
    :func:`_memory_limit_ctx` so the limit is scoped to user-code execution
    rather than the entire spawn worker bootstrap (see comment in
    :func:`_sandbox_worker_main`).
    """
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


def _current_vmsize_bytes() -> int:
    """Best-effort current virtual address space of this process in bytes.

    Used to decide whether an RLIMIT_AS cap is even physically achievable.
    On Linux this is VmSize from ``/proc/self/status``; on other platforms
    we fall back to 0 (meaning "unknown — assume any cap is safe to try").
    """
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmSize:"):
                    # "VmSize:\t   12345 kB\n"
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
                    break
    except (OSError, ValueError):
        pass
    return 0


@contextmanager
def _memory_limit_ctx(max_memory_mb: int) -> Iterator[None]:
    """Scope RLIMIT_AS around the body, then restore prior limits.

    On Linux, ``RLIMIT_AS`` caps the process's *total* virtual address space,
    not just future allocations — so once CPython, pandas, numpy and pyarrow
    have been imported (≈1GB of VmSize in a fresh Python 3.13 spawn worker on
    ubuntu-24.04), any cap below ~1.2GB makes the next ``pd.DataFrame(...)``
    fail with ``MemoryError``. We therefore skip the cap whenever the target
    is at or below the current VmSize, and additionally use a sizeable
    (32MB) canary allocation to catch the case where VmSize is unknown or
    grew during the call. The parent's timeout still bounds runaway code.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - non-Unix platforms
        yield
        return

    if max_memory_mb <= 0:
        yield
        return

    target = max_memory_mb * 1024 * 1024

    # Skip when the cap is physically unreachable given what's already mapped.
    current_vmsize = _current_vmsize_bytes()
    if current_vmsize > 0 and target <= current_vmsize:
        logger.debug(
            "sandbox_memory_cap_skipped_below_vmsize",
            target_bytes=target,
            current_vmsize_bytes=current_vmsize,
        )
        yield
        return

    soft_before, hard_before = resource.getrlimit(resource.RLIMIT_AS)
    capped_hard = hard_before if hard_before > 0 and hard_before < target else target
    capped_soft = min(target, capped_hard) if capped_hard > 0 else target
    if soft_before > 0 and capped_soft > soft_before:
        capped_soft = soft_before
    if capped_soft <= 0 or (hard_before > 0 and capped_soft > hard_before):
        yield
        return

    try:
        resource.setrlimit(resource.RLIMIT_AS, (capped_soft, capped_hard))
    except (OSError, ValueError):  # pragma: no cover - platform-specific
        yield
        return

    # Canary: a DataFrame-shaped allocation is a good proxy for "can the user
    # code actually run under this cap". 32MB is roughly the smallest
    # allocation that user feature code would plausibly make, and large
    # enough to fail when the cap is already saturated by the interpreter.
    try:
        _canary = bytearray(32 * 1024 * 1024)
        del _canary
    except MemoryError:
        logger.debug(
            "sandbox_memory_cap_canary_failed",
            target_bytes=target,
            current_vmsize_bytes=current_vmsize,
        )
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft_before, hard_before))
        except (OSError, ValueError):
            pass
        yield
        return

    try:
        yield
    finally:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft_before, hard_before))
        except (OSError, ValueError):  # pragma: no cover - platform-specific
            pass
