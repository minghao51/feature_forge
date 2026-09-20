"""Sandboxed code execution for LLM-generated feature engineering code.

Security model:
1. AST static validation blocks dangerous syntax/name patterns.
2. Execution happens in a dedicated worker process (not in-process).
3. Worker has bounded timeout and optional memory limits.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import ctypes
import errno
import hashlib
import logging
import multiprocessing as mp
import os
import platform
import queue
import signal
import socket
import struct
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from feature_forge.config import SandboxProfile
from feature_forge.exceptions import (
    CodeExecutionError,
    SandboxContainmentError,
    SandboxTimeoutError,
    SandboxValidationError,
)
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)

_WORKER_LOGGER_NAME = "feature_forge.evaluation.sandbox.worker"


def _worker_log_event(level: int, event: str, **fields: Any) -> None:
    """Emit a worker-process log event through the stdlib logger only.

    structlog's processor chain lazily imports OpenTelemetry on the first
    event; inside the RLIMIT_AS-capped worker that import can exhaust the
    remaining address space so the response queue's feeder thread fails to
    start (``RuntimeError: can't start new thread``, observed while
    qualifying plan 23 PR 5 on Python 3.13 with single-thread BLAS/OpenMP
    env). The worker therefore never emits structlog events; every
    worker-reported fact travels back in the response message and the
    parent logs it.
    """
    worker_logger = logging.getLogger(_WORKER_LOGGER_NAME)
    if not worker_logger.isEnabledFor(level):
        return
    rendered = " ".join(f"{key}={str(value)[:100]!r}" for key, value in fields.items())
    worker_logger.log(level, "%s %s", event, rendered)


def _containment_fields(containment: _SandboxContainment) -> dict[str, Any]:
    """Secret-free summary of the worker's actually-installed containment."""
    return {
        "sandbox_profile": containment.profile.value,
        "sandbox_degraded": containment.degraded,
        "sandbox_mechanism": containment.mechanism,
        "sandbox_landlock_abi": containment.landlock_abi,
    }


# Linux Landlock UAPI. Raw ctypes keeps the containment boundary dependency-free.
_LANDLOCK_CREATE_RULESET_NR = 444
_LANDLOCK_ADD_RULE_NR = 445
_LANDLOCK_RESTRICT_SELF_NR = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_PR_GET_SECCOMP = 21
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000

_LANDLOCK_FS_RIGHTS: dict[str, tuple[int, int]] = {
    "execute": (1 << 0, 1),
    "write_file": (1 << 1, 1),
    "read_file": (1 << 2, 1),
    "read_dir": (1 << 3, 1),
    "remove_dir": (1 << 4, 1),
    "remove_file": (1 << 5, 1),
    "make_char": (1 << 6, 1),
    "make_dir": (1 << 7, 1),
    "make_reg": (1 << 8, 1),
    "make_sock": (1 << 9, 1),
    "make_fifo": (1 << 10, 1),
    "make_block": (1 << 11, 1),
    "make_sym": (1 << 12, 1),
    "refer": (1 << 13, 2),
    "truncate": (1 << 14, 3),
    "ioctl_dev": (1 << 15, 5),
}
_LANDLOCK_NET_RIGHTS: dict[str, tuple[int, int]] = {
    "bind_tcp": (1 << 0, 4),
    "connect_tcp": (1 << 1, 4),
}
_SOCKET_SYSCALLS: dict[str, tuple[int, ...]] = {
    "x86_64": (41, 42, 49, 53),
    "aarch64": (198, 203, 200, 199),
    "arm64": (198, 203, 200, 199),
}


@dataclass(frozen=True)
class _SandboxContainment:
    profile: SandboxProfile
    degraded: bool
    mechanism: str
    landlock_abi: int | None


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _syscall(libc: ctypes.CDLL, number: int, *args: int) -> int:
    libc.syscall.restype = ctypes.c_long
    libc.syscall.argtypes = [ctypes.c_long] + [ctypes.c_long] * len(args)
    return int(libc.syscall(number, *args))


def _probe_landlock_abi() -> int | None:
    """Return the current process's Landlock ABI, or ``None`` if unavailable."""
    if platform.system() != "Linux" or platform.machine() not in _SOCKET_SYSCALLS:
        return None
    libc = _libc()
    result = _syscall(libc, _LANDLOCK_CREATE_RULESET_NR, 0, 0, _LANDLOCK_CREATE_RULESET_VERSION)
    return int(result) if result >= 1 else None


def _probe_seccomp() -> bool:
    """Probe classic seccomp without changing process state."""
    if platform.system() != "Linux" or platform.machine() not in _SOCKET_SYSCALLS:
        return False
    libc = _libc()
    libc.prctl.restype = ctypes.c_int
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    result = libc.prctl(_PR_GET_SECCOMP, 0, 0, 0, 0)
    return result >= 0 or ctypes.get_errno() != errno.EINVAL


def _landlock_fs_access(abi: int) -> dict[str, int]:
    """Return only FS rights understood by the detected ABI."""
    return {
        name: bit for name, (bit, minimum_abi) in _LANDLOCK_FS_RIGHTS.items() if abi >= minimum_abi
    }


def _landlock_net_access(abi: int) -> dict[str, int]:
    """Return network rights only when the runtime ABI supports them."""
    return {
        name: bit for name, (bit, minimum_abi) in _LANDLOCK_NET_RIGHTS.items() if abi >= minimum_abi
    }


def _runtime_read_roots() -> list[str]:
    """Find narrowly scoped directories containing trusted runtime modules."""
    candidates = [
        sys.prefix,
        os.path.dirname(os.__file__),
        os.path.dirname(np.__file__),
        os.path.dirname(pd.__file__),
    ]
    roots: list[str] = []
    for candidate in candidates:
        path = os.path.realpath(candidate)
        if os.path.isdir(path) and path not in roots:
            roots.append(path)
    return roots


def _landlock_path_rule(libc: ctypes.CDLL, ruleset_fd: int, path: str, mask: int) -> None:
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        # struct landlock_path_beneath_attr is __attribute__((packed)):
        # { __u64 allowed_access; __s32 parent_fd; } — 12 bytes, no padding.
        attr = struct.pack("=Qi", mask, fd)
        buffer = (ctypes.c_char * len(attr)).from_buffer_copy(attr)
        result = _syscall(
            libc,
            _LANDLOCK_ADD_RULE_NR,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.addressof(buffer),
            0,
        )
        if result != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule({path}) failed")
    finally:
        os.close(fd)


def _install_seccomp_network_filter() -> None:
    """Install a raw classic-BPF socket/TCP denial filter."""
    syscall_numbers = _SOCKET_SYSCALLS.get(platform.machine())
    if syscall_numbers is None:
        raise SandboxContainmentError("seccomp fallback is unsupported on this architecture")

    class _SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class _SockFprog(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]

    # BPF_LD | BPF_W | BPF_ABS loads seccomp_data.nr (offset 0).
    instructions: list[tuple[int, int, int, int]] = [(0x20, 0, 0, 0)]
    denied = _SECCOMP_RET_ERRNO | errno.EACCES
    for syscall_number in syscall_numbers:
        instructions.extend([(0x15, 0, 1, syscall_number), (0x06, 0, 0, denied)])
    instructions.append((0x06, 0, 0, _SECCOMP_RET_ALLOW))
    compiled = (_SockFilter * len(instructions))(*[_SockFilter(*item) for item in instructions])
    program = _SockFprog(len(compiled), compiled)
    libc = _libc()
    libc.prctl.restype = ctypes.c_int
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.addressof(program), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP network filter failed")


def _apply_landlock_containment(input_path: str, output_path: str, abi: int) -> str:
    """Install a deny-by-default Landlock policy for exact temp paths."""
    fs = _landlock_fs_access(abi)
    net = _landlock_net_access(abi)
    handled_fs = 0
    for bit in fs.values():
        handled_fs |= bit
    handled_net = 0
    for bit in net.values():
        handled_net |= bit
    attr = struct.pack("=QQQ", handled_fs, handled_net, 0)
    libc = _libc()
    attr_buffer = (ctypes.c_char * len(attr)).from_buffer_copy(attr)
    ruleset_fd = _syscall(
        libc,
        _LANDLOCK_CREATE_RULESET_NR,
        ctypes.addressof(attr_buffer),
        len(attr),
        0,
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        _landlock_path_rule(libc, ruleset_fd, input_path, fs["read_file"])
        _landlock_path_rule(
            libc,
            ruleset_fd,
            output_path,
            fs["write_file"] | fs.get("truncate", 0),
        )
        read_exec = fs["read_file"] | fs["read_dir"] | fs["execute"]
        for root in _runtime_read_roots():
            _landlock_path_rule(libc, ruleset_fd, root, read_exec)
        libc.prctl.restype = ctypes.c_int
        libc.prctl.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        ]
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
        if _syscall(libc, _LANDLOCK_RESTRICT_SELF_NR, ruleset_fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)
    return "landlock" if net else "landlock+seccomp"


def _apply_strict_containment(input_path: str, output_path: str) -> _SandboxContainment:
    """Install strict containment, failing closed on every unsupported path."""
    abi = _probe_landlock_abi()
    if abi is None:
        raise SandboxContainmentError("strict sandbox requires an available Landlock ABI")
    try:
        mechanism = _apply_landlock_containment(input_path, output_path, abi)
        if abi < 4:
            if not _probe_seccomp():
                raise SandboxContainmentError(
                    "strict sandbox requires seccomp for Landlock ABI below 4"
                )
            _install_seccomp_network_filter()
            mechanism = "landlock+seccomp"
        return _SandboxContainment(SandboxProfile.STRICT, False, mechanism, abi)
    except SandboxContainmentError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise SandboxContainmentError(f"strict sandbox containment failed: {exc}") from exc


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
                _worker_log_event(
                    logging.DEBUG,
                    "parquet_conversion_numeric_extract_failed",
                    column=col,
                    error=str(exc),
                )
                df[col] = series.astype(str)
        elif series.dtype == object:
            try:
                df[col] = pd.to_numeric(series, errors="coerce")
                if df[col].isna().all() and series.notna().any():
                    df[col] = series.astype(str)
            except Exception as exc:
                _worker_log_event(
                    logging.DEBUG, "parquet_conversion_coerce_failed", column=col, error=str(exc)
                )
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


@dataclass
class _WorkerHandle:
    """Parent-owned resources for one sandbox worker."""

    process: mp.Process
    response_queue: Any
    input_path: str
    output_path: str
    artifact_path: str = ""


_ACTIVE_HANDLES: set[int] = set()
"""Live parent-owned worker handles (process + queue + temp paths).

Cleanup removes its handle id; leak tests assert this registry is empty so
queue/process cleanup is observably complete, not merely non-raising.
"""


def active_worker_handle_count() -> int:
    """Number of live parent-owned sandbox worker handles (test seam)."""
    return len(_ACTIVE_HANDLES)


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
    # These are file-capable APIs on the two libraries exposed to generated
    # code.  Keep this an explicit list rather than blocking every ``read_*``
    # or ``to_*`` method: ordinary dataframe/numeric operations must remain
    # usable.  Names are normalized to lower case by the AST checker.
    BLOCKED_IO_ATTRS: ClassVar[set[str]] = {
        # NumPy file, memory-map, and DataSource APIs.
        "datasource",
        "fromfile",
        "fromregex",
        "genfromtxt",
        "load",
        "loadtxt",
        "memmap",
        "npzfile",
        "open_memmap",
        "read_array",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "tofile",
        "write_array",
        "write_array_header_1_0",
        "write_array_header_2_0",
        "write_array_header_3_0",
        # Pandas readers.
        "read_clipboard",
        "read_csv",
        "read_excel",
        "read_feather",
        "read_fwf",
        "read_gbq",
        "read_hdf",
        "read_html",
        "read_json",
        "read_orc",
        "read_parquet",
        "read_pickle",
        "read_sas",
        "read_spss",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "read_stata",
        "read_table",
        "read_xml",
        # Pandas writers.  ``to_numpy`` and other in-memory conversions are
        # intentionally absent.
        "to_clipboard",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_html",
        "to_json",
        "to_latex",
        "to_markdown",
        "to_orc",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "to_stata",
        "to_xml",
    }
    # Full paths provide an auditable canonical policy for the public APIs;
    # terminal-name matching below additionally catches module aliases and
    # nested paths such as ``np.lib.format.open_memmap``.
    BLOCKED_IO_PATHS: ClassVar[set[str]] = set()
    for _module in ("np", "numpy", "pd", "pandas"):
        for _name in BLOCKED_IO_ATTRS:
            BLOCKED_IO_PATHS.add(f"{_module}.{_name.lower()}")
    del _module, _name
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
        max_memory_mb: int = 2048,
        profile: SandboxProfile | str = SandboxProfile.STRICT,
    ) -> None:
        self.limits = SandboxLimits(timeout_seconds=timeout_seconds, max_memory_mb=max_memory_mb)
        self.profile = SandboxProfile(profile)

    @property
    def provenance(self) -> dict[str, Any]:
        """Secret-free resolved profile metadata for logs and run provenance."""
        degraded = self.profile is SandboxProfile.DEGRADED_DEVELOPMENT
        abi = _probe_landlock_abi() if not degraded else None
        if degraded:
            mechanism = "process_only_ast"
        elif abi is None:
            # Strict execution will fail closed; never advertise a
            # containment mechanism that is not actually installable.
            mechanism = "unavailable"
        elif abi >= 4:
            mechanism = "landlock"
        else:
            mechanism = "landlock+seccomp"
        return {
            "sandbox_profile": self.profile.value,
            "sandbox_degraded": degraded,
            "sandbox_landlock_abi": abi,
            "sandbox_mechanism": mechanism,
        }

    def execute(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
    ) -> pd.DataFrame:
        """Execute feature generation code safely in a bounded worker."""
        execute_t0 = time.perf_counter()
        deadline = self._execution_deadline()
        code_hash = self._log_execution_start(code, df, source, agent_name)
        tree = self._parse_and_validate(code)
        payload = ast.unparse(tree) if hasattr(ast, "unparse") else code
        result = self._execute_in_worker(
            payload, df, source=source, agent_name=agent_name, deadline=deadline
        )
        self._log_execution_complete(result, execute_t0, source, agent_name, code_hash)
        return result

    async def execute_async(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
    ) -> pd.DataFrame:
        """Execute code without delegating process ownership to an executor thread.

        The worker is started synchronously on the event-loop thread.  Once it
        has started, queue reads are non-blocking and yield to the loop so a
        caller can cancel this coroutine and trigger the same cleanup path as a
        timeout.
        """
        execute_t0 = time.perf_counter()
        deadline = self._execution_deadline()
        code_hash = self._log_execution_start(code, df, source, agent_name)
        tree = self._parse_and_validate(code)
        payload = ast.unparse(tree) if hasattr(ast, "unparse") else code
        result = await self._execute_in_worker_async(
            payload, df, source=source, agent_name=agent_name, deadline=deadline
        )
        self._log_execution_complete(result, execute_t0, source, agent_name, code_hash)
        return result

    def _log_execution_start(
        self, code: str, df: pd.DataFrame, source: str, agent_name: str
    ) -> str:
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        logger.info(
            "sandbox_execute_start",
            code_length=len(code),
            input_shape=df.shape,
            source=source,
            agent=agent_name,
            code_hash=code_hash,
            effective_timeout_seconds=max(0.0, self.limits.timeout_seconds),
            **self.provenance,
        )
        return code_hash

    def _log_execution_complete(
        self,
        result: pd.DataFrame,
        execute_t0: float,
        source: str,
        agent_name: str,
        code_hash: str,
    ) -> None:
        logger.info(
            "sandbox_execute_complete",
            result_shape=result.shape,
            latency_ms=round((time.perf_counter() - execute_t0) * 1000, 1),
            source=source,
            agent=agent_name,
            code_hash=code_hash,
            **self.provenance,
        )

    def _execution_deadline(self) -> float:
        return time.monotonic() + max(0.0, self.limits.timeout_seconds)

    def _assert_deadline(self, deadline: float) -> None:
        if time.monotonic() >= deadline:
            logger.error(
                "sandbox_timeout",
                timeout_seconds=self.limits.timeout_seconds,
                effective_timeout_seconds=max(0.0, self.limits.timeout_seconds),
            )
            raise SandboxTimeoutError(
                f"Sandbox execution timed out after {self.limits.timeout_seconds:.1f}s"
            )

    def _execute_in_worker(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
        deadline: float | None = None,
    ) -> pd.DataFrame:
        execution_deadline = self._execution_deadline() if deadline is None else deadline
        handle = self._start_worker(code, df, source, agent_name, execution_deadline)
        deadline = execution_deadline
        try:
            status, payload, containment = self._wait_for_response(handle, deadline)
            result = self._consume_response(
                handle, status, payload, containment, deadline, source=source, agent_name=agent_name
            )
            if not self._wait_for_exit(handle.process, deadline):
                raise self._timeout_error()
            return result
        finally:
            self._cleanup_worker(handle, deadline)

    async def _execute_in_worker_async(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
        deadline: float | None = None,
    ) -> pd.DataFrame:
        execution_deadline = self._execution_deadline() if deadline is None else deadline
        # This call, including Process.start(), intentionally runs on the
        # event-loop thread; process ownership never moves to a worker thread.
        handle = self._start_worker(code, df, source, agent_name, execution_deadline)
        deadline = execution_deadline
        cleaned = False
        try:
            status, payload, containment = await self._poll_response(handle, deadline)
            result = self._consume_response(
                handle, status, payload, containment, deadline, source=source, agent_name=agent_name
            )
            if not await self._wait_for_exit_async(handle.process, deadline):
                raise self._timeout_error()
            return result
        except asyncio.CancelledError:
            # Cleanup is synchronous but has only bounded process/queue/file
            # operations; importantly, no executor thread survives cancellation.
            self._cleanup_worker(handle, deadline, force=True)
            cleaned = True
            raise
        finally:
            if not cleaned:
                self._cleanup_worker(handle, deadline)

    def _start_worker(
        self,
        code: str,
        df: pd.DataFrame,
        source: str,
        agent_name: str,
        deadline: float,
    ) -> _WorkerHandle:
        self._ensure_strict_available()
        ctx = mp.get_context("spawn")
        response_queue: Any = ctx.Queue(maxsize=1)
        input_path = ""
        output_path = ""
        process: Any = None
        try:
            # Phase boundaries are checkpointed against the authoritative
            # deadline: input serialization and output predeclaration are
            # fast bounded writes, and ``Process.start()`` is fork/exec plus a
            # pipe write that cannot be interrupted from this thread without
            # reintroducing an executor thread; each phase is followed by a
            # deadline assertion so overrun is reported, never silently
            # absorbed (ADR 0019 decision 5).
            self._assert_deadline(deadline)
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".parquet", delete=False, prefix="feature_forge_input_"
            ) as input_file:
                input_path = input_file.name
                df.to_parquet(input_path)
            self._assert_deadline(deadline)
            # Predeclare the output inode before Landlock is installed in the
            # child so the child can publish exactly this file.
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".parquet", delete=False, prefix="ff_sandbox_"
            ) as output_file:
                output_path = output_file.name
            self._assert_deadline(deadline)
            process = ctx.Process(
                target=_sandbox_worker_main,
                args=(
                    code,
                    input_path,
                    output_path,
                    self.limits.max_memory_mb,
                    response_queue,
                    source,
                    agent_name,
                    self.profile.value,
                ),
                daemon=True,
            )
            # Direct call is intentional: async ownership starts here, on the
            # caller's event-loop/main thread.
            process.start()
            self._assert_deadline(deadline)
            handle = _WorkerHandle(
                cast("mp.Process", process), response_queue, input_path, output_path
            )
            _ACTIVE_HANDLES.add(id(handle))
            return handle
        except BaseException:
            if process is not None:
                self._kill_worker(cast("mp.Process", process), hard=True)
                try:
                    process.join(timeout=0.5)
                except (AssertionError, OSError, ValueError):
                    pass
            self._close_response_queue(response_queue)
            self._remove_paths(input_path, output_path)
            raise

    def _ensure_strict_available(self) -> None:
        if self.profile is SandboxProfile.STRICT:
            # Fail before launching a worker when the host cannot provide the
            # production boundary; the worker repeats the probe to catch races
            # and container policy changes, then reports failure explicitly.
            abi = _probe_landlock_abi()
            if abi is None or (abi < 4 and not _probe_seccomp()):
                raise SandboxContainmentError(
                    "strict sandbox is unavailable on this host; select "
                    "degraded_development explicitly for local development"
                )

    def _timeout_error(self) -> SandboxTimeoutError:
        return SandboxTimeoutError(
            f"Sandbox execution timed out after {self.limits.timeout_seconds:.1f}s"
        )

    def _wait_for_response(
        self, handle: _WorkerHandle, deadline: float
    ) -> tuple[str, str, dict[str, Any]]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._timeout_error()
        try:
            return cast(
                "tuple[str, str, dict[str, Any]]", handle.response_queue.get(timeout=remaining)
            )
        except queue.Empty as exc:
            raise self._timeout_error() from exc
        except EOFError as exc:
            # A worker killed before posting (e.g. the RLIMIT_AS feeder-thread
            # failure mode in _worker_log_event) closes the queue write-end
            # with no item; surface it through the typed hierarchy instead of
            # a raw EOFError that would bypass callers' CodeExecutionError
            # handling.
            raise CodeExecutionError("sandbox worker died before reporting a result") from exc

    async def _poll_response(
        self, handle: _WorkerHandle, deadline: float
    ) -> tuple[str, str, dict[str, Any]]:
        while True:
            try:
                return cast("tuple[str, str, dict[str, Any]]", handle.response_queue.get_nowait())
            except queue.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._timeout_error() from None
                await asyncio.sleep(min(0.01, remaining))
            except EOFError as exc:
                raise CodeExecutionError("sandbox worker died before reporting a result") from exc

    def _consume_response(
        self,
        handle: _WorkerHandle,
        status: str,
        payload: str,
        containment: dict[str, Any],
        deadline: float,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
    ) -> pd.DataFrame:
        if status == "ok":
            if os.path.realpath(payload) != os.path.realpath(handle.output_path):
                raise CodeExecutionError("Sandbox worker returned an invalid output artifact")
            # Only record the parent-created output path for cleanup.  Never
            # unlink an arbitrary path supplied by the worker response.
            handle.artifact_path = handle.output_path
            # Worker-reported containment is logged here in the parent: the
            # worker itself must not emit structlog/OTel events (see
            # _worker_log_event).
            logger.info(
                "sandbox_worker_containment",
                source=source,
                agent=agent_name,
                **containment,
            )
            self._assert_deadline(deadline)
            result = pd.read_parquet(handle.output_path)
            self._assert_deadline(deadline)
            if not isinstance(result, pd.DataFrame):
                raise CodeExecutionError(
                    f"generate_features must return a DataFrame, got {type(result).__name__}"
                )
            return result
        if status == "blocked":
            raise SandboxValidationError(payload[:300])
        if status == "containment_unavailable":
            raise SandboxContainmentError(payload[:300])
        raise CodeExecutionError(payload[:300])

    @staticmethod
    def _wait_for_exit(process: mp.Process, deadline: float) -> bool:
        if not process.is_alive():
            return True
        remaining = deadline - time.monotonic()
        if remaining > 0:
            process.join(timeout=remaining)
        return not process.is_alive()

    @staticmethod
    async def _wait_for_exit_async(process: mp.Process, deadline: float) -> bool:
        """Wait for exit without blocking the event loop on ``join``."""
        while process.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.01, remaining))
        return True

    def _cleanup_worker(
        self, handle: _WorkerHandle, deadline: float, *, force: bool = False
    ) -> None:
        process = handle.process
        if force:
            self._kill_worker(process)
        try:
            alive = process.is_alive()
        except AssertionError:
            alive = False
        if alive:
            # A join is always finite.  If the authoritative deadline has
            # passed, retain only a short termination grace period.
            join_timeout = max(0.0, min(0.5, deadline - time.monotonic()))
            process.join(timeout=join_timeout)
        try:
            alive = process.is_alive()
        except AssertionError:
            alive = False
        if alive:
            self._kill_worker(process, hard=True)
            # SIGKILL is terminal but reaping can lag under load: wait it out
            # within a bounded grace window instead of assuming death.
            kill_deadline = time.monotonic() + 1.0
            while process.is_alive() and time.monotonic() < kill_deadline:
                process.join(timeout=min(0.1, max(0.0, kill_deadline - time.monotonic())))
        try:
            if process.is_alive():
                logger.error(
                    "sandbox_worker_terminate_failed",
                    pid=process.pid,
                )
            else:
                process.close()
        except (AssertionError, OSError, ValueError):
            pass
        self._close_response_queue(handle.response_queue)
        self._remove_paths(handle.artifact_path, handle.output_path, handle.input_path)
        _ACTIVE_HANDLES.discard(id(handle))

    @staticmethod
    def _kill_worker(process: mp.Process, *, hard: bool = False) -> None:
        try:
            alive = process.is_alive()
        except AssertionError:
            return
        if not alive:
            return
        if os.name == "posix":
            try:
                # Only signal a group that the child demonstrably owns; this
                # avoids ever killing the parent's process group during launch.
                pid = process.pid
                if pid is not None and os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
                    return
            except (OSError, ProcessLookupError):
                pass
        try:
            if hard and hasattr(process, "kill"):
                process.kill()
            else:
                process.terminate()
        except (OSError, ValueError):
            pass

    @staticmethod
    def _close_response_queue(response_queue: Any) -> None:
        try:
            response_queue.cancel_join_thread()
        except (AttributeError, OSError):
            pass
        try:
            response_queue.close()
        except (AttributeError, OSError):
            pass

    @staticmethod
    def _remove_paths(*paths: str) -> None:
        for path in set(paths):
            if not path:
                continue
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                # One bounded retry; a persistent failure (e.g. EBUSY on a
                # foreign mount) surfaces as a warning rather than silently
                # swallowing a leaked temporary artifact.
                time.sleep(0.01)
                try:
                    os.unlink(path)
                except OSError as exc:
                    logger.warning("sandbox_temp_cleanup_failed", path=path, error=str(exc))

    @staticmethod
    def _normalized_attribute_path(node: ast.Attribute) -> str | None:
        """Return a lowercase dotted path for a simple attribute expression."""
        components: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            components.append(current.attr.lower())
            current = current.value
        if isinstance(current, ast.Name):
            components.append(current.id.lower())
            return ".".join(reversed(components))
        return None

    def _blocked_io_reason(self, node: ast.Attribute) -> str | None:
        """Return a policy reason for a file-capable library attribute."""
        terminal = node.attr.lower()
        path = self._normalized_attribute_path(node)
        if terminal in {name.lower() for name in self.BLOCKED_IO_ATTRS}:
            return terminal
        if path in self.BLOCKED_IO_PATHS:
            return path
        return None

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
                for alias in node.names:
                    imported_name = alias.name.rsplit(".", maxsplit=1)[-1].lower()
                    if alias.name == "*" or imported_name in {
                        name.lower() for name in self.BLOCKED_IO_ATTRS
                    }:
                        logger.warning(
                            "sandbox_validation_blocked",
                            reason=f"blocked_io_import: {node.module}.{alias.name}",
                        )
                        raise SandboxValidationError(
                            f"Blocked file I/O API import: {node.module}.{alias.name}"
                        )
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
                blocked_io = self._blocked_io_reason(node)
                if blocked_io is not None:
                    logger.warning(
                        "sandbox_validation_blocked", reason=f"blocked_io_attr: {blocked_io}"
                    )
                    raise SandboxValidationError(f"Blocked file I/O API usage: {blocked_io}")
                if node.attr in self.BLOCKED_NETWORK_ATTRS:
                    logger.warning(
                        "sandbox_validation_blocked", reason=f"blocked_network_attr: {node.attr}"
                    )
                    raise SandboxValidationError(f"Blocked network API usage: {node.attr}")

        return tree


def _establish_worker_process_group() -> None:
    """Give the worker a private group so termination reaches its descendants."""
    if os.name != "posix":
        return
    try:
        # Make the worker its own process-group leader before any worker
        # operation can spawn a descendant.  The parent verifies this group
        # identity before using killpg, so a failed setup cannot signal itself.
        os.setpgid(0, 0)
        return
    except OSError:
        try:
            os.setsid()
        except OSError:
            # Parent-side process termination remains the portable fallback.
            pass


_WORKER_ENV_ALLOWLIST = frozenset(
    {
        "LANG",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MALLOC_ARENA_MAX",
        "KMP_DUPLICATE_LIB_OK",
    }
)


def _scrub_worker_environment() -> None:
    """Drop every inherited environment variable the worker does not need.

    An explicit allowlist keeps only locale entries and runtime thread /
    allocation cap variables: clearing the thread caps makes the BLAS/OpenMP
    runtimes default to one pool per core, which alone reserves an extra
    ~0.5-1.3 GB of address space and can push the RLIMIT_AS-capped worker
    against its limit so the response queue's feeder thread cannot start
    (found during plan 23 PR 5 qualification on Python 3.13 under
    single-thread BLAS/OpenMP env). Provider credentials and every other
    inherited value (HOME, PATH, TMPDIR included) never reach generated code;
    this function deliberately never reads or logs a value.
    """
    for name in list(os.environ):
        if name not in _WORKER_ENV_ALLOWLIST and not name.startswith("LC_"):
            del os.environ[name]


def _install_library_io_guards(blocked: Callable[..., Any]) -> None:
    """Install best-effort I/O guards inside the worker process.

    AST validation is the authoritative, deterministic check.  These runtime
    patches are defense in depth only: they do not provide filesystem
    isolation and must not be described as such.  Some NumPy extension types
    cannot be monkeypatched; the static policy still covers their syntax.
    """

    def patch(owner: Any, name: str) -> None:
        try:
            if hasattr(owner, name):
                setattr(owner, name, blocked)
        except (AttributeError, TypeError):
            # A read-only extension attribute is still covered by AST policy.
            pass

    for name in SandboxedExecutor.BLOCKED_IO_ATTRS:
        patch(np, name)
        patch(pd, name)

    # Public pandas writers live on these classes rather than on ``pandas``.
    for owner in (pd.DataFrame, pd.Series):
        for name in SandboxedExecutor.BLOCKED_IO_ATTRS:
            if name.startswith("to_"):
                patch(owner, name)

    # NumPy's DataSource and format helpers are nested under ``np.lib`` on
    # supported versions.  Keep this best-effort because those paths vary by
    # NumPy release.
    np_lib = getattr(np, "lib", None)
    for library_owner in (
        np_lib,
        getattr(np_lib, "npyio", None),
        getattr(np_lib, "format", None),
    ):
        if library_owner is None:
            continue
        for name in SandboxedExecutor.BLOCKED_IO_ATTRS:
            patch(library_owner, name)


def _sandbox_worker_main(
    code: str,
    input_parquet_path: str,
    output_parquet_path: str,
    max_memory_mb: int,
    response_queue: mp.Queue[tuple[str, str, dict[str, Any]]],
    source: str = "unknown",
    agent_name: str = "unknown",
    profile: str = SandboxProfile.STRICT.value,
) -> None:
    # Establish containment's termination boundary before any potentially
    # expensive worker phase, then remove inherited secrets before loading data.
    _establish_worker_process_group()
    _scrub_worker_environment()
    _apply_resource_limits(max_memory_mb=max_memory_mb)
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]

    def _blocked_network(*_args: Any, **_kwargs: Any) -> Any:
        _worker_log_event(
            logging.WARNING,
            "sandbox_runtime_blocked",
            category="network",
            reason="network_blocked",
            source=source,
            agent=agent_name,
            code_hash=code_hash,
        )
        raise PermissionError("Network access is blocked in sandbox runtime")

    def _blocked_io(*_args: Any, **_kwargs: Any) -> Any:
        _worker_log_event(
            logging.WARNING,
            "sandbox_runtime_blocked",
            category="filesystem_io",
            reason="library_io_blocked",
            source=source,
            agent=agent_name,
            code_hash=code_hash,
        )
        raise PermissionError("Library file I/O is blocked in sandbox runtime")

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
        # Input loading happens before restrictions because it is trusted
        # sandbox plumbing. The output inode was predeclared by the parent.
        df = pd.read_parquet(input_parquet_path)
    except Exception as exc:
        response_queue.put(("error", f"Failed to read input data: {exc}", {}))
        return

    output_to_parquet = pd.DataFrame.to_parquet
    try:
        resolved_profile = SandboxProfile(profile)
        if resolved_profile is SandboxProfile.STRICT:
            containment = _apply_strict_containment(input_parquet_path, output_parquet_path)
        else:
            containment = _SandboxContainment(
                SandboxProfile.DEGRADED_DEVELOPMENT,
                True,
                "process_only_ast",
                None,
            )
    except SandboxContainmentError as exc:
        response_queue.put(("containment_unavailable", str(exc), {}))
        return

    # Never emit structlog events inside the worker (see _worker_log_event):
    # the containment report travels back in the response and the parent logs
    # it as `sandbox_worker_containment`.
    containment_report = _containment_fields(containment)
    _install_library_io_guards(_blocked_io)

    socket.create_connection = _blocked_network
    socket.socket = _BlockedSocket  # type: ignore[assignment,misc]
    # Landlock (ABI >= 4) cannot deny AF_UNIX sockets, so pairs stay
    # kernel-permitted; block their Python-level construction too.
    socket.socketpair = _blocked_network

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
            response_queue.put(("error", "Code must define a 'generate_features(df)' function", {}))
            return
        result = generate_features(df)
        if not isinstance(result, pd.DataFrame):
            response_queue.put(
                (
                    "error",
                    f"generate_features must return a DataFrame, got {type(result).__name__}",
                    {},
                )
            )
            return
        # Convert non-serializable types (Interval, Categorical, object) to safe numeric/string
        result = _to_parquet_safe(result)
        try:
            output_to_parquet(result, output_parquet_path)
            response_queue.put(("ok", output_parquet_path, containment_report))
        except Exception:
            raise
    except BaseException as exc:  # pragma: no cover - subprocess path
        response_queue.put(("error", f"Feature generation execution failed: {exc}", {}))


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
