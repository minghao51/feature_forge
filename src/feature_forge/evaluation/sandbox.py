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
import signal
import socket
import struct
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
    remaining address space and fail before the worker can put its response
    on the pipe (``RuntimeError: can't start new thread``, observed while
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


_MAX_DIAGNOSTIC_MS = 3_600_000.0
"""Upper bound for a plausible phase duration (one hour).

Diagnostic durations are omitted past this bound so a mismatched clock or a
wedged host can never produce a misleading unbounded field.
"""

_MAX_RESPONSE_PAYLOAD_CHARS = 4096
"""Hard cap on the response payload the worker puts on the wire.

Mirrors the parent-side ``payload[:300]`` truncation convention at the source:
generated code can inflate exception text arbitrarily (e.g.
``ValueError("A" * 10_000_000)``), and the parent's post-poll ``recv`` has no
deadline of its own.  Capping here bounds that drain window to one capped
message, so the authoritative deadline plus a single capped transfer is the
total worst case.
"""
_PHASE_NOT_STARTED = -1
"""Worker phase marker value before the worker's first statement has run."""

_WORKER_PHASES: tuple[str, ...] = (
    "bootstrap",
    "containment_setup",
    "input_load",
    "landlock",
    "code_exec",
    "output_publish",
    "result_send",
    "done",
)
"""Ordered worker-side phases exposed through the shared progress marker.

The worker records the phase it is *entering*; a timed-out parent can then
name the in-progress phase without any extra wait, pipe, thread, or semaphore
(ADR 0019 decision 5 keeps one authoritative deadline).  ``bootstrap`` is the
spawn exec + interpreter + module-import window, which only the parent's
spawn stamp can measure because it precedes ``_sandbox_worker_main`` entry.
"""


def _elapsed_ms(start: float, end: float) -> float | None:
    """Milliseconds between two monotonic stamps, or ``None`` when unusable.

    ``None`` covers an unset start marker (``<= 0``) and an implausible
    reading (negative or above :data:`_MAX_DIAGNOSTIC_MS`), so a broken or
    cross-epoch clock degrades to "unknown" instead of a misleading number.
    """
    if start <= 0 or end < start:
        return None
    elapsed = (end - start) * 1000.0
    return round(elapsed, 1) if elapsed <= _MAX_DIAGNOSTIC_MS else None


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
class _WorkerProgress:
    """Lock-free shared progress marker for one worker (diagnostics only).

    ``phase`` holds the index in :data:`_WORKER_PHASES` the worker is
    currently entering and ``stamps`` holds its ``time.monotonic()`` instant
    at each transition.  The parent only reads this while reporting a
    timeout, so no wait, retry, thread, pipe, or semaphore is added to the
    bounded lifecycle (ADR 0019 decision 5).  Every method is best-effort:
    diagnostics must never be able to fail an execution.
    """

    phase: Any
    stamps: Any

    def enter(self, name: str) -> None:
        """Record that *name* is now in progress.  Never raises."""
        try:
            index = _WORKER_PHASES.index(name)
            # Stamp-then-index store order is intentional; readers tolerate
            # reordering by design (an unset stamp makes ``_elapsed_ms``
            # return None, so the field is simply omitted).  Adding a lock
            # here would violate ADR 0019 decision 5 (no new locks/threads
            # in the worker path).
            self.stamps[index] = time.monotonic()
            self.phase.value = index
        except (AttributeError, IndexError, TypeError, ValueError):
            return

    def _current_index(self) -> int:
        try:
            return int(self.phase.value)
        except (AttributeError, TypeError, ValueError):
            return _PHASE_NOT_STARTED

    def phase_latency_ms(self, name: str) -> float | None:
        """Elapsed ms since the worker entered *name* (shared monotonic clock)."""
        try:
            index = _WORKER_PHASES.index(name)
            started = float(self.stamps[index])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        return _elapsed_ms(started, time.monotonic())

    def fields(self) -> dict[str, Any]:
        """Bounded, secret-free worker progress fields for logs and errors."""
        index = self._current_index()
        if index < 0:
            return {"worker_phase": "not_started"}
        index = min(index, len(_WORKER_PHASES) - 1)
        fields: dict[str, Any] = {"worker_phase": _WORKER_PHASES[index]}
        if _WORKER_PHASES[index] != "done":
            # "done" means the worker finished its work; the remaining tail is
            # parent-side join time, already reported as phase_parent_join_ms.
            pending = self.phase_latency_ms(_WORKER_PHASES[index])
            if pending is not None:
                fields["worker_phase_elapsed_ms"] = pending
        # Phase 0 (spawn exec + interpreter + module import) precedes
        # ``stamps[0]``, so its marker delta would be a misleading ~0ms; the
        # worker measures it against the parent's spawn stamp instead.
        for completed in range(1, index):
            try:
                started = float(self.stamps[completed])
                ended = float(self.stamps[completed + 1])
            except (AttributeError, IndexError, TypeError, ValueError):
                break
            elapsed = _elapsed_ms(started, ended)
            if elapsed is not None:
                fields[f"phase_worker_{_WORKER_PHASES[completed]}_ms"] = elapsed
        return fields


@dataclass
class _ExecutionDiagnostics:
    """Bounded parent-side phase diagnostics for one sandbox execution.

    Every value derives from one ``time.monotonic()`` origin: no extra clock,
    wait, thread, retry, or input-sized allocation is introduced.
    ``mark`` records the duration of the phase that just completed as
    ``phase_parent_<name>_ms`` and remembers it as the last completed phase,
    so a timeout report can name the phase that was in progress without extra
    bookkeeping in the execution path.
    """

    origin: float = field(default_factory=time.monotonic)
    phases_ms: dict[str, float] = field(default_factory=dict)
    reported_phases_ms: dict[str, float] = field(default_factory=dict)
    last_phase: str = "start"
    worker: _WorkerProgress | None = None
    _marked_at: float = field(default_factory=time.monotonic)

    def merge_reported(self, worker_metadata: dict[str, Any]) -> None:
        """Adopt the worker-reported timings that reached the parent.

        These are authoritative when present (the worker measured them around
        its own phases); the shared-marker fields remain the fallback for a
        worker that never posted a response.
        """
        for name, value in worker_metadata.items():
            if name.startswith("phase_") and isinstance(value, (int, float)):
                self.reported_phases_ms[name] = float(value)

    def merge_reported_defaults(self, derived: dict[str, float]) -> None:
        """Fill reported phases from marker-derived values without overriding.

        Worker-reported timings adopted via :meth:`merge_reported` are
        authoritative; these marker-derived fields (e.g.
        ``phase_worker_result_send_ms``) only fill gaps.  Never raises:
        diagnostics must never be able to fail an execution.
        """
        try:
            for name, value in derived.items():
                self.reported_phases_ms.setdefault(name, float(value))
        except (AttributeError, TypeError, ValueError):
            return

    def mark(self, phase: str) -> None:
        """Record the just-completed parent phase.  Never raises."""
        now = time.monotonic()
        elapsed = _elapsed_ms(self._marked_at, now)
        self._marked_at = now
        if elapsed is not None:
            self.phases_ms[f"phase_parent_{phase}_ms"] = elapsed
        self.last_phase = phase

    def elapsed_ms(self) -> float:
        return round(max(0.0, (time.monotonic() - self.origin) * 1000.0), 1)

    def summary(self) -> dict[str, Any]:
        """Bounded, secret-free diagnostics for logs and error reporting."""
        fields: dict[str, Any] = {
            "last_parent_phase": self.last_phase,
            "elapsed_ms": self.elapsed_ms(),
            **self.phases_ms,
        }
        pending = _elapsed_ms(self._marked_at, time.monotonic())
        if pending is not None:
            fields["parent_phase_elapsed_ms"] = pending
        if self.worker is not None:
            fields.update(self.worker.fields())
        fields.update(self.reported_phases_ms)
        return fields

    def describe(self) -> str:
        """One-line, bounded phase summary appended to timeout messages."""
        parts = [f"last completed parent phase={self.last_phase}"]
        pending = _elapsed_ms(self._marked_at, time.monotonic())
        if pending is not None:
            parts.append(f"in progress for {pending:.1f}ms")
        if self.phases_ms:
            completed = " ".join(
                f"{name.removeprefix('phase_parent_').removesuffix('_ms')}={value:.1f}ms"
                for name, value in self.phases_ms.items()
            )
            parts.append(f"completed[{completed}]")
        if self.worker is not None:
            worker = self.worker.fields()
            phase = worker.get("worker_phase")
            if phase is not None:
                parts.append(f"worker phase={phase}")
            pending_worker = worker.get("worker_phase_elapsed_ms")
            if pending_worker is not None:
                parts.append(f"worker in progress for {pending_worker:.1f}ms")
            worker_completed = " ".join(
                f"{name.removeprefix('phase_worker_').removesuffix('_ms')}={value:.1f}ms"
                for name, value in worker.items()
                if name.startswith("phase_worker_")
            )
            if worker_completed:
                parts.append(f"worker completed[{worker_completed}]")
        return "; ".join(parts)


@dataclass
class _WorkerHandle:
    """Parent-owned resources for one sandbox worker."""

    process: mp.Process
    response_conn: Any
    """Parent-held read end of the one-way response pipe.

    The parent never keeps a write handle: it closes its copy immediately
    after ``Process.start()`` so that worker death (or an explicit worker-side
    close) surfaces as a real EOF on this end, distinguishable from a stall.
    """
    input_path: str
    output_path: str
    artifact_path: str = ""
    worker_progress: _WorkerProgress | None = None


_ACTIVE_HANDLES: set[int] = set()
"""Live parent-owned worker handles (process + response pipe + temp paths).

Cleanup removes its handle id; leak tests assert this registry is empty so
process/pipe/temp cleanup is observably complete, not merely non-raising.
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
        diagnostics = _ExecutionDiagnostics()
        execute_t0 = time.perf_counter()
        deadline = self._execution_deadline()
        code_hash = self._log_execution_start(code, df, source, agent_name)
        tree = self._parse_and_validate(code)
        payload = ast.unparse(tree) if hasattr(ast, "unparse") else code
        diagnostics.mark("validate")
        result = self._execute_in_worker(
            payload,
            df,
            source=source,
            agent_name=agent_name,
            deadline=deadline,
            diagnostics=diagnostics,
        )
        self._log_execution_complete(result, execute_t0, source, agent_name, code_hash, diagnostics)
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
        has started, response-pipe reads are non-blocking and yield to the loop
        so a caller can cancel this coroutine and trigger the same cleanup path
        as a timeout.
        """
        execute_t0 = time.perf_counter()
        diagnostics = _ExecutionDiagnostics()
        deadline = self._execution_deadline()
        code_hash = self._log_execution_start(code, df, source, agent_name)
        tree = self._parse_and_validate(code)
        payload = ast.unparse(tree) if hasattr(ast, "unparse") else code
        diagnostics.mark("validate")
        result = await self._execute_in_worker_async(
            payload,
            df,
            source=source,
            agent_name=agent_name,
            deadline=deadline,
            diagnostics=diagnostics,
        )
        self._log_execution_complete(result, execute_t0, source, agent_name, code_hash, diagnostics)
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
        diagnostics: _ExecutionDiagnostics | None = None,
    ) -> None:
        logger.info(
            "sandbox_execute_complete",
            result_shape=result.shape,
            latency_ms=round((time.perf_counter() - execute_t0) * 1000, 1),
            source=source,
            agent=agent_name,
            code_hash=code_hash,
            **(diagnostics.summary() if diagnostics is not None else {}),
            **self.provenance,
        )

    def _execution_deadline(self) -> float:
        return time.monotonic() + max(0.0, self.limits.timeout_seconds)

    def _assert_deadline(
        self, deadline: float, diagnostics: _ExecutionDiagnostics | None = None
    ) -> None:
        if time.monotonic() >= deadline:
            logger.error(
                "sandbox_timeout",
                timeout_seconds=self.limits.timeout_seconds,
                effective_timeout_seconds=max(0.0, self.limits.timeout_seconds),
            )
            raise self._timeout_error(diagnostics)

    def _execute_in_worker(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
        deadline: float | None = None,
        diagnostics: _ExecutionDiagnostics | None = None,
    ) -> pd.DataFrame:
        execution_deadline = self._execution_deadline() if deadline is None else deadline
        handle = self._start_worker(code, df, source, agent_name, execution_deadline, diagnostics)
        deadline = execution_deadline
        if diagnostics is not None:
            diagnostics.worker = handle.worker_progress
        try:
            status, payload, worker_metadata = self._wait_for_response(
                handle, deadline, diagnostics=diagnostics
            )
            if diagnostics is not None:
                diagnostics.mark("response_wait")
            result = self._consume_response(
                handle,
                status,
                payload,
                worker_metadata,
                deadline,
                source=source,
                agent_name=agent_name,
                diagnostics=diagnostics,
            )
            if not self._wait_for_exit(handle.process, deadline):
                # Best-effort exit_code: False implies still alive, so this is
                # normally None and populated only via the tiny
                # is_alive()->exitcode race.
                raise self._timeout_error(diagnostics, exit_code=handle.process.exitcode)
            if diagnostics is not None:
                diagnostics.mark("join")
            return result
        finally:
            self._cleanup_worker(handle, deadline)
            if diagnostics is not None:
                diagnostics.mark("cleanup")

    async def _execute_in_worker_async(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
        deadline: float | None = None,
        diagnostics: _ExecutionDiagnostics | None = None,
    ) -> pd.DataFrame:
        execution_deadline = self._execution_deadline() if deadline is None else deadline
        # This call, including Process.start(), intentionally runs on the
        # event-loop thread; process ownership never moves to a worker thread.
        handle = self._start_worker(code, df, source, agent_name, execution_deadline, diagnostics)
        deadline = execution_deadline
        if diagnostics is not None:
            diagnostics.worker = handle.worker_progress
        cleaned = False
        try:
            status, payload, worker_metadata = await self._poll_response(
                handle, deadline, diagnostics=diagnostics
            )
            if diagnostics is not None:
                diagnostics.mark("response_wait")
            result = self._consume_response(
                handle,
                status,
                payload,
                worker_metadata,
                deadline,
                source=source,
                agent_name=agent_name,
                diagnostics=diagnostics,
            )
            if not await self._wait_for_exit_async(handle.process, deadline):
                # Best-effort exit_code: False implies still alive, so this is
                # normally None and populated only via the tiny
                # is_alive()->exitcode race.
                raise self._timeout_error(diagnostics, exit_code=handle.process.exitcode)
            if diagnostics is not None:
                diagnostics.mark("join")
            return result
        except asyncio.CancelledError:
            # Cleanup is synchronous but has only bounded process/pipe/file
            # operations; importantly, no executor thread survives cancellation.
            self._cleanup_worker(handle, deadline, force=True)
            cleaned = True
            raise
        finally:
            if not cleaned:
                self._cleanup_worker(handle, deadline)
            if diagnostics is not None:
                diagnostics.mark("cleanup")

    def _start_worker(
        self,
        code: str,
        df: pd.DataFrame,
        source: str,
        agent_name: str,
        deadline: float,
        diagnostics: _ExecutionDiagnostics | None = None,
    ) -> _WorkerHandle:
        self._ensure_strict_available()
        ctx = mp.get_context("spawn")
        # One-way pipe, one message per execution.  ``duplex=False`` returns
        # ``(read_end, write_end)``: the parent keeps only the read end and
        # hands the write end to the child.  There is no background writer
        # thread and no semaphore allocation inside the RLIMIT_AS-capped
        # worker, and a worker that dies without sending closes the last write
        # handle so the parent observes a real EOF instead of a stall (see
        # ``_wait_for_response``).
        response_conn: Any
        worker_conn: Any
        response_conn, worker_conn = ctx.Pipe(duplex=False)
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
            self._assert_deadline(deadline, diagnostics)
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".parquet", delete=False, prefix="feature_forge_input_"
            ) as input_file:
                input_path = input_file.name
                df.to_parquet(input_path)
            self._assert_deadline(deadline, diagnostics)
            # Predeclare the output inode before Landlock is installed in the
            # child so the child can publish exactly this file.
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".parquet", delete=False, prefix="ff_sandbox_"
            ) as output_file:
                output_path = output_file.name
            self._assert_deadline(deadline, diagnostics)
            if diagnostics is not None:
                diagnostics.mark("serialize")
            progress = self._new_worker_progress(ctx)
            if diagnostics is not None:
                # Attach before Process.start() so a spawn-phase timeout can
                # still report whatever the child managed to record.
                diagnostics.worker = progress
            # Monotonic stamp handed to the child so the spawn exec +
            # interpreter + module-import window (which precedes
            # ``_sandbox_worker_main`` entry) stays measurable.  CPython
            # monotonic clocks are system-wide on every supported platform.
            parent_start = time.monotonic()
            process = ctx.Process(
                target=_sandbox_worker_main,
                args=(
                    code,
                    input_path,
                    output_path,
                    self.limits.max_memory_mb,
                    worker_conn,
                    source,
                    agent_name,
                    self.profile.value,
                    progress,
                    parent_start,
                ),
                daemon=True,
            )
            # Direct call is intentional: async ownership starts here, on the
            # caller's event-loop/main thread.
            process.start()
            # The parent must never retain a write handle: closing our copy
            # here leaves the child as the sole writer, so worker death (or
            # the worker's own explicit close) yields EOF on ``response_conn``.
            self._close_response_conn(worker_conn)
            worker_conn = None
            if diagnostics is not None:
                diagnostics.mark("spawn")
            self._assert_deadline(deadline, diagnostics)
            handle = _WorkerHandle(
                cast("mp.Process", process),
                response_conn,
                input_path,
                output_path,
                worker_progress=progress,
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
            # Release both ends on every failure path.  ``worker_conn`` is
            # already None when the close above succeeded; ``None`` is a no-op.
            self._close_response_conn(response_conn)
            self._close_response_conn(worker_conn)
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

    @staticmethod
    def _new_worker_progress(ctx: Any) -> _WorkerProgress | None:
        """Best-effort lock-free worker progress marker (diagnostics only).

        Creation failures (unsupported platform, mmap exhaustion) are not an
        execution failure: the sandbox contract is unchanged and the timeout
        report simply omits the worker-side fields.
        """
        try:
            return _WorkerProgress(
                phase=ctx.Value(ctypes.c_int, _PHASE_NOT_STARTED, lock=False),
                stamps=ctx.RawArray(ctypes.c_double, len(_WORKER_PHASES)),
            )
        except (AttributeError, OSError, RuntimeError, ValueError):
            return None

    def _timeout_error(
        self,
        diagnostics: _ExecutionDiagnostics | None = None,
        *,
        exit_code: int | None = None,
    ) -> SandboxTimeoutError:
        """Build the typed timeout error, recording the phase trail once.

        The WARNING event is the CI-visible diagnostic: CI runs at WARNING, so
        the timeout path must carry the last completed phase, per-phase
        elapsed milliseconds, and the worker's shared-memory phase here rather
        than only in INFO-level completion logs.
        """
        message = f"Sandbox execution timed out after {self.limits.timeout_seconds:.1f}s"
        fields: dict[str, Any] = {}
        if diagnostics is not None:
            fields = diagnostics.summary()
            message = f"{message} ({diagnostics.describe()})"
        if exit_code is not None:
            # Abnormal worker exit observed while the deadline expired.
            fields["worker_exit_code"] = exit_code
            message = f"{message} (worker exit_code={exit_code})"
        logger.warning(
            "sandbox_timeout_diagnostics",
            timeout_seconds=self.limits.timeout_seconds,
            **fields,
        )
        return SandboxTimeoutError(message)

    @staticmethod
    def _worker_death_message(handle: _WorkerHandle) -> str:
        """Typed worker-death message, including an abnormal exit code."""
        message = "sandbox worker died before reporting a result"
        try:
            exit_code = handle.process.exitcode
        except (AssertionError, AttributeError, ValueError):
            return message
        if exit_code is None:
            return message
        return f"{message} (worker exit_code={exit_code})"

    def _wait_for_response(
        self,
        handle: _WorkerHandle,
        deadline: float,
        *,
        diagnostics: _ExecutionDiagnostics | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._timeout_error(diagnostics)
        try:
            # ``poll`` is the only wait: it is bounded by the single
            # authoritative deadline.  It returns True both when a message is
            # available *and* when the pipe is at EOF (all write handles
            # closed), so a dead worker is detected immediately rather than
            # racing the deadline.
            if not handle.response_conn.poll(timeout=remaining):
                raise self._timeout_error(diagnostics)
            return cast("tuple[str, str, dict[str, Any]]", handle.response_conn.recv())
        except EOFError as exc:
            # The worker died (or closed its write end) before posting.  This
            # is now deterministic: poll() reported readability, recv() found
            # no message, so the read end is at EOF.  Surface it through the
            # typed hierarchy instead of a raw EOFError that would bypass
            # callers' CodeExecutionError handling.  An abnormal exit code is
            # included when observable.
            raise CodeExecutionError(self._worker_death_message(handle)) from exc

    async def _poll_response(
        self,
        handle: _WorkerHandle,
        deadline: float,
        *,
        diagnostics: _ExecutionDiagnostics | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        while True:
            try:
                if handle.response_conn.poll(0):
                    return cast("tuple[str, str, dict[str, Any]]", handle.response_conn.recv())
            except EOFError as exc:
                # Same deterministic EOF-vs-stall distinction as the sync
                # path: non-blocking poll() saw readability (EOF included),
                # recv() found no message.
                raise CodeExecutionError(self._worker_death_message(handle)) from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timeout_error(diagnostics) from None
            await asyncio.sleep(min(0.01, remaining))

    def _log_worker_phases(
        self,
        handle: _WorkerHandle,
        worker_metadata: dict[str, Any],
        *,
        source: str,
        agent_name: str,
    ) -> dict[str, float]:
        """Log the worker's reported phase timings (secret-free, bounded).

        Worker-reported facts are logged in the parent only: the worker must
        never emit structlog/OTel events (see _worker_log_event).  The result
        send is completed by the worker but observable only here, so its
        latency is derived from the shared progress marker when available.
        Returns the logged fields so the caller can fold them into the
        execution summary.
        """
        fields: dict[str, float] = {
            name: float(value)
            for name, value in worker_metadata.items()
            if name.startswith("phase_") and isinstance(value, (int, float))
        }
        progress = handle.worker_progress
        if progress is not None:
            send_ms = progress.phase_latency_ms("result_send")
            if send_ms is not None:
                fields.setdefault("phase_worker_result_send_ms", send_ms)
        if not fields:
            return fields
        logger.info("sandbox_worker_phases", source=source, agent=agent_name, **fields)
        return fields

    def _consume_response(
        self,
        handle: _WorkerHandle,
        status: str,
        payload: str,
        worker_metadata: dict[str, Any],
        deadline: float,
        *,
        source: str = "unknown",
        agent_name: str = "unknown",
        diagnostics: _ExecutionDiagnostics | None = None,
    ) -> pd.DataFrame:
        phase_fields = self._log_worker_phases(
            handle, worker_metadata, source=source, agent_name=agent_name
        )
        if diagnostics is not None:
            diagnostics.merge_reported(worker_metadata)
            # Fold in marker-derived timings that never travel in the worker
            # response tuple (e.g. `phase_worker_result_send_ms`).  The
            # worker-reported values recorded above always win; these only
            # fill gaps.  Best-effort: never fail an execution.
            diagnostics.merge_reported_defaults(phase_fields)
        if status == "ok":
            if os.path.realpath(payload) != os.path.realpath(handle.output_path):
                raise CodeExecutionError("Sandbox worker returned an invalid output artifact")
            # Only record the parent-created output path for cleanup.  Never
            # unlink an arbitrary path supplied by the worker response.
            handle.artifact_path = handle.output_path
            # Worker-reported containment is logged here in the parent: the
            # worker itself must not emit structlog/OTel events (see
            # _worker_log_event).  Phase timings travel in the same metadata
            # and are logged separately as `sandbox_worker_phases`.
            # Explicit exclusion convention over the four fixed containment
            # keys: keep every non-`phase_*` entry, drop the phase diagnostics.
            containment = {
                name: value
                for name, value in worker_metadata.items()
                if not name.startswith("phase_")
            }
            logger.info(
                "sandbox_worker_containment",
                source=source,
                agent=agent_name,
                **containment,
            )
            self._assert_deadline(deadline, diagnostics)
            result = pd.read_parquet(handle.output_path)
            self._assert_deadline(deadline, diagnostics)
            if diagnostics is not None:
                diagnostics.mark("result_read")
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
        self._close_response_conn(handle.response_conn)
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
    def _close_response_conn(response_conn: Any) -> None:
        """Close one end of the response pipe.  Idempotent; never raises.

        ``None`` is accepted so the failure path can close both ends
        unconditionally, and a double close (e.g. the parent's write end after
        a successful ``Process.start()``) is a no-op.
        """
        if response_conn is None:
            return
        try:
            response_conn.close()
        except (AttributeError, OSError, ValueError):
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
    against its limit so it cannot start the thread it needs before sending
    its response (found during plan 23 PR 5 qualification on Python 3.13
    under single-thread BLAS/OpenMP env). Provider credentials and every other
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
    response_conn: Any,
    source: str = "unknown",
    agent_name: str = "unknown",
    profile: str = SandboxProfile.STRICT.value,
    progress: _WorkerProgress | None = None,
    parent_start: float = 0.0,
) -> None:
    """Worker entry point that owns the response pipe's write end.

    The parent holds only the read end, so this end is closed before the
    worker exits on *every* path (success, typed worker error, unexpected
    exception).  That explicit close, rather than process teardown, is what
    makes worker death a deterministic EOF for the parent instead of a
    deadline stall.  Closing after ``send`` never discards the message: the
    bytes are already in the pipe buffer and remain readable until drained.
    """
    try:
        _run_sandbox_worker(
            code,
            input_parquet_path,
            output_parquet_path,
            max_memory_mb,
            response_conn,
            source=source,
            agent_name=agent_name,
            profile=profile,
            progress=progress,
            parent_start=parent_start,
        )
    finally:
        try:
            response_conn.close()
        except (AttributeError, OSError, ValueError):
            pass


def _run_sandbox_worker(
    code: str,
    input_parquet_path: str,
    output_parquet_path: str,
    max_memory_mb: int,
    response_conn: Any,
    source: str = "unknown",
    agent_name: str = "unknown",
    profile: str = SandboxProfile.STRICT.value,
    progress: _WorkerProgress | None = None,
    parent_start: float = 0.0,
) -> None:
    # Phase diagnostics: worker-local monotonic deltas ride back in the
    # response metadata, while the shared ``progress`` marker (when the parent
    # provided one) lets a *timed-out* parent name the phase the worker was in
    # without any extra wait, pipe, or thread (ADR 0019 decision 5).
    worker_started_at = time.monotonic()
    worker_phases: dict[str, Any] = {}
    if progress is not None:
        progress.enter("bootstrap")
    if (bootstrap_ms := _elapsed_ms(parent_start, worker_started_at)) is not None:
        # Spawn exec + interpreter start + module import all run before this
        # function is entered, so only the parent's spawn stamp can measure
        # them; CPython's monotonic clock is system-wide on every supported
        # platform, so the two processes share one timeline.
        worker_phases["phase_worker_bootstrap_ms"] = bootstrap_ms

    def _begin_phase(name: str) -> float:
        """Mark *name* in progress and return its monotonic start stamp."""
        started = time.monotonic()
        if progress is not None:
            progress.enter(name)
        return started

    def _end_phase(name: str, started: float) -> float:
        """Record the duration of *name* and return the end stamp."""
        ended = time.monotonic()
        if (elapsed := _elapsed_ms(started, ended)) is not None:
            worker_phases[f"phase_worker_{name}_ms"] = elapsed
        return ended

    def _response_metadata(containment: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = dict(containment or {})
        metadata.update(worker_phases)
        if (total := _elapsed_ms(worker_started_at, time.monotonic())) is not None:
            metadata["phase_worker_total_ms"] = total
        return metadata

    def _respond(status: str, payload: str, containment: dict[str, Any] | None = None) -> None:
        """Publish the single response, timing the actual pipe send.

        ``Connection.send`` is synchronous, so the ``result_send`` marker
        brackets the real write.  The parent derives
        ``phase_worker_result_send_ms`` from that marker once it has received
        the response (see ``_log_worker_phases``); the metadata itself cannot
        carry it because it is computed before the send that transports it.

        The payload is capped at :data:`_MAX_RESPONSE_PAYLOAD_CHARS` before the
        send: generated code can inflate exception text without bound, and the
        parent's post-poll ``recv`` is not separately deadline-bounded.
        Capping at the source bounds that drain window to one capped message,
        so the authoritative deadline plus a single capped transfer is the
        total worst case.
        """
        if progress is not None:
            progress.enter("result_send")
        payload = payload[:_MAX_RESPONSE_PAYLOAD_CHARS]
        response_conn.send((status, payload, _response_metadata(containment)))
        if progress is not None:
            progress.enter("done")

    # Establish containment's termination boundary before any potentially
    # expensive worker phase, then remove inherited secrets before loading data.
    phase_started_at = _begin_phase("containment_setup")
    _establish_worker_process_group()
    _scrub_worker_environment()
    _apply_resource_limits(max_memory_mb=max_memory_mb)
    phase_started_at = _end_phase("containment_setup", phase_started_at)
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

    phase_started_at = _begin_phase("input_load")
    try:
        # Input loading happens before restrictions because it is trusted
        # sandbox plumbing. The output inode was predeclared by the parent.
        df = pd.read_parquet(input_parquet_path)
    except Exception as exc:
        _respond("error", f"Failed to read input data: {exc}")
        return
    phase_started_at = _end_phase("input_load", phase_started_at)

    output_to_parquet = pd.DataFrame.to_parquet
    phase_started_at = _begin_phase("landlock")
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
        _end_phase("landlock", phase_started_at)
        _respond("containment_unavailable", str(exc))
        return
    phase_started_at = _end_phase("landlock", phase_started_at)

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

    phase_started_at = _begin_phase("code_exec")
    try:
        exec(compile(code, filename="<sandbox>", mode="exec"), safe_globals, local_vars)
        generate_features = local_vars.get("generate_features")
        if generate_features is None:
            _end_phase("code_exec", phase_started_at)
            _respond("error", "Code must define a 'generate_features(df)' function")
            return
        result = generate_features(df)
        if not isinstance(result, pd.DataFrame):
            _end_phase("code_exec", phase_started_at)
            _respond(
                "error",
                f"generate_features must return a DataFrame, got {type(result).__name__}",
            )
            return
        # Convert non-serializable types (Interval, Categorical, object) to safe numeric/string
        result = _to_parquet_safe(result)
        phase_started_at = _end_phase("code_exec", phase_started_at)
        phase_started_at = _begin_phase("output_publish")
        output_to_parquet(result, output_parquet_path)
        _end_phase("output_publish", phase_started_at)
        _respond("ok", output_parquet_path, containment_report)
    except BaseException as exc:  # pragma: no cover - subprocess path
        _respond("error", f"Feature generation execution failed: {exc}")


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
