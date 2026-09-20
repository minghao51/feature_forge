#!/usr/bin/env python
"""OS-isolation spike probe for plan 23 §5.2 / ADR 0019 (PR 1).

Standalone stdlib-only probe that determines, on the *current* Linux host,
which OS-enforced containment mechanism the production sandbox can rely on:

- platform/kernel/arch identity;
- Landlock availability and ABI version (raw syscall 444);
- unprivileged user namespaces (fork + ``unshare(CLONE_NEWUSER|CLONE_NEWNS)``);
- seccomp classic availability (``prctl(PR_GET_SECCOMP)`` errno probe) plus
  libseccomp presence;
- bubblewrap / firejail binaries on PATH;
- a live Landlock containment demo (ABI >= 1): a forked child restricts itself
  to runtime-library reads plus two temp input/output roots, then attempts to
  open a sentinel created *outside* the allowed roots (expected: denied with
  EACCES/EPERM) and to open/connect a socket (expected: permitted under
  Landlock ABI < 4 — network rules need seccomp or Landlock ABI >= 4).

Writes a JSON report with resolved results and provenance. Exit code is 0
unless the report cannot be written; probe failures are data, not errors.

Usage:
    uv run python scripts/spike_sandbox_isolation.py \
        --report experiments/sandbox_isolation_spike/2026-09-15/report.json
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import json
import os
import platform
import select
import shutil
import socket
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# --- Landlock constants (kernel UAPI, linux/landlock.h) ----------------------

LANDLOCK_CREATE_RULESET_VERSION = 1  # flags bit: query the ABI version only

# Access-rights bit layout per ABI (1<<0 .. 1<<14). Bits are additive: a ruleset
# must only handle bits the running ABI understands, else create_ruleset fails
# with EINVAL.
_ACCESS_BITS_ABI1 = {
    "execute": 1 << 0,
    "write_file": 1 << 1,
    "read_file": 1 << 2,
    "read_dir": 1 << 3,
    "remove_dir": 1 << 4,
    "remove_file": 1 << 5,
    "make_char": 1 << 6,
    "make_dir": 1 << 7,
    "make_reg": 1 << 8,
    "make_sock": 1 << 9,
    "make_fifo": 1 << 10,
    "make_block": 1 << 11,
    "make_sym": 1 << 12,
}
_ACCESS_BITS_ABI2 = {"refer": 1 << 13}  # ABI >= 2 only
_ACCESS_BITS_ABI3 = {"truncate": 1 << 14}  # ABI >= 3 only

LANDLOCK_RULE_PATH_BENEATH = 1

# struct landlock_path_beneath_attr { __u64 allowed_access; __u64 parent_fd; }
_PATH_BENEATH_STRUCT = struct.Struct("=QQ")

PR_SET_NO_NEW_PRIVS = 38
PR_GET_SECCOMP = 21

CLONE_NEWUSER = 0x10000000
CLONE_NEWNS = 0x00020000

# Same unified syscall numbers on x86_64 and aarch64.
_LANDLOCK_CREATE_RULESET_NR = 444
_LANDLOCK_ADD_RULE_NR = 445
_LANDLOCK_RESTRICT_SELF_NR = 446


def _libc() -> ctypes.CDLL:
    """Load the C library with errno preservation for syscall probes."""
    return ctypes.CDLL(None, use_errno=True)


def _syscall(libc: ctypes.CDLL, number: int, *args: int) -> int:
    """Invoke a raw syscall; returns the raw long result (errno via ctypes)."""
    libc.syscall.restype = ctypes.c_long
    libc.syscall.argtypes = [ctypes.c_long] + [ctypes.c_long] * len(args)
    result = libc.syscall(number, *args)
    return int(result)


def _load_abi_access_bits(abi: int) -> dict[str, int]:
    """Access bits valid for the detected Landlock ABI."""
    bits = dict(_ACCESS_BITS_ABI1)
    if abi >= 2:
        bits |= _ACCESS_BITS_ABI2
    if abi >= 3:
        bits |= _ACCESS_BITS_ABI3
    return bits


# --- probes ------------------------------------------------------------------


def probe_platform() -> dict[str, str]:
    """Probe a: OS/kernel/arch identity."""
    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "python": platform.python_version(),
    }


def probe_landlock() -> dict[str, Any]:
    """Probe b: landlock_create_ruleset(444) with flags=VERSION, attr=NULL, size=0.

    Returns the ABI version on success; errno (ENOSYS/EOPNOTSUPP/EINVAL/...) on
    failure. Only x86_64/aarch64 syscall numbers are hard-coded; other arches
    report an "unsupported arch" note rather than guessing numbers.
    """
    machine = platform.machine()
    if machine not in ("x86_64", "aarch64", "arm64"):
        return {
            "available": False,
            "abi_version": None,
            "errno": None,
            "notes": f"no hard-coded syscall numbers for arch {machine!r}; probe skipped",
        }
    libc = _libc()
    result = _syscall(libc, _LANDLOCK_CREATE_RULESET_NR, 0, 0, LANDLOCK_CREATE_RULESET_VERSION)
    if result >= 0:
        return {
            "available": True,
            "abi_version": int(result),
            "errno": None,
            "notes": "landlock_create_ruleset(VERSION) succeeded (kernel < 5.19 needs "
            "CONFIG_SECURITY_LANDLOCK=y and landlock LSM enabled)",
        }
    err = ctypes.get_errno()
    meanings = {
        errno.ENOSYS: "ENOSYS: syscall not compiled in (CONFIG_SECURITY_LANDLOCK=n)",
        errno.EOPNOTSUPP: "EOPNOTSUPP: Landlock enabled but not supported by all backing filesystems",
        errno.EINVAL: "EINVAL: invalid flags",
    }
    return {
        "available": False,
        "abi_version": None,
        "errno": err,
        "notes": meanings.get(err, f"unexpected errno {err}"),
    }


def _userns_child(write_fd: int) -> None:  # pragma: no cover - runs in forked child
    """Child body for probe c: try unshare(CLONE_NEWUSER|CLONE_NEWNS)."""
    try:
        libc = _libc()
        libc.unshare.restype = ctypes.c_int
        libc.unshare.argtypes = [ctypes.c_int]
        result = libc.unshare(CLONE_NEWUSER | CLONE_NEWNS)
        err = ctypes.get_errno() if result != 0 else None
        payload = json.dumps({"status": "ok" if result == 0 else "error", "errno": err})
    except Exception as exc:  # pragma: no cover - defensive
        payload = json.dumps({"status": "error", "exception": repr(exc)})
    os.write(write_fd, payload.encode())
    os._exit(0)


def probe_userns(timeout: float) -> dict[str, Any]:
    """Probe c: fork a child and report unshare(CLONE_NEWUSER|CLONE_NEWNS) result."""
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            os.close(read_fd)
            try:
                _userns_child(write_fd)
            finally:
                os._exit(127)
        os.close(write_fd)
        deadline = time.monotonic() + timeout
        data = b""
        while time.monotonic() < deadline:
            ready, _, _ = select.select([read_fd], [], [], max(0.0, deadline - time.monotonic()))
            if ready:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                data += chunk
        _, status = os.waitpid(pid, os.WNOHANG if data else 0)
        if not data:
            os.kill(pid, 9)
            _, status = os.waitpid(pid, 0)
            return {
                "available": False,
                "errno": None,
                "notes": f"child timed out (status {status})",
            }
        result: dict[str, Any] = json.loads(data.decode())
        meanings = {
            errno.EPERM: "EPERM: unprivileged userns denied (sysctl or container policy)",
            errno.EINVAL: "EINVAL: invalid flags",
        }
        if result["status"] == "ok":
            return {
                "available": True,
                "errno": None,
                "notes": "unshare(CLONE_NEWUSER|CLONE_NEWNS) succeeded",
            }
        err = result.get("errno")
        return {
            "available": False,
            "errno": err,
            "notes": meanings.get(err, f"unshare failed with errno {err}"),
        }
    finally:
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def probe_seccomp() -> dict[str, Any]:
    """Probe d: libseccomp presence + classic seccomp via prctl(PR_GET_SECCOMP).

    prctl(PR_GET_SECCOMP) returning EFAULT means seccomp is compiled in and
    enabled (the probe writes through a bad address on purpose); EINVAL means
    the kernel lacks seccomp support.
    """
    lib = ctypes.util.find_library("seccomp")
    libc = _libc()
    libc.prctl.restype = ctypes.c_int
    libc.prctl.argtypes = [ctypes.c_int] * 5
    result = libc.prctl(PR_GET_SECCOMP, 0, 0, 0, 0)
    err = ctypes.get_errno() if result != 0 else None
    if result >= 0:
        notes = f"PR_GET_SECCOMP returned mode {result} (seccomp active in this process)"
        available = True
    elif err == errno.EFAULT:
        notes = "EFAULT: seccomp supported (classic probe convention)"
        available = True
    elif err == errno.EINVAL:
        notes = "EINVAL: seccomp not available (CONFIG_SECCOMP=n)"
        available = False
    else:
        notes = f"unexpected errno {err}"
        available = False
    return {"available": available, "errno": err, "libseccomp": lib, "notes": notes}


def probe_tools() -> dict[str, str | None]:
    """Probe e: bubblewrap / firejail on PATH."""
    return {
        "bubblewrap": shutil.which("bwrap"),
        "firejail": shutil.which("firejail"),
    }


# --- containment demo (probe f) ----------------------------------------------


def _demo_child(
    write_fd: int,
    sentinel_path: str,
    input_root: str,
    output_root: str,
    abi: int,
) -> None:  # pragma: no cover - runs in forked child
    """Landlock live containment demo child; reports via JSON over the pipe."""
    result: dict[str, Any] = {
        "sentinel_read_denied": None,
        "errno": None,
        "socket_connect_result": None,
    }
    try:
        # (i) Sentinel with random bytes OUTSIDE the allowed roots, created
        # before restriction so its absence cannot masquerade as a denial.
        Path(sentinel_path).write_bytes(os.urandom(64))
        libc = _libc()

        # (ii) Ruleset handling every access bit valid for the detected ABI
        # (requested READ_FILE|READ_DIR|WRITE_FILE|EXECUTE are ABI-1 bits; the
        # extra ABI-1 bits are needed because the temp-root rules grant
        # MAKE_REG/MAKE_DIR/REMOVE_FILE, which must be in the handled set).
        access = _load_abi_access_bits(abi)
        handled = 0
        for bit in access.values():
            handled |= bit
        attr = (ctypes.c_char * 24)()  # landlock_ruleset_attr { __u64 handled_access; }
        ctypes.memmove(attr, struct.pack("=Q", handled), 8)
        ruleset_fd = _syscall(libc, _LANDLOCK_CREATE_RULESET_NR, ctypes.addressof(attr), 24, 0)
        if ruleset_fd < 0:
            raise RuntimeError(f"create_ruleset failed errno={ctypes.get_errno()}")

        def allow(path: str, mask: int) -> None:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                parent = _PATH_BENEATH_STRUCT.pack(mask & handled, fd)
                buf = (ctypes.c_char * len(parent)).from_buffer_copy(parent)
                rc = _syscall(
                    libc,
                    _LANDLOCK_ADD_RULE_NR,
                    ruleset_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.addressof(buf),
                    0,
                )
                if rc != 0:
                    raise RuntimeError(f"add_rule({path}) failed errno={ctypes.get_errno()}")
            finally:
                os.close(fd)

        # (iii) PATH_BENEATH rules: runtime reads + the two temp roots.
        read_exec = access["read_file"] | access["read_dir"] | access["execute"]
        read_write = (
            read_exec
            | access["write_file"]
            | access["remove_file"]
            | access["make_reg"]
            | access["make_dir"]
        )
        runtime_roots = [
            p
            for p in (
                "/usr",
                "/lib",
                "/lib64",
                "/etc",  # /etc/ld.so.cache + loader config
                sys.prefix,
                os.path.dirname(os.__file__),  # stdlib
            )
            if os.path.isdir(p)
        ]
        for path in runtime_roots:
            allow(path, read_exec)
        allow(input_root, read_write)
        allow(output_root, read_write)

        # (iv) Commit: no_new_privs then restrict_self.
        libc.prctl.restype = ctypes.c_int
        libc.prctl.argtypes = [ctypes.c_int] * 5
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise RuntimeError(f"PR_SET_NO_NEW_PRIVS failed errno={ctypes.get_errno()}")
        if _syscall(libc, _LANDLOCK_RESTRICT_SELF_NR, ruleset_fd, 0) != 0:
            raise RuntimeError(f"restrict_self failed errno={ctypes.get_errno()}")
        os.close(ruleset_fd)

        # (v) Post-restriction probes.
        try:
            with open(sentinel_path, "rb") as fh:
                fh.read(1)
        except OSError as exc:
            result["errno"] = exc.errno
            result["sentinel_read_denied"] = exc.errno in (errno.EACCES, errno.EPERM)
        else:
            result["sentinel_read_denied"] = False

        sock = socket.socket()
        sock.settimeout(2.0)
        try:
            sock.connect(("127.0.0.1", 9))
            result["socket_connect_result"] = "connected (permitted)"
        except ConnectionRefusedError:
            # Refusal comes from the kernel, not an LSM denial: the socket
            # call itself was permitted by the sandbox.
            result["socket_connect_result"] = "permitted (ECONNREFUSED; socket syscall allowed)"
        except OSError as exc:
            result["socket_connect_result"] = f"error errno={exc.errno}"
        finally:
            sock.close()
    except Exception as exc:  # pragma: no cover - defensive
        result["child_error"] = repr(exc)
    finally:
        os.write(write_fd, json.dumps(result).encode())
        os._exit(0)


def run_containment_demo(abi: int, timeout: float) -> dict[str, Any]:
    """Fork the demo child and collect its results; cleans up temp dirs."""
    if abi < 1:
        return {
            "performed": False,
            "sentinel_read_denied": None,
            "errno": None,
            "socket_connect_result": None,
            "notes": "demo not attempted: Landlock ABI < 1 (unavailable)",
        }
    tmp = tempfile.mkdtemp(prefix="ff-sandbox-spike-")
    input_root = os.path.join(tmp, "input")
    output_root = os.path.join(tmp, "output")
    os.mkdir(input_root)
    os.mkdir(output_root)
    # Sentinel lives in the tmp base dir, OUTSIDE both allowed roots.
    sentinel_path = os.path.join(tmp, "sentinel.bin")
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            os.close(read_fd)
            try:
                _demo_child(write_fd, sentinel_path, input_root, output_root, abi)
            finally:
                os._exit(127)
        os.close(write_fd)
        deadline = time.monotonic() + timeout
        data = b""
        while time.monotonic() < deadline:
            ready, _, _ = select.select([read_fd], [], [], max(0.0, deadline - time.monotonic()))
            if ready:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                data += chunk
        if not data:
            os.kill(pid, 9)
            _, status = os.waitpid(pid, 0)
            return {
                "performed": True,
                "sentinel_read_denied": None,
                "errno": None,
                "socket_connect_result": None,
                "notes": f"demo child timed out after {timeout}s (status {status})",
            }
        result: dict[str, Any] = json.loads(data.decode())
        result["performed"] = True
        if "child_error" in result:
            result["notes"] = (
                f"demo child failed before restriction probes: {result['child_error']}"
            )
        else:
            result.setdefault(
                "notes",
                "child restricted via landlock_restrict_self with runtime reads "
                "(usr/lib/lib64/etc/sys.prefix/stdlib) + two temp roots; sentinel outside roots",
            )
        return result
    finally:
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


# --- main --------------------------------------------------------------------


def build_report(sentinel_timeout: float) -> dict[str, Any]:
    """Run every probe and assemble the JSON-serializable report."""
    uname = platform.uname()
    landlock = probe_landlock()
    report: dict[str, Any] = {
        "schema": "ff.sandbox_isolation_spike.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": probe_platform(),
        "kernel": uname.release,
        "arch": uname.machine,
        "landlock": landlock,
        "userns": probe_userns(sentinel_timeout),
        "seccomp": probe_seccomp(),
        "bubblewrap": probe_tools()["bubblewrap"],
        "firejail": probe_tools()["firejail"],
        "containment_demo": {},
        "provenance": {
            "script": "scripts/spike_sandbox_isolation.py",
            "plan": "docs/plan/23_evaluation_integrity_security_hardening.md §5.2 (PR 1 spike)",
            "sentinel_timeout_s": sentinel_timeout,
            "landlock_syscalls": {
                "create_ruleset": _LANDLOCK_CREATE_RULESET_NR,
                "add_rule": _LANDLOCK_ADD_RULE_NR,
                "restrict_self": _LANDLOCK_RESTRICT_SELF_NR,
            },
        },
    }
    if landlock["available"] and landlock["abi_version"] is not None:
        report["containment_demo"] = run_containment_demo(
            int(landlock["abi_version"]), sentinel_timeout
        )
    else:
        report["containment_demo"] = {
            "performed": False,
            "sentinel_read_denied": None,
            "errno": None,
            "socket_connect_result": None,
            "notes": "demo not attempted: Landlock unavailable",
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="path for the JSON report")
    parser.add_argument(
        "--sentinel-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for forked probe/demo children (default: 10)",
    )
    args = parser.parse_args()

    report = build_report(args.sentinel_timeout)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report written to {report_path}")
    print(
        json.dumps(
            {k: report[k] for k in ("landlock", "userns", "seccomp", "containment_demo")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
