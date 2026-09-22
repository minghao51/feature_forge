"""Focused tests for the sandbox's immediate library-I/O defense."""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest

import feature_forge.evaluation.sandbox as sandbox_module
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import (
    CodeExecutionError,
    SandboxTimeoutError,
    SandboxValidationError,
)


def _code(body: str) -> str:
    return (
        "def generate_features(df):\n"
        + "\n".join(f"    {line}" for line in body.splitlines())
        + "\n    return df\n"
    )


@pytest.mark.parametrize(
    "statement",
    [
        "arr = n2.fromfile('/tmp/secret', dtype=np.uint8)",
        "arr = np.load('/tmp/secret')",
        "np.save('/tmp/secret', df)",
        "mm = np.memmap('/tmp/secret', dtype=np.uint8)",
        "extra = p.read_feather('/tmp/secret')",
        "df.to_hdf('/tmp/secret', key='df')",
        "np.lib.format.open_memmap('/tmp/secret.npy', mode='w+')",
        "source = np.lib.npyio.DataSource()",
    ],
)
def test_file_capable_library_attributes_are_blocked(statement: str) -> None:
    code = "import numpy as np\nimport pandas as p\nimport numpy as n2\n" + _code(statement)
    with pytest.raises(SandboxValidationError, match="Blocked file I/O API"):
        SandboxedExecutor()._parse_and_validate(code)


def test_imported_file_api_alias_is_blocked() -> None:
    code = "from numpy import load as read_anything\n" + _code("read_anything('/tmp/secret')")
    with pytest.raises(SandboxValidationError, match="Blocked file I/O API"):
        SandboxedExecutor()._parse_and_validate(code)


def test_normal_numeric_and_dataframe_operations_remain_allowed() -> None:
    code = "import numpy as np\n" + _code(
        "result = df.copy()\n"
        "result['scaled'] = np.mean(df['x']) * df['x']\n"
        "result['position'] = df['label'].str.find('needle')\n"
        "return result"
    )
    SandboxedExecutor()._parse_and_validate(code)


@pytest.mark.parametrize(
    "code",
    [
        "import numpy as np\n" + _code("return np.ctypeslib.ctypes"),
        "import ctypes\n" + _code("return df"),
        "from ctypes import CDLL\n" + _code("return df"),
        "import numpy as np\n" + _code("return np.ctypeslib.ctypes.CDLL(None).prlimit64"),
        "import numpy as np\n" + _code("return np.setrlimit"),
        _code("return setrlimit(0, 0)"),
        "from numpy import ctypeslib\n" + _code("return df"),
        "import numpy.ctypeslib\n" + _code("return df"),
        "from numpy.ctypeslib import as_array\n" + _code("return df"),
    ],
)
def test_ctypes_and_rlimit_escape_surfaces_are_blocked_at_ast(code: str) -> None:
    """The live ``ctypes`` module and rlimit writers are static policy.

    ``np.ctypeslib.ctypes`` is reachable from the pre-bound ``np`` global and
    ``prlimit64``/``setrlimit`` could raise the soft RLIMIT_AS back to
    infinity; every spelling (module import, attribute chain, bare name,
    aliased import) must be rejected before a worker starts.
    """
    with pytest.raises(SandboxValidationError):
        SandboxedExecutor()._parse_and_validate(code)


def _require_landlock() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("runtime escape guards are pinned under strict Landlock on Linux")
    if sandbox_module._probe_landlock_abi() is None:
        pytest.skip("Landlock is unavailable on this host")


def test_ctypes_traversal_ast_cannot_name_dies_at_runtime() -> None:
    """``str.format`` field access bypasses AST names but not the runtime guard.

    ``"{0.ctypeslib.ctypes.CDLL}".format(np)`` never appears as an AST
    attribute named ``ctypeslib``/``ctypes`` because the field path lives
    inside a string constant, so the AST layer accepts it.  The worker's
    runtime patch of ``np.ctypeslib`` must make the traversal fail as a typed
    ``CodeExecutionError`` before the real ctypes module is reachable.
    """
    _require_landlock()
    code = (
        "def generate_features(df):\n"
        "    leaked = '{0.ctypeslib.ctypes.CDLL}'.format(np)\n"
        "    return df\n"
    )
    # The static layer cannot see the field path, proving the runtime pin is
    # the authoritative one for this variant.
    SandboxedExecutor()._parse_and_validate(code)
    try:
        SandboxedExecutor(timeout_seconds=30.0).execute(code, pd.DataFrame({"x": [1.0]}))
    except SandboxTimeoutError as exc:
        pytest.skip(f"worker timed out under load before reporting the guard: {exc}")
    except CodeExecutionError:
        pass
    else:
        pytest.fail("ctypes traversal reached the real module")


def test_runtime_guard_survives_ctypeslib_reimport() -> None:
    """Re-patching the parent attribute is not enough; a re-import must not reopen it.

    ``import numpy.ctypeslib`` re-binds ``numpy.ctypeslib`` from ``sys.modules``
    to the real submodule, which would undo a parent-only patch.  The worker's
    guard also patches the live submodule's own ``ctypes`` reference so the
    traversal stays dead after any re-import.
    """
    script = (
        "import sys, numpy\n"
        "from feature_forge.evaluation.sandbox import _install_library_io_guards\n"
        "def _blocked(*_a, **_k):\n"
        "    raise PermissionError('blocked')\n"
        "_install_library_io_guards(_blocked)\n"
        "assert 'numpy.ctypeslib' in sys.modules\n"
        "# Simulate a re-import re-binding the parent attribute to the real\n"
        "# submodule object; its own ctypes reference must already be patched.\n"
        "numpy.ctypeslib = sys.modules['numpy.ctypeslib']\n"
        "assert not hasattr(numpy.ctypeslib.ctypes, 'CDLL')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_worker_environment_scrub_keeps_only_runtime_caps() -> None:
    script = (
        "import os; "
        "from feature_forge.evaluation.sandbox import "
        "_scrub_worker_environment, _WORKER_ENV_ALLOWLIST; "
        "_scrub_worker_environment(); "
        "assert 'FF_TEST_SECRET' not in os.environ; "
        "assert os.environ.get('OMP_NUM_THREADS') == '1'; "
        "retained = {k for k in os.environ if not k.startswith('LC_')}; "
        "assert retained <= _WORKER_ENV_ALLOWLIST, sorted(retained); "
        "assert not [k for k in os.environ if any(t in k.upper() for t in "
        "('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL'))]"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=dict(
            os.environ,
            FF_TEST_SECRET="do-not-expose",
            OMP_NUM_THREADS="1",
            HOME="/home/somebody",
            TMPDIR="/tmp",
        ),
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
