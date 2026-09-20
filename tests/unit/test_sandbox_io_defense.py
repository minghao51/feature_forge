"""Focused tests for the sandbox's immediate library-I/O defense."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import SandboxValidationError


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
