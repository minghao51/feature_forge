"""End-to-end smoke test for Intel Extension (sklearnex) OpenMP safety.

Verifies the deadlock-hardened path actually works:
  - feature_forge patches sklearn first (is_intel_active() is True),
  - the platform's boosting models (pinned to n_jobs=1) train under a parallel
    cross-validation (n_jobs=-1 -> threading backend, same process) without
    hanging. Forking backends are avoided when Intel is active because forking
    after OpenMP is initialized can deadlock in the child.

The training runs as a fresh subprocess under a hard wall-clock timeout so a
regression can never hang your terminal.

Run:
    uv run --extra intel python scripts/intel_smoke_test.py
    FF_INTEL_ACCELERATION=true uv run --extra intel python scripts/intel_smoke_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys

INTEL_SMOKE_TIMEOUT = int(os.environ.get("INTEL_SMOKE_TIMEOUT", "120"))


def main() -> int:
    from feature_forge.runtime.intel_bootstrap import is_intel_active

    if not is_intel_active():
        print(
            "SKIP: Intel acceleration not active.\n"
            "Install with: uv add --optional intel\n"
            "Then run with: FF_INTEL_ACCELERATION=true "
            "uv run --extra intel python scripts/intel_smoke_test.py",
            file=sys.stderr,
        )
        return 2

    worker = os.path.join(os.path.dirname(__file__), "intel_training_worker.py")
    try:
        proc = subprocess.run(
            [sys.executable, worker],
            env={**os.environ, "FF_INTEL_ACCELERATION": "true"},
            capture_output=True,
            text=True,
            timeout=INTEL_SMOKE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(
            f"FAIL: training did not finish within {INTEL_SMOKE_TIMEOUT}s "
            "(possible OpenMP deadlock).",
            file=sys.stderr,
        )
        return 1

    if proc.returncode != 0:
        print(f"FAIL: worker exited {proc.returncode}\n{proc.stderr}", file=sys.stderr)
        return 1

    payload = proc.stdout.strip().splitlines()[-1]
    print("PASS: Intel acceleration active, parallel CV completed (no deadlock).")
    print(f"  results: {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
