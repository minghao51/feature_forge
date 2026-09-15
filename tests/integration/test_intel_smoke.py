"""Integration smoke test for Intel Extension OpenMP safety.

Skipped automatically when scikit-learn-intelex is not installed or
FF_INTEL_ACCELERATION is false. When active, it spawns a fresh subprocess that
trains the platform's boosting models (pinned to n_jobs=1) under a parallel
(n_jobs=-1) cross-validation -- exactly CorePipeline's Intel path, which uses the
threading backend in the same process. A wall-clock timeout fails the test if a
deadlock regression ever reappears, so the runner can never hang.

NOTE: n_jobs=-1 on the *model* itself is the forbidden configuration (it would
spawn libgomp threads alongside sklearnex's libiomp5 and deadlock). The platform
keeps models at n_jobs=1, which is what intel_training_worker.py exercises.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

from feature_forge.runtime.intel_bootstrap import is_intel_active

_WORKER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "scripts",
    "intel_training_worker.py",
)


@pytest.mark.integration
@pytest.mark.skipif(not is_intel_active(), reason="Intel acceleration not active")
@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None,
    reason="XGBoost optional extra not installed",
)
def test_parallel_cv_does_not_deadlock() -> None:
    try:
        proc = subprocess.run(
            [sys.executable, _WORKER],
            env={**os.environ, "FF_INTEL_ACCELERATION": "true"},
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Parallel CV hung (possible OpenMP deadlock).")
    assert proc.returncode == 0, proc.stderr
