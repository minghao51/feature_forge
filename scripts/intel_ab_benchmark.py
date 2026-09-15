"""A/B benchmark: Intel acceleration ON vs OFF on one dataset x model.

Because the sklearnex patch is import-time and process-global, the two modes are
run as separate subprocesses. Each subprocess trains XGBoost + LightGBM (pinned
to n_jobs=1, as the platform enforces) and reports CV score + wall time. This
both validates any speedup and documents the numerical delta DAAL introduces for
reproducibility.

Run: uv run --extra intel python scripts/intel_ab_benchmark.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

_WORKER = """
import json, time
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from feature_forge.evaluation.model_factory import create_xgboost, create_lightgbm
from feature_forge.runtime.intel_bootstrap import is_intel_active

X, y = make_classification(n_samples=5000, n_features=30, random_state=0)
xgb = create_xgboost("classification")
lgb = create_lightgbm("classification")
t = time.perf_counter()
sx = float(cross_val_score(xgb, X, y, cv=3, n_jobs=1).mean())
sl = float(cross_val_score(lgb, X, y, cv=3, n_jobs=1).mean())
print(json.dumps({
    "intel_active": is_intel_active(),
    "xgb_auc": round(sx, 4),
    "lgb_auc": round(sl, 4),
    "seconds": round(time.perf_counter() - t, 2),
}))
"""


def _run(mode: str) -> dict[str, object]:
    env = {**os.environ, "FF_INTEL_ACCELERATION": mode}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_WORKER)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed ({mode}):\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _cell(value: object) -> str:
    return f"{value!s:>12}"


def main() -> int:
    on = _run("true")
    off = _run("false")
    print(f"{'metric':<14}{_cell('intel_on')}{_cell('intel_off')}")
    print(f"{'intel_active':<14}{_cell(on['intel_active'])}{_cell(off['intel_active'])}")
    print(f"{'xgb_auc':<14}{_cell(on['xgb_auc'])}{_cell(off['xgb_auc'])}")
    print(f"{'lgb_auc':<14}{_cell(on['lgb_auc'])}{_cell(off['lgb_auc'])}")
    print(f"{'seconds':<14}{_cell(on['seconds'])}{_cell(off['seconds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
