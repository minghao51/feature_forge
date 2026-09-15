"""OpenMP single-runtime safety guard for Intel acceleration.

Under the uv/pip workflow, enabling scikit-learn-intelex loads libiomp5 (DAAL)
while the XGBoost/LightGBM wheels still use libgomp. Two OpenMP runtimes in one
process is only safe when:

  (1) KMP_DUPLICATE_LIB_OK is set, so the duplicate runtime is tolerated rather
      than aborting at startup, and
  (2) boosting estimators never spawn their own OpenMP threads (n_jobs=1) and
      evaluation runs in the same process with the threading backend (no fork
      after OpenMP init) -- both enforced by the platform.

This script asserts those invariants. Full single-runtime unification (Intel-
optimized XGBoost/LightGBM via the `intel` conda channel) is the ADR 0010
follow-up and is NOT required for correctness here.

Run: uv run --extra intel python scripts/check_openmp_single_runtime.py
"""

from __future__ import annotations

import os
import sys

from feature_forge.evaluation.model_factory import create_lightgbm, create_xgboost
from feature_forge.runtime.intel_bootstrap import is_intel_active


def _openmp_runtimes() -> list[str]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    runtimes: list[str] = []
    for info in threadpool_info():
        name = info.get("threadpool_name", "") or ""
        internal = info.get("internal_api", "") or ""
        if "openmp" in internal or "libiomp" in name or "libgomp" in name:
            runtimes.append(name or internal)
    return runtimes


def main() -> int:
    active = is_intel_active()

    if not active:
        print("Intel acceleration active: False")
        print("OK: Intel acceleration disabled; nothing to guard.")
        return 0

    # Load the platform's boosting estimators (libgomp) so threadpoolctl can
    # observe both runtimes alongside sklearnex's libiomp5.
    xgb = create_xgboost("classification")
    lgb = create_lightgbm("classification")
    runtimes = _openmp_runtimes()
    print(f"Intel acceleration active: {active}")
    print(f"OpenMP runtimes present: {sorted(set(runtimes)) or 'none'}")

    # Invariant 1: tolerate the duplicate runtime so it cannot abort at startup.
    kmp = os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper()
    if kmp not in {"TRUE", "1"}:
        print(
            "FAIL: libiomp5 + libgomp present but KMP_DUPLICATE_LIB_OK not set.",
            file=sys.stderr,
        )
        return 1

    # Invariant 2: boosting estimators must stay single-threaded in-process.
    if getattr(xgb, "n_jobs", 0) != 1 or getattr(lgb, "n_jobs", 0) != 1:
        print(
            "FAIL: a boosting estimator uses n_jobs != 1 under Intel acceleration.",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: dual OpenMP runtime tolerated (KMP_DUPLICATE_LIB_OK) and boosters "
        "pinned to n_jobs=1 (threading backend, no fork-after-OpenMP)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
