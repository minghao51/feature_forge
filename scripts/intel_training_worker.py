"""Worker: train platform boosting models under parallel CV with sklearnex active.

Run by scripts/intel_smoke_test.py and tests/integration/test_intel_smoke.py as a
fresh subprocess so there is no fork-after-OpenMP hazard. Prints one JSON line
with CV scores + wall time; exits non-zero on failure.
"""

from __future__ import annotations

import json
import time

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from feature_forge.evaluation.model_factory import create_lightgbm, create_xgboost


def main() -> None:
    X, y = make_classification(n_samples=2000, n_features=20, random_state=0)
    # n_jobs=1: libgomp stays idle so it cannot deadlock with libiomp5.
    xgb = create_xgboost("classification")
    lgb = create_lightgbm("classification")
    t = time.perf_counter()
    sx = float(cross_val_score(xgb, X, y, cv=3, n_jobs=-1).mean())
    sl = float(cross_val_score(lgb, X, y, cv=3, n_jobs=-1).mean())
    print(
        json.dumps(
            {
                "xgb_cv": round(sx, 4),
                "lgb_cv": round(sl, 4),
                "seconds": round(time.perf_counter() - t, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
