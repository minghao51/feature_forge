#!/usr/bin/env python3
"""Emit a deterministic synthetic PR 8 snapshot fixture; never reads production data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def synthetic_snapshot() -> dict[str, Any]:
    """Return one allowlisted snapshot with pseudonymous synthetic records."""
    payload: dict[str, Any] = {
        "schema_version": "1",
        "snapshot_id": "",
        "catalog_schema_version": 1,
        "datasets": [{"dataset_id": "ds_0123456789abcdef", "run_count": 1, "feature_count": 1}],
        "runs": [
            {
                "run_id": "run_0123456789abcdef",
                "dataset_id": "ds_0123456789abcdef",
                "method_id": "id_0123456789abcdef",
                "model_id": "id_fedcba9876543210",
                "seed": 19,
                "state": "succeeded",
                "stage_count": 4,
                "selected_feature_count": 1,
            }
        ],
        "features": [
            {
                "feature_id": "feature_0123456789abcdef",
                "run_id": "run_0123456789abcdef",
                "disposition": "selected",
                "decision_code": "accepted",
            }
        ],
        "lineage_edges": [],
        "checks": [
            {
                "check_id": "check_0123456789abcdef",
                "severity": "info",
                "passed": True,
                "required": True,
            }
        ],
        "metrics": [
            {
                "metric_id": "metric_0123456789abcdef",
                "run_id": "run_0123456789abcdef",
                "arm": "enhanced",
                "metric_name": "r2",
                "value": 0.75,
            }
        ],
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    payload["snapshot_id"] = f"snap_{digest}"
    return payload


def write_snapshot(destination: Path) -> str:
    """Write the fixture and return its SHA-256 digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(synthetic_snapshot(), indent=2, sort_keys=True) + "\n"
    destination.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def main() -> int:
    args = _args()
    print(write_snapshot(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
