#!/usr/bin/env python3
"""Qualify warm Hamilton cache behavior on provider-free Bronze/Silver DAGs."""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from feature_forge.contracts import DatasetRequest, EnvironmentSnapshot
from feature_forge.dataflows import bronze, silver
from feature_forge.dataflows.driver import SILVER_FINAL_VARS, build_driver
from feature_forge.dataflows.inventory import EXPECTED_NODE_NAMES
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.storage.hamilton_cache import HamiltonCacheManager


class _SyntheticRegistry:
    def __init__(self, rows: int) -> None:
        self.loads = 0
        self.rows = rows

    def info(self, name: str) -> dict[str, str]:
        return {"source": "memory", "target": "target", "task": "classification"}

    def load(self, name: str) -> dict[str, Any]:
        self.loads += 1
        return {
            "train": pd.DataFrame(
                {
                    "value": list(range(self.rows)),
                    "target": [index % 2 for index in range(self.rows)],
                }
            ),
            "metadata": {"source": "memory", "target": "target", "task": "classification"},
        }


def _cache_behavior(function: object) -> str | None:
    for decorator in getattr(function, "decorate_nodes", []):
        tags = {
            **getattr(decorator, "tags", {}),
            **getattr(decorator, "cache_tags", {}),
        }
        if "cache.behavior" in tags:
            return str(tags["cache.behavior"])
    return None


def qualify(cache_dir: Path, *, rows: int = 16) -> dict[str, Any]:
    """Execute cold/warm attempts and return machine-readable measurements."""
    if rows < 4:
        raise ValueError("rows must be >= 4")
    registry = _SyntheticRegistry(rows)
    driver = build_driver(profile=ExecutionProfile.CI, cache_dir=cache_dir)
    environment = EnvironmentSnapshot(
        python_version="qualification",
        operating_system="qualification",
        architecture="qualification",
        feature_forge_version="0+qualification",
    )
    durations: list[float] = []
    for attempt in ("qualification-cold", "qualification-warm"):
        request = DatasetRequest(
            name="synthetic",
            target="target",
            task="classification",
            run_id=attempt,
            case_fingerprint="qualification-case",
            split_seed=42,
            cv_folds=2,
            evaluation_protocol="compatibility",
            evaluation_holdout_fraction=0.0,
        )
        started = time.perf_counter()
        driver.execute(
            SILVER_FINAL_VARS,
            inputs={
                "dataset_request": request,
                "dataset_registry": registry,
                "environment_snapshot": environment,
                "artifact_store": None,
            },
        )
        durations.append((time.perf_counter() - started) * 1000)

    logs = driver.cache.logs(driver.cache.last_run_id, level="debug")
    eligible = {
        name
        for module in (bronze, silver)
        for name in EXPECTED_NODE_NAMES[module.__name__]
        if _cache_behavior(getattr(module, name)) != "recompute"
    }
    hits = sum(
        any(event.event_type.value == "get_result" for event in logs.get(name, []))
        for name in eligible
    )
    status = HamiltonCacheManager(cache_dir).status()
    return {
        "schema_version": "1",
        "dataset_rows": rows,
        "cold_duration_ms": durations[0],
        "warm_duration_ms": durations[1],
        "eligible_nodes": len(eligible),
        "warm_hits": hits,
        "warm_hit_rate": hits / len(eligible),
        "source_loads": registry.loads,
        "serialized_results": status["serialized_results"],
        "serialized_bytes": status["serialized_bytes"],
        "cache_total_bytes": status["total_bytes"],
        "provider_calls": 0,
        "sandbox_executions": 0,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--rows",
        type=int,
        default=16,
        help="Synthetic row count (default: 16; use a representative size for timing).",
    )
    args = parser.parse_args(argv)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    cache_dir = args.cache_dir
    if cache_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="feature-forge-hamilton-")
        cache_dir = Path(temporary.name)
    try:
        report = qualify(cache_dir, rows=args.rows)
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        passed = (
            report["warm_hit_rate"] >= 0.8
            and report["source_loads"] == 2
            and report["serialized_results"] > 0
        )
        return 0 if passed else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
