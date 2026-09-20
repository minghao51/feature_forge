"""Offline smoke tests for the first Hamilton dataflow slice."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from feature_forge.contracts import DatasetRequest, EnvironmentSnapshot
from feature_forge.dataflows.driver import (
    BRONZE_FINAL_VARS,
    build_bronze_driver,
    build_driver,
    build_silver_driver,
)
from feature_forge.dataflows.profile import ExecutionProfile


class _Registry:
    def __init__(self) -> None:
        self.loads = 0

    def info(self, name: str) -> dict[str, Any]:
        return {"name": name, "source": "local", "target": "target", "task": "classification"}

    def load(self, name: str) -> dict[str, Any]:
        self.loads += 1
        return {
            "train": pd.DataFrame({"value": [1, 2, 3, 4], "target": [0, 1, 0, 1]}),
            "test": pd.DataFrame(),
            "target": "target",
            "metadata": self.info(name),
        }


def test_separate_stage_drivers_handoff_source_without_reloading(tmp_path: Path) -> None:
    registry = _Registry()
    request = DatasetRequest(
        name="demo",
        target="target",
        task="classification",
        run_id="split-run",
        case_fingerprint="case",
        split_seed=42,
        cv_folds=2,
        evaluation_holdout_fraction=0.0,
        evaluation_protocol="compatibility",
    )
    environment = EnvironmentSnapshot(
        python_version="3.12",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
    )
    bronze = build_bronze_driver(
        profile=ExecutionProfile.CI, cache_dir=tmp_path / "bronze-cache"
    ).execute(
        BRONZE_FINAL_VARS,
        inputs={
            "dataset_request": request,
            "dataset_registry": registry,
            "environment_snapshot": environment,
            "artifact_store": None,
        },
    )
    silver = build_silver_driver(
        profile=ExecutionProfile.CI, cache_dir=tmp_path / "silver-cache"
    ).execute(
        ["silver_materialization"],
        inputs={
            "dataset_request": request,
            "dataset_computation_request": bronze["dataset_computation_request"],
            "raw_dataset": bronze["raw_dataset"],
            "bronze_snapshot": bronze["bronze_snapshot"],
            "source_metadata": bronze["source_metadata"],
            "bronze_materialization": bronze["bronze_materialization"],
            "environment_snapshot": environment,
            "artifact_store": None,
        },
    )

    assert registry.loads == 1
    assert silver["silver_materialization"].persisted is False


def test_ci_driver_loads_source_once_and_produces_silver_materialization(tmp_path: Path) -> None:
    registry = _Registry()
    driver = build_driver(profile=ExecutionProfile.CI, cache_dir=tmp_path / "hamilton-cache")
    result = driver.execute(
        ["silver_materialization"],
        inputs={
            "dataset_request": DatasetRequest(
                name="demo",
                target="target",
                task="classification",
                run_id="silver-run",
                case_fingerprint="case",
                split_seed=42,
                cv_folds=2,
                evaluation_holdout_fraction=0.0,
                evaluation_protocol="compatibility",
            ),
            "dataset_registry": registry,
            "artifact_store": None,
            "environment_snapshot": EnvironmentSnapshot(
                python_version="3.12",
                operating_system="test",
                architecture="test",
                feature_forge_version="0+test",
            ),
        },
    )
    assert registry.loads == 1
    assert result["silver_materialization"].persisted is False
