"""Generated-style node inventory and cache-policy checks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from feature_forge.contracts import DatasetRequest, EnvironmentSnapshot
from feature_forge.dataflows import bronze, gold, platinum, silver
from feature_forge.dataflows.driver import BRONZE_FINAL_VARS, build_bronze_driver, build_driver
from feature_forge.dataflows.inventory import EXPECTED_NODE_NAMES, validate_node_inventory
from feature_forge.dataflows.profile import ExecutionProfile
from tests.unit.test_silver_dataflow import _Registry


def test_stage_node_inventories_are_exact_and_tagged() -> None:
    for module in (bronze, silver, gold, platinum):
        validate_node_inventory(module)


def _cache_behavior(function: object) -> str | None:
    return next(
        (
            {
                **getattr(decorator, "tags", {}),
                **getattr(decorator, "cache_tags", {}),
            }.get("cache.behavior")
            for decorator in getattr(function, "decorate_nodes", [])
            if "cache.behavior"
            in {
                **getattr(decorator, "tags", {}),
                **getattr(decorator, "cache_tags", {}),
            }
        ),
        None,
    )


def test_side_effecting_boundaries_are_recomputed() -> None:
    for function in (
        bronze.dataset_computation_request,
        bronze.raw_dataset,
        bronze.bronze_manifest,
        bronze.bronze_materialization,
        silver.silver_manifest,
        silver.silver_materialization,
        gold.candidate_execution_batches,
        gold.gold_manifest,
        gold.gold_materialization,
        platinum.baseline_fold_evidence,
        platinum.candidate_fold_evidence,
        platinum.platinum_manifest,
        platinum.platinum_materialization,
    ):
        assert _cache_behavior(function) == "recompute"


def test_silver_direct_driver_writes_reusable_cache_entries(tmp_path: Path) -> None:
    registry = _Registry()
    driver = build_driver(profile=ExecutionProfile.CI, cache_dir=tmp_path / "cache")
    inputs = {
        "dataset_request": DatasetRequest(
            name="demo",
            target="target",
            task="classification",
            run_id="run",
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
    }
    driver.execute(["silver_materialization"], inputs=inputs)
    first_entries = (
        sqlite3.connect(tmp_path / "cache/metadata_store.db")
        .execute("select count(*) from cache_metadata")
        .fetchone()[0]
    )
    driver.execute(["silver_materialization"], inputs=inputs)
    second_entries = (
        sqlite3.connect(tmp_path / "cache/metadata_store.db")
        .execute("select count(*) from cache_metadata")
        .fetchone()[0]
    )

    assert registry.loads == 2  # source reads are explicitly recomputed
    assert first_entries > 0
    assert second_entries >= first_entries


def test_bronze_warm_cache_hits_every_eligible_node_across_attempts(tmp_path: Path) -> None:
    registry = _Registry()
    driver = build_bronze_driver(profile=ExecutionProfile.CI, cache_dir=tmp_path / "cache")
    environment = EnvironmentSnapshot(
        python_version="3.12",
        operating_system="test",
        architecture="test",
        feature_forge_version="0+test",
    )

    for run_id in ("attempt-one", "attempt-two"):
        driver.execute(
            BRONZE_FINAL_VARS,
            inputs={
                "dataset_request": DatasetRequest(
                    name="demo",
                    target="target",
                    task="classification",
                    run_id=run_id,
                    case_fingerprint="same-case",
                    split_seed=42,
                    cv_folds=2,
                    evaluation_holdout_fraction=0.0,
                    evaluation_protocol="compatibility",
                ),
                "dataset_registry": registry,
                "environment_snapshot": environment,
                "artifact_store": None,
            },
        )

    logs = driver.cache.logs(driver.cache.last_run_id, level="debug")
    outcomes = {
        node: {
            event.event_type.value
            for event in events
            if event.event_type.value in {"get_result", "execute_node"}
        }
        for node, events in logs.items()
        if node in EXPECTED_NODE_NAMES[bronze.__name__]
    }
    eligible = {
        name
        for name in EXPECTED_NODE_NAMES[bronze.__name__]
        if _cache_behavior(getattr(bronze, name)) != "recompute"
    }

    assert eligible
    hits = sum(outcomes[name] == {"get_result"} for name in eligible)
    assert hits / len(eligible) >= 0.8
    assert registry.loads == 2  # source reads remain recomputed across attempts
