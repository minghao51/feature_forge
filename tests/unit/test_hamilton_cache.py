"""PR 5 coverage: Hamilton cache inspection, retention, and blast-radius isolation.

Cache fixtures use the production store pair (``hamilton_cache_stores``) so the
SQLite schema, WAL journal mode, and hash-named result layout match reality.
Integration coverage also executes a real Bronze Hamilton stage driver.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from feature_forge.contracts import ArtifactNamespace, FailureClass, Layer
from feature_forge.llm.cache import DiskCache
from feature_forge.storage.hamilton_cache import (
    HamiltonCacheManager,
    hamilton_cache_stores,
)
from feature_forge.storage.local import LocalArtifactStore

_OLD = "2020-01-01 00:00:00"
_NEW = "2030-06-01 12:00:00"


def _hex_version(tag: str) -> str:
    """Return a deterministic 64-hex data_version matching the manager pattern."""
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _seed_entry(
    cache_dir: Path,
    *,
    node: str,
    data_version: str,
    payload: bytes = b"",
    created_at: str | None = None,
    cache_key: str | None = None,
) -> str:
    """Seed one entry through the production stores and return its cache key."""
    key = cache_key or f"{node}-{data_version[:8]}"
    metadata_store, result_store = hamilton_cache_stores(cache_dir)
    metadata_store.set(
        cache_key=key,
        data_version=data_version,
        run_id="seed-run",
        node_name=node,
        code_version="code-v1",
    )
    result_store.set(data_version, payload)
    metadata_store.connection.close()
    if created_at is not None:
        connection = sqlite3.connect(cache_dir / "metadata_store.db")
        try:
            connection.execute(
                "UPDATE cache_metadata SET created_at = ? WHERE cache_key = ?",
                (created_at, key),
            )
            connection.commit()
        finally:
            connection.close()
    return key


def _result_size(cache_dir: Path, data_version: str) -> int:
    """On-disk size of one seeded result file (payload plus StoredResult wrap)."""
    return (cache_dir / data_version).stat().st_size


def _snapshot(root: Path) -> dict[str, bytes]:
    """Capture payload files below ``root`` (metadata DB sidecars excluded)."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.startswith("metadata_store.db")
    }


def test_status_on_missing_cache_reports_without_creating_it(tmp_path: Path) -> None:
    missing = tmp_path / "absent" / "hamilton"

    status = HamiltonCacheManager(missing).status()

    assert status["exists"] is False
    assert status["entries"] == 0
    assert status["serialized_results"] == 0
    assert status["serialized_bytes"] == 0
    assert status["oldest_entry"] is None
    assert status["newest_entry"] is None
    assert not missing.exists()  # inspection never creates the cache


def test_status_reports_seeded_entries_and_sizes(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hamilton"
    version_a, version_b = _hex_version("a"), _hex_version("b")
    _seed_entry(
        cache_dir,
        node="bronze_snapshot",
        data_version=version_a,
        payload=b"x" * 11,
        created_at=_OLD,
    )
    _seed_entry(
        cache_dir,
        node="canonical_features",
        data_version=version_b,
        payload=b"y" * 22,
        created_at=_NEW,
    )
    manager = HamiltonCacheManager(cache_dir)

    status = manager.status()

    assert status["exists"] is True
    assert status["entries"] == 2
    assert status["serialized_results"] == 2
    assert status["serialized_bytes"] == (
        _result_size(cache_dir, version_a) + _result_size(cache_dir, version_b)
    )
    assert status["oldest_entry"] == "2020-01-01T00:00:00Z"
    assert status["newest_entry"] == "2030-06-01T12:00:00Z"
    assert status["total_bytes"] >= status["serialized_bytes"]


def test_inspect_orders_and_filters_by_limit_node_and_layer(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hamilton"
    _seed_entry(
        cache_dir,
        node="bronze_snapshot",
        data_version=_hex_version("1"),
        payload=b"a",
        created_at="2021-01-01 00:00:00",
    )
    _seed_entry(
        cache_dir,
        node="raw_dataset",
        data_version=_hex_version("2"),
        payload=b"bb",
        created_at="2022-01-01 00:00:00",
    )
    _seed_entry(
        cache_dir,
        node="canonical_features",
        data_version=_hex_version("3"),
        payload=b"ccc",
        created_at="2023-01-01 00:00:00",
    )
    _seed_entry(
        cache_dir,
        node="gold_manifest",
        data_version=_hex_version("4"),
        payload=b"dddd",
        created_at="2024-01-01 00:00:00",
    )
    _seed_entry(
        cache_dir,
        node="platinum_checks",
        data_version=_hex_version("5"),
        payload=b"eeeee",
        created_at="2025-01-01 00:00:00",
    )
    _seed_entry(
        cache_dir,
        node="unknown_node",
        data_version=_hex_version("6"),
        payload=b"f",
        created_at="2026-01-01 00:00:00",
    )
    manager = HamiltonCacheManager(cache_dir)

    entries = manager.inspect(limit=None)
    assert [entry.node for entry in entries] == [
        "unknown_node",
        "platinum_checks",
        "gold_manifest",
        "canonical_features",
        "raw_dataset",
        "bronze_snapshot",
    ]
    assert [entry.layer for entry in entries] == [
        None,
        "platinum",
        "gold",
        "silver",
        "bronze",
        "bronze",
    ]
    by_node = {entry.node: entry for entry in entries}
    assert by_node["gold_manifest"].serialized_bytes == _result_size(cache_dir, _hex_version("4"))
    assert by_node["unknown_node"].serialized_bytes == _result_size(cache_dir, _hex_version("6"))

    # ``limit`` keeps only the newest entries.
    assert [entry.node for entry in manager.inspect(limit=2)] == [
        "unknown_node",
        "platinum_checks",
    ]

    # Node and layer filters are exact-match.
    assert [entry.node for entry in manager.inspect(node="raw_dataset")] == ["raw_dataset"]
    assert [entry.node for entry in manager.inspect(layer="silver")] == ["canonical_features"]
    assert [entry.node for entry in manager.inspect(layer="platinum")] == ["platinum_checks"]
    assert manager.inspect(layer="gold", limit=None)[0].layer == "gold"
    assert [entry.node for entry in manager.inspect(layer="bronze", limit=1)] == ["raw_dataset"]
    assert manager.inspect(node="does-not-exist") == []


def test_inspect_validates_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit must be >= 1"):
        HamiltonCacheManager(tmp_path / "hamilton").inspect(limit=0)


def test_age_gc_removes_only_expired_entries_and_their_results(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hamilton"
    old_version, fresh_version = _hex_version("old"), _hex_version("fresh")
    _seed_entry(
        cache_dir,
        node="bronze_snapshot",
        data_version=old_version,
        payload=b"stale",
        created_at=_OLD,
    )
    _seed_entry(
        cache_dir,
        node="bronze_snapshot",
        data_version=fresh_version,
        payload=b"kept",
        created_at=_NEW,
    )
    manager = HamiltonCacheManager(cache_dir)
    old_size = _result_size(cache_dir, old_version)

    result = manager.collect(max_age_days=30)

    assert result["dry_run"] is False
    assert result["cleared"] is False
    assert result["removed_entries"] == 1
    assert result["remaining_entries"] == 1
    assert result["reclaimed_bytes"] == old_size
    assert not (cache_dir / old_version).exists()
    assert b"kept" in (cache_dir / fresh_version).read_bytes()
    remaining_keys = {
        row[0]
        for row in sqlite3.connect(cache_dir / "metadata_store.db").execute(
            "SELECT cache_key FROM cache_metadata"
        )
    }
    assert remaining_keys == {f"bronze_snapshot-{fresh_version[:8]}"}


def test_size_gc_evicts_oldest_versions_until_under_cap(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hamilton"
    small, medium, large = (_hex_version(tag) for tag in ("s", "m", "l"))
    _seed_entry(
        cache_dir,
        node="bronze_snapshot",
        data_version=small,
        payload=b"#" * 100,
        created_at="2021-01-01 00:00:00",
    )
    _seed_entry(
        cache_dir,
        node="canonical_features",
        data_version=medium,
        payload=b"#" * 200,
        created_at="2022-01-01 00:00:00",
    )
    _seed_entry(
        cache_dir,
        node="gold_manifest",
        data_version=large,
        payload=b"#" * 400,
        created_at="2023-01-01 00:00:00",
    )
    manager = HamiltonCacheManager(cache_dir)
    small_size, medium_size, large_size = (
        _result_size(cache_dir, version) for version in (small, medium, large)
    )

    # Cap between the largest version alone and the two newest combined:
    # evict oldest-first until only the large version's bytes remain.
    cap_mib = (large_size + medium_size - 1) / (1024 * 1024)
    result = manager.collect(max_size_mb=cap_mib)

    assert result["removed_entries"] == 2
    assert result["remaining_entries"] == 1
    assert result["reclaimed_bytes"] == small_size + medium_size
    assert not (cache_dir / small).exists()
    assert not (cache_dir / medium).exists()
    assert (cache_dir / large).is_file()


def test_size_gc_is_a_no_op_when_cache_is_under_cap(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hamilton"
    _seed_entry(
        cache_dir, node="bronze_snapshot", data_version=_hex_version("tiny"), payload=b"x" * 10
    )
    manager = HamiltonCacheManager(cache_dir)

    result = manager.collect(max_size_mb=1)

    assert result["removed_entries"] == 0
    assert result["remaining_entries"] == 1
    assert (cache_dir / _hex_version("tiny")).exists()


def test_dry_run_reports_selection_without_deleting(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hamilton"
    old_version = _hex_version("old")
    _seed_entry(
        cache_dir,
        node="bronze_snapshot",
        data_version=old_version,
        payload=b"stale",
        created_at=_OLD,
    )
    _seed_entry(
        cache_dir,
        node="bronze_snapshot",
        data_version=_hex_version("fresh"),
        payload=b"kept",
        created_at=_NEW,
    )
    before = _snapshot(cache_dir)
    manager = HamiltonCacheManager(cache_dir)

    dry_run = manager.collect(max_age_days=30, dry_run=True)
    assert _snapshot(cache_dir) == before  # dry run changed nothing on disk

    applied = manager.collect(max_age_days=30)

    assert dry_run["dry_run"] is True
    assert dry_run["removed_entries"] == applied["removed_entries"] == 1
    assert dry_run["reclaimed_bytes"] == applied["reclaimed_bytes"]
    assert _snapshot(cache_dir) != before
    assert not (cache_dir / old_version).exists()


def test_clear_removes_all_rows_and_hash_results_but_keeps_foreign_files(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "hamilton"
    version_a, version_b = _hex_version("a"), _hex_version("b")
    _seed_entry(cache_dir, node="bronze_snapshot", data_version=version_a, payload=b"one")
    _seed_entry(cache_dir, node="canonical_features", data_version=version_b, payload=b"two!")
    (cache_dir / "cache_logs.jsonl").write_text("{}\n", encoding="utf-8")
    manager = HamiltonCacheManager(cache_dir)
    total_size = _result_size(cache_dir, version_a) + _result_size(cache_dir, version_b)

    result = manager.collect(clear=True)

    assert result["cleared"] is True
    assert result["removed_entries"] == 2
    assert result["reclaimed_bytes"] == total_size
    assert result["remaining_entries"] == 0
    # Every hash-named result is gone; non-result files and the store remain.
    remaining_files = {path.name for path in cache_dir.iterdir() if path.is_file()}
    assert remaining_files <= {
        "metadata_store.db",
        "metadata_store.db-shm",
        "metadata_store.db-wal",
        "cache_logs.jsonl",
    }
    assert (cache_dir / "metadata_store.db").is_file()
    assert (cache_dir / "cache_logs.jsonl").read_text(encoding="utf-8") == "{}\n"
    connection = sqlite3.connect(cache_dir / "metadata_store.db")
    try:
        for table in ("cache_metadata", "history", "run_ids"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        connection.close()


def test_clear_on_missing_cache_creates_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "absent"

    result = HamiltonCacheManager(missing).collect(clear=True)

    assert result["removed_entries"] == 0
    assert result["reclaimed_bytes"] == 0
    assert not missing.exists()


def test_clear_dry_run_keeps_everything(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hamilton"
    _seed_entry(cache_dir, node="bronze_snapshot", data_version=_hex_version("a"), payload=b"one")
    before = _snapshot(cache_dir)

    result = HamiltonCacheManager(cache_dir).collect(clear=True, dry_run=True)

    assert result["dry_run"] is True
    assert result["cleared"] is True
    assert result["removed_entries"] == 1
    assert _snapshot(cache_dir) == before
    connection = sqlite3.connect(cache_dir / "metadata_store.db")
    try:
        assert connection.execute("SELECT COUNT(*) FROM cache_metadata").fetchone()[0] == 1
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_age_days": 0}, "max_age_days must be > 0"),
        ({"max_age_days": -1}, "max_age_days must be > 0"),
        ({"max_size_mb": 0}, "max_size_mb must be > 0"),
        ({"clear": True, "max_age_days": 1}, "clear cannot be combined"),
        ({}, "provide max_age_days"),
    ],
)
def test_collect_validates_policy_arguments(
    tmp_path: Path, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        HamiltonCacheManager(tmp_path / "hamilton").collect(**kwargs)


def test_unreadable_metadata_raises_actionable_value_error(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hamilton"
    cache_dir.mkdir()
    (cache_dir / "metadata_store.db").write_bytes(b"definitely not sqlite")

    with pytest.raises(ValueError, match="unreadable"):
        HamiltonCacheManager(cache_dir).inspect()


def test_runtime_store_quarantines_corrupt_metadata_and_recomputes(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hamilton"
    cache_dir.mkdir()
    (cache_dir / "metadata_store.db").write_bytes(b"definitely not sqlite")

    metadata_store, result_store = hamilton_cache_stores(cache_dir)
    version = _hex_version("recovered")
    metadata_store.set(
        cache_key="recovered-key",
        data_version=version,
        run_id="recovered-run",
        node_name="canonical_features",
        code_version="code-v1",
    )
    result_store.set(version, b"recomputed")
    metadata_store.connection.close()

    status = HamiltonCacheManager(cache_dir).status()
    assert status["entries"] == 1
    assert status["serialized_results"] == 1
    quarantined = list(cache_dir.glob("metadata_store.db.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"definitely not sqlite"


def test_gc_leaves_sibling_medallion_package_and_llm_cache_untouched(
    tmp_path: Path,
) -> None:
    """Age GC and full clear never leave the Hamilton cache directory."""
    root = tmp_path / "artifacts"
    cache_dir = root / "control" / "cache" / "hamilton"

    store = LocalArtifactStore(root)
    staging = store.begin(ArtifactNamespace(layer=Layer.BRONZE, run_id="bronze-1"))
    staging.write_json("snapshot.json", {"rows": 4})
    staging.set_manifest(_manifest("bronze-1", Layer.BRONZE))
    ref = store.commit(staging)
    assert store.verify(ref).valid

    with DiskCache(cache_dir=str(tmp_path / "llm_cache")) as llm_cache:
        llm_cache.set("0" * 64, {"response": "cached"})

        _seed_entry(
            cache_dir,
            node="bronze_snapshot",
            data_version=_hex_version("a"),
            payload=b"one",
            created_at=_OLD,
        )
        _seed_entry(
            cache_dir,
            node="canonical_features",
            data_version=_hex_version("b"),
            payload=b"two!",
            created_at=_OLD,
        )
        package_before = _snapshot(root / "01_bronze")

        manager = HamiltonCacheManager(cache_dir)
        manager.collect(max_age_days=1)
        manager.collect(clear=True)

        # Sibling medallion evidence is byte-identical.
        assert _snapshot(root / "01_bronze") == package_before
        assert store.verify(ref).valid
        # The mandatory LLM response cache is a different store and is intact.
        assert llm_cache.get("0" * 64) == {"response": "cached"}
        assert llm_cache.stats()["items"] == 1

    # The Hamilton cache itself is emptied of hash-named results.
    assert not (cache_dir / _hex_version("a")).exists()
    assert not (cache_dir / _hex_version("b")).exists()


def test_cache_lock_contention_is_classified_as_resource_failure() -> None:
    from feature_forge.experiment.hamilton_executor import _classify_failure

    assert (
        _classify_failure(sqlite3.OperationalError("database is locked")) is FailureClass.RESOURCE
    )


def test_two_spawned_workers_share_cache_without_corruption(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared-cache"
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        futures = [pool.submit(_execute_bronze_stage, cache_dir) for _ in range(2)]
        for future in futures:
            future.result(timeout=60)

    manager = HamiltonCacheManager(cache_dir)
    assert manager.status()["entries"] > 0
    assert manager.inspect(limit=None)
    with sqlite3.connect(cache_dir / "metadata_store.db") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_manager_reads_real_bronze_stage_cache(tmp_path: Path) -> None:
    """End-to-end: a real stage driver populates the cache the manager inspects."""
    cache_dir = tmp_path / "cache"
    _execute_bronze_stage(cache_dir)
    manager = HamiltonCacheManager(cache_dir)

    status = manager.status()
    assert status["exists"] is True
    assert status["entries"] > 0
    assert status["oldest_entry"] is not None

    entries = manager.inspect(limit=None)
    assert {entry.layer for entry in entries} == {"bronze"}
    assert {"raw_dataset", "bronze_snapshot"} <= {entry.node for entry in entries}
    filtered = manager.inspect(node="raw_dataset")
    assert len(filtered) == 1
    assert filtered[0].layer == "bronze"


def test_real_stage_cache_results_are_tracked_and_collected(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _execute_bronze_stage(cache_dir)
    manager = HamiltonCacheManager(cache_dir)

    status = manager.status()
    assert status["serialized_results"] > 0
    assert status["serialized_bytes"] > 0

    result = manager.collect(clear=True)
    assert result["reclaimed_bytes"] == status["serialized_bytes"]
    leftover = [
        path.name
        for path in cache_dir.iterdir()
        if path.is_file() and not path.name.startswith("metadata_store.db")
    ]
    assert leftover == []


def _execute_bronze_stage(cache_dir: Path) -> None:
    """Populate ``cache_dir`` with a genuine Bronze Hamilton stage cache."""
    from feature_forge.contracts import DatasetRequest, EnvironmentSnapshot
    from feature_forge.dataflows.driver import BRONZE_FINAL_VARS, build_bronze_driver
    from feature_forge.dataflows.profile import ExecutionProfile
    from tests.unit.test_silver_dataflow import _Registry

    build_bronze_driver(profile=ExecutionProfile.CI, cache_dir=cache_dir).execute(
        BRONZE_FINAL_VARS,
        inputs={
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
            "dataset_registry": _Registry(),
            "artifact_store": None,
            "environment_snapshot": EnvironmentSnapshot(
                python_version="3.12",
                operating_system="test",
                architecture="test",
                feature_forge_version="0+test",
            ),
        },
    )


def _manifest(run_id: str, layer: Layer) -> Any:
    from feature_forge.contracts import EnvironmentSnapshot, RunManifest, RunRequest, RunState

    return RunManifest(
        layer=layer,
        package_kind=layer.value,
        layer_fingerprint=f"{layer.value}-fingerprint",
        run_id=run_id,
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="demo", method="dummy", model="rf", seed=42),
        environment=EnvironmentSnapshot(
            python_version="3.12",
            operating_system="test",
            architecture="test",
            feature_forge_version="0+test",
        ),
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        upstream_manifests=[],
    )
