"""Tests for the evidence-only PR 8 readiness artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.emit_pr8_snapshot_spike import synthetic_snapshot
from scripts.measure_pr8_readiness import PROFILES, iter_records, write_corpus

ROOT = Path(__file__).parents[2]
SCHEMA_PATH = ROOT / "docs/decisions/schemas/pr8-public-snapshot-v1.schema.json"


def test_synthetic_profiles_match_the_readiness_handoff() -> None:
    assert PROFILES["small"].counts() == {
        "runs": 100,
        "features": 2_000,
        "lineage_edges": 5_000,
        "stages": 400,
        "checks": 800,
        "fold_metrics": 500,
    }
    assert PROFILES["expected"].counts() == {
        "runs": 5_000,
        "features": 100_000,
        "lineage_edges": 300_000,
        "stages": 20_000,
        "checks": 40_000,
        "fold_metrics": 25_000,
    }


def test_synthetic_corpus_is_byte_stable_across_destinations(tmp_path: Path) -> None:
    first_digest, first_bytes = write_corpus(PROFILES["small"], tmp_path / "first.jsonl", 19)
    second_digest, second_bytes = write_corpus(PROFILES["small"], tmp_path / "second.jsonl", 19)

    assert first_digest == second_digest
    assert first_bytes == second_bytes
    assert (tmp_path / "first.jsonl").read_bytes() == (tmp_path / "second.jsonl").read_bytes()


def test_synthetic_records_have_no_sensitive_payload_fields() -> None:
    records = list(iter_records(PROFILES["small"], 19))
    serialized = json.dumps(records, sort_keys=True)

    assert len(records) == sum(PROFILES["small"].counts().values())
    assert not any(
        token in serialized.lower()
        for token in ("api_key", "authorization", "prompt", "prediction", "filesystem", "memory")
    )


def test_public_snapshot_schema_is_deny_by_default() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert all(
        definition["additionalProperties"] is False for definition in schema["$defs"].values()
    )
    assert schema["properties"]["schema_version"]["const"] == "1"
    assert schema["properties"]["catalog_schema_version"]["const"] == 1


def test_synthetic_snapshot_fixture_is_allowlisted_and_self_identifying() -> None:
    snapshot = synthetic_snapshot()
    serialized = json.dumps(snapshot, sort_keys=True)

    assert snapshot["snapshot_id"].startswith("snap_")
    assert set(snapshot) == {
        "schema_version",
        "snapshot_id",
        "catalog_schema_version",
        "datasets",
        "runs",
        "features",
        "lineage_edges",
        "checks",
        "metrics",
    }
    assert not any(
        token in serialized.lower()
        for token in ("api_key", "authorization", "prompt", "prediction", "filesystem", "memory")
    )
