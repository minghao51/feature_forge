"""PR 5 coverage: JSON CLI contracts for cache operations and artifact verification.

Every ``--format json`` invocation must emit exactly one parseable JSON
document on stdout (docs/operations.md), destructive cache commands must be
guarded, and artifact list/verify must report broken packages without ever
deleting them.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from feature_forge.cli import main
from feature_forge.contracts import ArtifactNamespace, Layer, ManifestRef
from feature_forge.storage.local import LocalArtifactStore
from tests.unit.test_hamilton_cache import _hex_version, _seed_entry


def _run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    """Run one CLI invocation and assert its stdout is exactly one JSON document."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert out.count("\n") == 1  # --format json is a single compact line
    return exit_code, json.loads(out)


def _manifest(run_id: str, layer: Layer, case: str) -> Any:
    from feature_forge.contracts import EnvironmentSnapshot, RunManifest, RunRequest, RunState

    return RunManifest(
        layer=layer,
        package_kind=layer.value,
        layer_fingerprint=f"{layer.value}-fingerprint",
        run_id=run_id,
        case_fingerprint=case,
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


def _publish_bronze_package(root: Path, run_id: str, case: str) -> ManifestRef:
    """Commit one real verified Bronze package via the production store."""
    store = LocalArtifactStore(root)
    staging = store.begin(ArtifactNamespace(layer=Layer.BRONZE, run_id=run_id))
    staging.write_json("snapshot.json", {"run": run_id})
    staging.set_manifest(_manifest(run_id, Layer.BRONZE, case))
    return store.commit(staging)


def _metadata_keys(cache_dir: Path) -> set[str]:
    connection = sqlite3.connect(cache_dir / "metadata_store.db")
    try:
        return {row[0] for row in connection.execute("SELECT cache_key FROM cache_metadata")}
    finally:
        connection.close()


def _seeded_cache(cache_dir: Path) -> None:
    _seed_entry(
        cache_dir,
        node="bronze_snapshot",
        data_version=_hex_version("bronze"),
        payload=b"x" * 12,
        created_at="2020-01-01 00:00:00",
    )
    _seed_entry(
        cache_dir,
        node="canonical_features",
        data_version=_hex_version("silver"),
        payload=b"y" * 34,
        created_at="2030-06-01 12:00:00",
    )
    _seed_entry(
        cache_dir,
        node="gold_manifest",
        data_version=_hex_version("gold"),
        payload=b"z" * 56,
        created_at="2030-06-01 12:00:01",
    )


class TestCacheCommands:
    def test_status_json_contract_on_seeded_cache(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache_dir = tmp_path / "hamilton"
        _seeded_cache(cache_dir)
        expected_bytes = sum(
            (cache_dir / _hex_version(tag)).stat().st_size for tag in ("bronze", "silver", "gold")
        )

        exit_code, payload = _run_cli(
            capsys, "cache", "status", "--format", "json", "--cache-path", str(cache_dir)
        )

        assert exit_code == 0
        assert payload["schema_version"] == "1"
        assert payload["status"] == "valid"
        cache = payload["cache"]
        assert cache["path"] == str(cache_dir)
        assert cache["exists"] is True
        assert cache["entries"] == 3
        assert cache["serialized_results"] == 3
        assert cache["serialized_bytes"] == expected_bytes
        assert cache["oldest_entry"] == "2020-01-01T00:00:00Z"
        assert cache["newest_entry"] is not None

    def test_status_human_format_is_still_one_document(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache_dir = tmp_path / "hamilton"
        _seeded_cache(cache_dir)

        exit_code = main(["cache", "status", "--cache-path", str(cache_dir)])
        out = capsys.readouterr().out

        assert exit_code == 0
        assert json.loads(out)["status"] == "valid"  # human output still parses fully

    def test_status_on_missing_cache_never_creates_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "absent"

        exit_code, payload = _run_cli(
            capsys, "cache", "status", "--format", "json", "--cache-path", str(missing)
        )

        assert exit_code == 0
        assert payload["cache"]["exists"] is False
        assert payload["cache"]["entries"] == 0
        assert not missing.exists()

    def test_inspect_json_filters_by_node_and_layer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache_dir = tmp_path / "hamilton"
        _seeded_cache(cache_dir)
        argv = ["cache", "inspect", "--format", "json", "--cache-path", str(cache_dir)]

        exit_code, payload = _run_cli(capsys, *argv)
        assert exit_code == 0
        assert payload["status"] == "valid"
        entries = payload["cache"]["entries"]
        assert payload["cache"]["path"] == str(cache_dir)
        assert [entry["node"] for entry in entries] == [
            "gold_manifest",
            "canonical_features",
            "bronze_snapshot",
        ]
        assert entries[0].keys() == {
            "cache_key",
            "data_version",
            "node",
            "layer",
            "created_at",
            "serialized_bytes",
        }
        assert {entry["layer"] for entry in entries} == {"gold", "silver", "bronze"}

        exit_code, payload = _run_cli(capsys, *argv, "--node", "canonical_features")
        assert exit_code == 0
        assert [entry["layer"] for entry in payload["cache"]["entries"]] == ["silver"]

        exit_code, payload = _run_cli(capsys, *argv, "--layer", "gold")
        assert exit_code == 0
        assert [entry["node"] for entry in payload["cache"]["entries"]] == ["gold_manifest"]

        exit_code, payload = _run_cli(capsys, *argv, "--limit", "1")
        assert exit_code == 0
        assert len(payload["cache"]["entries"]) == 1

    def test_gc_dry_run_then_apply_reports_and_deletes_exactly_the_old_entry(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache_dir = tmp_path / "hamilton"
        _seeded_cache(cache_dir)
        old_version = _hex_version("bronze")
        old_size = (cache_dir / old_version).stat().st_size
        argv = [
            "cache",
            "gc",
            "--format",
            "json",
            "--cache-path",
            str(cache_dir),
            "--max-age-days",
            "30",
        ]

        exit_code, payload = _run_cli(capsys, *argv, "--dry-run")

        assert exit_code == 0
        assert payload["cache"]["dry_run"] is True
        assert payload["cache"]["removed_entries"] == 1
        assert payload["cache"]["reclaimed_bytes"] == old_size
        assert (cache_dir / old_version).is_file()  # dry run kept the file

        exit_code, payload = _run_cli(capsys, *argv)

        assert exit_code == 0
        assert payload["cache"]["dry_run"] is False
        assert payload["cache"]["removed_entries"] == 1
        assert payload["cache"]["remaining_entries"] == 2
        assert not (cache_dir / old_version).exists()
        assert _metadata_keys(cache_dir) == {
            f"canonical_features-{_hex_version('silver')[:8]}",
            f"gold_manifest-{_hex_version('gold')[:8]}",
        }

    def test_gc_without_any_policy_fails_closed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache_dir = tmp_path / "hamilton"
        _seeded_cache(cache_dir)

        exit_code, payload = _run_cli(
            capsys, "cache", "gc", "--format", "json", "--cache-path", str(cache_dir)
        )

        assert exit_code == 2
        assert payload["status"] == "invalid"
        assert "max_age_days" in payload["error"]
        assert len(_metadata_keys(cache_dir)) == 3  # nothing was deleted

    def test_clear_requires_force_and_guard_leaves_cache_intact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cache_dir = tmp_path / "hamilton"
        _seeded_cache(cache_dir)

        exit_code, payload = _run_cli(
            capsys, "cache", "clear", "--format", "json", "--cache-path", str(cache_dir)
        )

        assert exit_code == 2
        assert payload["status"] == "invalid"
        assert "--force" in payload["error"]
        assert len(_metadata_keys(cache_dir)) == 3  # guarded clear deleted nothing

        exit_code, payload = _run_cli(
            capsys,
            "cache",
            "clear",
            "--format",
            "json",
            "--cache-path",
            str(cache_dir),
            "--dry-run",
        )

        assert exit_code == 0
        assert payload["cache"]["dry_run"] is True
        assert payload["cache"]["removed_entries"] == 3
        assert len(_metadata_keys(cache_dir)) == 3

        exit_code, payload = _run_cli(
            capsys,
            "cache",
            "clear",
            "--format",
            "json",
            "--cache-path",
            str(cache_dir),
            "--force",
        )

        assert exit_code == 0
        assert payload["cache"]["cleared"] is True
        assert payload["cache"]["removed_entries"] == 3
        assert _metadata_keys(cache_dir) == set()
        assert not (cache_dir / _hex_version("silver")).exists()


class TestArtifactCommands:
    def test_list_reports_complete_and_incomplete_packages_without_deletion(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "artifacts"
        _publish_bronze_package(root, "good-run", case="case-a")
        broken = root / "01_bronze" / "runs" / "broken-run"
        broken.mkdir(parents=True)
        (broken / "manifest.json").write_text("{not valid", encoding="utf-8")
        before = sorted(path.name for path in broken.iterdir())

        exit_code, payload = _run_cli(
            capsys, "artifacts", "list", "--format", "json", "--artifact-root", str(root)
        )

        assert exit_code == 0
        assert payload["status"] == "valid"
        rows = {row["run_id"]: row for row in payload["packages"]}
        assert set(rows) == {"good-run", "broken-run"}

        complete = rows["good-run"]
        assert complete["complete"] is True
        assert complete["layer"] == "bronze"
        assert complete["case_fingerprint"] == "case-a"
        assert complete["manifest_ref"]["run_id"] == "good-run"
        assert complete["manifest_ref"]["sha256"]

        incomplete = rows["broken-run"]
        assert incomplete["complete"] is False
        assert incomplete["error"]

        # Reporting never deletes: the incomplete package is still on disk.
        assert broken.is_dir()
        assert sorted(path.name for path in broken.iterdir()) == before

    def test_list_layer_and_case_filters(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "artifacts"
        _publish_bronze_package(root, "run-a", case="case-a")
        _publish_bronze_package(root, "run-b", case="case-b")
        argv = ["artifacts", "list", "--format", "json", "--artifact-root", str(root)]

        exit_code, payload = _run_cli(capsys, *argv, "--layer", "silver")
        assert exit_code == 0
        assert payload["packages"] == []

        exit_code, payload = _run_cli(capsys, *argv, "--case", "case-b")
        assert exit_code == 0
        assert [row["run_id"] for row in payload["packages"]] == ["run-b"]

    def test_verify_flags_invalid_and_incomplete_packages_without_deletion(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "artifacts"
        _publish_bronze_package(root, "good-run", case="case-a")
        tampered = _publish_bronze_package(root, "tampered-run", case="case-a")
        package = root / "01_bronze" / "runs" / "tampered-run"
        (package / "snapshot.json").write_bytes(b'{"run": "corrupted"}')
        broken = root / "01_bronze" / "runs" / "broken-run"
        broken.mkdir(parents=True)
        (broken / "manifest.json").write_text("{not valid", encoding="utf-8")

        def all_files() -> dict[str, bytes]:
            return {
                str(path.relative_to(root)): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }

        before = all_files()
        exit_code, payload = _run_cli(
            capsys, "artifacts", "verify", "--format", "json", "--artifact-root", str(root)
        )

        assert exit_code == 2
        assert payload["status"] == "invalid"
        results = {row["run_id"]: row for row in payload["packages"]}
        assert set(results) == {"good-run", "tampered-run", "broken-run"}

        assert results["good-run"]["valid"] is True
        assert results["good-run"]["errors"] == []

        assert results["tampered-run"]["valid"] is False
        assert any("hash mismatch" in error for error in results["tampered-run"]["errors"])

        assert results["broken-run"]["valid"] is False
        assert results["broken-run"]["errors"]

        # Verification is read-only: every byte, including broken packages, remains.
        assert all_files() == before
        assert store_verify_still_reports(root, tampered)

    def test_verify_semantic_failure_remains_machine_readable(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from feature_forge import cli
        from feature_forge.exceptions import DatasetError

        root = tmp_path / "artifacts"
        _publish_bronze_package(root, "good-run", case="case-a")

        def fail_semantics(store: Any, row: dict[str, Any]) -> None:
            raise DatasetError("semantic profile mismatch")

        monkeypatch.setattr(cli, "_semantic_verify", fail_semantics)
        exit_code, payload = _run_cli(
            capsys, "artifacts", "verify", "--format", "json", "--artifact-root", str(root)
        )

        assert exit_code == 2
        assert payload["status"] == "invalid"
        assert payload["packages"][0]["errors"] == ["semantic profile mismatch"]

    def test_verify_empty_selection_fails_closed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code, payload = _run_cli(
            capsys,
            "artifacts",
            "verify",
            "--format",
            "json",
            "--artifact-root",
            str(tmp_path / "missing"),
        )

        assert exit_code == 2
        assert payload == {"schema_version": "1", "status": "invalid", "packages": []}

    def test_verify_all_valid_packages_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "artifacts"
        _publish_bronze_package(root, "good-run", case="case-a")

        exit_code, payload = _run_cli(
            capsys, "artifacts", "verify", "--format", "json", "--artifact-root", str(root)
        )

        assert exit_code == 0
        assert payload["status"] == "valid"
        assert len(payload["packages"]) == 1
        assert payload["packages"][0]["valid"] is True


def store_verify_still_reports(root: Path, ref: ManifestRef) -> bool:
    """The tampered package is still detectable by the store after CLI use."""
    return not LocalArtifactStore(root).verify(ref).valid
