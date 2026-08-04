"""PR6 catalog, verification CLI, and lifecycle telemetry tests."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import pytest

from feature_forge.cli import main
from feature_forge.config import Settings
from feature_forge.contracts import (
    ArtifactNamespace,
    EnvironmentSnapshot,
    Layer,
    RunManifest,
    RunRequest,
    RunState,
)
from feature_forge.contracts.catalog import CLIExitCode, VerificationStatus
from feature_forge.contracts.orchestration import FailureRecord, StageDisposition
from feature_forge.contracts.stages import FailureClass
from feature_forge.exceptions import DatasetError
from feature_forge.experiment.case_executor import CaseComputation, ExperimentCaseExecutor
from feature_forge.experiment.execution import ExperimentCase, ExperimentResult
from feature_forge.experiment.lifecycle import LocalRunRepository
from feature_forge.experiment.tracker import NoOpTracker
from feature_forge.observability.hamilton_adapter import (
    HamiltonObservedDriver,
    HamiltonTelemetryRecorder,
    create_hamilton_lifecycle_adapter,
)
from feature_forge.storage.catalog import LocalCatalog
from feature_forge.storage.local import LocalArtifactStore
from feature_forge.verification.service import VerificationService


def _manifest(run_id: str, layer: Layer = Layer.BRONZE) -> RunManifest:
    now = datetime.now(UTC)
    return RunManifest(
        layer=layer,
        package_kind=layer.value,
        layer_fingerprint=f"{layer.value}-fingerprint",
        run_id=run_id,
        case_fingerprint="case-fingerprint",
        state=RunState.SUCCEEDED,
        request=RunRequest(dataset="demo", method="dummy", model="rf", seed=42),
        environment=EnvironmentSnapshot(
            python_version="3.13",
            operating_system="test",
            architecture="test",
            feature_forge_version="0+test",
        ),
        created_at=now,
        completed_at=now,
    )


def _package(store: LocalArtifactStore, run_id: str, layer: Layer = Layer.BRONZE):
    staging = store.begin(ArtifactNamespace(layer=layer, run_id=run_id))
    staging.write_json("evidence.json", {"run_id": run_id})
    staging.set_manifest(_manifest(run_id, layer))
    return store.commit(staging)


def _service(tmp_path: Path) -> VerificationService:
    return VerificationService(
        artifact_root=tmp_path / "lake",
        catalog_path=tmp_path / "control/catalog.duckdb",
        lifecycle_root=tmp_path / "control",
        stale_run_seconds=60,
    )


def test_importing_cli_does_not_import_optional_pipeline_dependencies() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; import feature_forge.cli; assert 'duckdb' not in sys.modules; assert 'hamilton' not in sys.modules",
    ]
    subprocess.run(command, check=True)


def test_incremental_and_rebuild_have_identical_logical_catalog(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    refs = [_package(store, "run-b"), _package(store, "run-a")]
    incremental = LocalCatalog(tmp_path / "incremental.duckdb")
    for ref in reversed(refs):
        incremental.index_package(store, ref)
    rebuilt = LocalCatalog(tmp_path / "rebuilt.duckdb")
    first = rebuilt.rebuild(store)
    first_digest = rebuilt.logical_digest()
    second = rebuilt.rebuild(store)
    assert first.status is VerificationStatus.VALID
    assert first.logical_digest == incremental.logical_digest() == first_digest
    assert second.logical_digest == first_digest


def test_package_index_is_transactional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import feature_forge.storage.catalog as catalog_module

    store = LocalArtifactStore(tmp_path / "lake")
    ref = _package(store, "rollback")
    catalog = LocalCatalog(tmp_path / "catalog.duckdb")
    original = catalog_module._insert_normalized

    def fail_after_insert(connection, data):
        original(connection, data)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(catalog_module, "_insert_normalized", fail_after_insert)
    with pytest.raises(RuntimeError, match="injected failure"):
        catalog.index_package(store, ref)
    assert catalog.manifest_rows() == []


def test_corrupt_package_is_reported_and_not_indexed(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    ref = _package(store, "corrupt")
    store.resolve(
        __import__("feature_forge.contracts", fromlist=["ArtifactRef"]).ArtifactRef(
            layer=Layer.BRONZE, run_id="corrupt", relative_path="evidence.json"
        )
    ).write_text("tampered", encoding="utf-8")
    catalog = LocalCatalog(tmp_path / "catalog.duckdb")
    report = catalog.rebuild(store)
    assert report.status is VerificationStatus.INVALID
    assert report.indexed_manifests == 0
    assert report.issues[0].code == "invalid_package"
    assert catalog.manifest_rows() == []
    assert ref.run_id == "corrupt"


def test_catalog_status_detects_stale_and_unindexed_rows(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    first = _package(store, "indexed")
    catalog = LocalCatalog(tmp_path / "control/catalog.duckdb")
    catalog.index_package(store, first)
    _package(store, "new")
    result = _service(tmp_path).verify_catalog()
    assert result.status is VerificationStatus.INVALID
    assert {issue.code for issue in result.issues} == {
        "catalog_row_mismatch",
        "unindexed_package",
    }


def test_verification_commands_are_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    _package(store, "read-only")
    before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    code = main(
        [
            "--format",
            "json",
            "--artifact-root",
            str(tmp_path / "lake"),
            "--catalog-path",
            str(tmp_path / "missing.duckdb"),
            "--lifecycle-root",
            str(tmp_path / "control"),
            "verify",
            "run",
            "read-only",
        ]
    )
    after = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert code == CLIExitCode.OK
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    assert before == after
    assert not (tmp_path / "missing.duckdb").exists()


def test_cli_exit_codes_and_json_are_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = [
        "--format",
        "json",
        "--artifact-root",
        str(tmp_path / "lake"),
        "--catalog-path",
        str(tmp_path / "catalog.duckdb"),
        "--lifecycle-root",
        str(tmp_path / "control"),
    ]
    assert main([*common, "verify", "run", "missing"]) == CLIExitCode.MISSING
    missing = json.loads(capsys.readouterr().out)
    assert missing["schema_version"] == "1"
    assert missing["status"] == "missing"
    _package(LocalArtifactStore(tmp_path / "lake"), "valid")
    assert main([*common, "verify", "artifact", "bronze:valid:evidence.json"]) == CLIExitCode.OK
    assert json.loads(capsys.readouterr().out)["status"] == "valid"


def test_stale_running_journal_is_reported(tmp_path: Path) -> None:
    repository = LocalRunRepository(tmp_path / "control")
    event = repository.record(
        run_id="stale", case_id="stale", event_type="case_running", state=RunState.RUNNING
    )
    path = tmp_path / "control/runs/stale/events.jsonl"
    stale = event.model_copy(update={"occurred_at": datetime.now(UTC) - timedelta(hours=2)})
    path.write_text(json.dumps(stale.model_dump(mode="json")) + "\n", encoding="utf-8")
    result = _service(tmp_path).verify_run("stale")
    assert result.status is VerificationStatus.INVALID
    assert result.issues[0].code == "stale_run"


def test_telemetry_redacts_secrets_and_never_records_inputs_or_results() -> None:
    events: list[dict[str, object]] = []
    recorder = HamiltonTelemetryRecorder(run_id="run", case_id="case", sink=events.append)
    recorder.before(
        hamilton_run_id="h",
        node_name="gold",
        task_id="1",
        tags={"layer": "gold", "owner": "Authorization: Bearer owner-secret"},
    )
    recorder.after(
        hamilton_run_id="h",
        node_name="gold",
        task_id="1",
        tags={"layer": "gold"},
        success=False,
        error=RuntimeError(
            "Authorization: Bearer auth-secret OPENAI_API_KEY=provider-secret "
            "standalone sk-proj-1234567890abcdef"
        ),
    )
    rendered = json.dumps(events)
    assert "auth-secret" not in rendered
    assert "provider-secret" not in rendered
    assert "owner-secret" not in rendered
    assert "sk-proj-1234567890abcdef" not in rendered
    assert "<redacted>" in rendered
    assert "node_kwargs" not in rendered
    assert "result" not in rendered


def test_cache_outcomes_use_hamilton_cache_events() -> None:
    events: list[dict[str, object]] = []
    recorder = HamiltonTelemetryRecorder(run_id="run", case_id="case", sink=events.append)
    recorder.cache_events(
        [
            SimpleNamespace(
                event_type=SimpleNamespace(value="get_result"),
                run_id="h",
                node_name="silver",
                task_id=None,
            ),
            SimpleNamespace(
                event_type=SimpleNamespace(value="execute_node"),
                run_id="h",
                node_name="gold",
                task_id="1",
            ),
        ]
    )
    assert [event["cache_outcome"] for event in events] == ["hit", "miss"]
    assert create_hamilton_lifecycle_adapter(recorder).__class__.__name__ == "Adapter"


def test_observed_driver_flushes_cache_events_after_execution() -> None:
    events: list[dict[str, object]] = []
    recorder = HamiltonTelemetryRecorder(run_id="run", case_id="case", sink=events.append)
    cache_event = SimpleNamespace(
        event_type=SimpleNamespace(value="get_result"),
        run_id="h",
        node_name="silver",
        task_id=None,
        timestamp=1.0,
    )

    class Cache:
        run_ids: ClassVar[list[str]] = ["h"]
        last_run_id = "h"

        def logs(self, run_id: str, level: str) -> dict[str, list[object]]:
            return {"silver": [cache_event]}

    class Driver:
        cache = Cache()

        def execute(self, *args: object, **kwargs: object) -> dict[str, str]:
            return {"value": "ok"}

    driver = HamiltonObservedDriver(Driver(), recorder)
    assert driver.execute(["value"]) == {"value": "ok"}
    assert events[0]["cache_outcome"] == "hit"


def test_tracker_reference_api_logs_only_reference() -> None:
    tracker = NoOpTracker(project="test")
    with patch.object(tracker, "log_params") as logged:
        tracker.log_artifact_reference("platinum:run:manifest.json", sha256="a" * 64)
    logged.assert_called_once_with(
        {
            "artifact_uri": "platinum:run:manifest.json",
            "artifact_type": "manifest",
            "artifact_sha256": "a" * 64,
        }
    )


@pytest.mark.parametrize(
    ("policy", "expected_state", "has_error"),
    [("optional", "partial", False), ("required", "failed", True)],
)
def test_tracker_failure_policy_preserves_computation(
    policy: str,
    expected_state: str,
    has_error: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTracker(NoOpTracker):
        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            raise RuntimeError("tracker unavailable")

    monkeypatch.setattr(
        CaseComputation,
        "compute",
        lambda self, payload: ExperimentResult(
            payload.case.dataset,
            payload.case.method,
            payload.case.model,
            payload.case.seed,
            cv_score=0.8,
            state="succeeded",
            manifest_uri="platinum:run:manifest.json",
        ),
    )
    executor = ExperimentCaseExecutor(
        Settings(), FailingTracker(project="test"), tracker_failure_policy=policy
    )
    result = executor.execute(ExperimentCase("demo", "dummy", "rf", 42))
    assert result.state == expected_state
    assert (result.error is not None) is has_error
    assert result.cv_score == 0.8


def test_catalog_normalizes_feature_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import feature_forge.storage.catalog as catalog_module

    monkeypatch.setattr(catalog_module, "_semantic_verify", lambda store, ref: None)
    store = LocalArtifactStore(tmp_path / "lake")
    staging = store.begin(ArtifactNamespace(layer=Layer.GOLD, run_id="gold"))
    staging.write_json(
        "candidates.json",
        [
            {
                "candidate_id": "candidate-1",
                "name": "ratio",
                "batch_index": 0,
                "code_path": "code/ratio.py",
                "specification": {"op": "divide"},
            }
        ],
    )
    staging.write_json(
        "decisions.json",
        [
            {
                "candidate_id": "candidate-1",
                "state": "accepted",
                "reason_code": "valid",
                "reason": "passed",
            }
        ],
    )
    staging.set_manifest(_manifest("gold", Layer.GOLD))
    ref = store.commit(staging)
    catalog = LocalCatalog(tmp_path / "catalog.duckdb")
    catalog.index_package(store, ref)
    dump = catalog.logical_dump()
    assert len(dump["features"]) == 1
    assert len(dump["feature_decisions"]) == 1


def test_artifact_verification_rejects_traversal_metadata_and_undeclared_paths(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    _package(store, "safe")
    service = _service(tmp_path)
    assert service.verify_artifact("bronze:safe:../outside").status is VerificationStatus.INVALID
    assert service.verify_artifact("bronze:safe:manifest.json").status is VerificationStatus.INVALID
    assert (
        service.verify_artifact("bronze:safe:not-declared.json").status
        is VerificationStatus.INVALID
    )
    valid = service.verify_artifact("bronze:safe:evidence.json")
    assert valid.status is VerificationStatus.VALID
    assert valid.details["declared"] is True


def test_semantic_loader_failure_blocks_index_and_run_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import feature_forge.storage.catalog as catalog_module

    store = LocalArtifactStore(tmp_path / "lake")
    ref = _package(store, "semantic")

    def reject(store: LocalArtifactStore, ref: object) -> None:
        raise DatasetError("semantic lineage mismatch")

    monkeypatch.setattr(catalog_module, "_semantic_verify", reject)
    with pytest.raises(DatasetError, match="semantic lineage mismatch"):
        LocalCatalog(tmp_path / "catalog.duckdb").index_package(store, ref)
    result = _service(tmp_path).verify_run("semantic")
    assert result.status is VerificationStatus.INVALID
    assert result.issues[0].code == "semantic_verification_failed"


def test_catalog_reconciles_dependent_rows_not_only_manifests(tmp_path: Path) -> None:
    import duckdb

    store = LocalArtifactStore(tmp_path / "lake")
    _package(store, "rows")
    catalog_path = tmp_path / "control/catalog.duckdb"
    LocalCatalog(catalog_path).rebuild(store)
    connection = duckdb.connect(str(catalog_path))
    connection.execute("DELETE FROM artifacts")
    connection.close()
    result = _service(tmp_path).verify_catalog()
    assert result.status is VerificationStatus.INVALID
    assert any(
        issue.code == "catalog_row_mismatch" and issue.message.endswith("artifacts")
        for issue in result.issues
    )


def test_catalog_reconciliation_is_filesystem_read_only(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    _package(store, "read-catalog")
    catalog = LocalCatalog(tmp_path / "control/catalog.duckdb")
    catalog.rebuild(store, lifecycle_root=tmp_path / "control")
    before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    result = _service(tmp_path).verify_catalog()
    after = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert result.status is VerificationStatus.VALID
    assert before == after


def test_catalog_reports_malformed_journal_without_reparsing(tmp_path: Path) -> None:
    catalog = LocalCatalog(tmp_path / "control/catalog.duckdb")
    catalog.rebuild(LocalArtifactStore(tmp_path / "lake"))
    journal = tmp_path / "control/runs/malformed/events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text("{not valid json}\n", encoding="utf-8")

    result = _service(tmp_path).verify_catalog()

    assert result.status is VerificationStatus.INVALID
    assert [issue.code for issue in result.issues] == ["invalid_journal"]


def test_stage_event_after_running_does_not_hide_stale_run(tmp_path: Path) -> None:
    repository = LocalRunRepository(tmp_path / "control")
    running = repository.record(
        run_id="stage-only",
        case_id="stage-only",
        event_type="case_running",
        state=RunState.RUNNING,
    )
    stage = repository.record(
        run_id="stage-only",
        case_id="stage-only",
        event_type="stage_terminal",
        state=RunState.SUCCEEDED,
        stage="silver",
        layer=Layer.SILVER,
    )
    old = datetime.now(UTC) - timedelta(hours=2)
    path = tmp_path / "control/runs/stage-only/events.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(event.model_copy(update={"occurred_at": old}).model_dump(mode="json"))
            for event in (running, stage)
        )
        + "\n",
        encoding="utf-8",
    )
    result = _service(tmp_path).verify_run("stage-only")
    assert result.status is VerificationStatus.INVALID
    assert result.issues[0].code == "stale_run"


def test_case_terminal_after_running_prevents_stale_classification(tmp_path: Path) -> None:
    repository = LocalRunRepository(tmp_path / "control")
    running = repository.record(
        run_id="terminal",
        case_id="terminal",
        event_type="case_running",
        state=RunState.RUNNING,
    )
    terminal = repository.record(
        run_id="terminal",
        case_id="terminal",
        event_type="case_terminal",
        state=RunState.SUCCEEDED,
    )
    old = datetime.now(UTC) - timedelta(hours=2)
    path = tmp_path / "control/runs/terminal/events.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(event.model_copy(update={"occurred_at": old}).model_dump(mode="json"))
            for event in (running, terminal)
        )
        + "\n",
        encoding="utf-8",
    )
    result = _service(tmp_path).verify_run("terminal")
    assert result.status is VerificationStatus.VALID


def test_catalog_persists_complete_typed_run_event(tmp_path: Path) -> None:
    repository = LocalRunRepository(tmp_path / "control")
    repository.record(
        run_id="typed",
        case_id="typed",
        event_type="stage_terminal",
        state=RunState.FAILED,
        stage="gold",
        layer=Layer.GOLD,
        disposition=StageDisposition.EXECUTED,
        failure=FailureRecord(
            failure_class=FailureClass.DETERMINISTIC,
            error_type="DatasetError",
            message="safe failure",
            stage="gold",
            attempt=3,
        ),
        fingerprint="gold-fingerprint",
        attempt=3,
        details={"resource": "bounded"},
    )
    catalog = LocalCatalog(tmp_path / "catalog.duckdb")
    catalog.rebuild(LocalArtifactStore(tmp_path / "lake"), lifecycle_root=tmp_path / "control")
    row = catalog.logical_dump()["run_events"][0]
    assert row[9] == "executed"
    assert row[10] == 3
    assert "DatasetError" in row[11]
    assert row[12] == "gold-fingerprint"
    assert json.loads(row[13]) == {"resource": "bounded"}


def test_catalog_writer_lock_coordinates_incremental_and_rebuild(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    ref = _package(store, "locked")
    catalog = LocalCatalog(tmp_path / "catalog.duckdb")
    lock = tmp_path / ".catalog.duckdb.write-lock"
    lock.mkdir()
    with pytest.raises(RuntimeError, match="writer is already active"):
        catalog.index_package(store, ref)
    with pytest.raises(RuntimeError, match="writer is already active"):
        catalog.rebuild(store)


def test_undeclared_file_is_reported(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "lake")
    _package(store, "extra")
    package = store.package_path(ArtifactNamespace(layer=Layer.BRONZE, run_id="extra"))
    (package / "orphan.txt").write_text("orphan", encoding="utf-8")
    result = _service(tmp_path).verify_run("extra")
    assert result.status is VerificationStatus.INVALID
    assert result.issues[0].code == "undeclared_artifact"


def test_run_plan_cli_is_side_effect_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_root = tmp_path / "lake"
    code = main(
        [
            "--format",
            "json",
            "--artifact-root",
            str(artifact_root),
            "run",
            "plan",
            "--dataset",
            "titanic",
            "--method",
            "openfe",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == CLIExitCode.OK
    assert payload["status"] == "valid"
    assert payload["plans"][0]["state"] == "planned"
    assert not artifact_root.exists()
