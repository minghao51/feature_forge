"""Derived DuckDB catalog built only from verified local evidence."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pandas as pd

from feature_forge.contracts.artifacts import ArtifactNamespace, ManifestRef
from feature_forge.contracts.catalog import (
    CATALOG_SCHEMA_VERSION,
    CatalogIssue,
    CatalogReport,
    VerificationStatus,
)
from feature_forge.contracts.orchestration import RunEvent
from feature_forge.contracts.runs import RunManifest
from feature_forge.contracts.stages import Layer
from feature_forge.exceptions import FeatureForgeError
from feature_forge.storage.hashing import canonical_json_bytes, sha256_bytes
from feature_forge.storage.local import LocalArtifactStore

CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS catalog_metadata (
  singleton BOOLEAN PRIMARY KEY, schema_version INTEGER NOT NULL,
  manifest_schema_version VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS manifests (
  manifest_sha256 VARCHAR PRIMARY KEY, manifest_uri VARCHAR UNIQUE NOT NULL,
  run_id VARCHAR NOT NULL, case_fingerprint VARCHAR NOT NULL, layer VARCHAR NOT NULL,
  layer_fingerprint VARCHAR NOT NULL, state VARCHAR NOT NULL, dataset VARCHAR NOT NULL,
  method VARCHAR NOT NULL, model VARCHAR NOT NULL, seed BIGINT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL, completed_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS stages (
  manifest_sha256 VARCHAR NOT NULL, ordinal INTEGER NOT NULL, stage VARCHAR NOT NULL,
  state VARCHAR NOT NULL, started_at TIMESTAMPTZ NOT NULL, finished_at TIMESTAMPTZ,
  duration_ms DOUBLE, failure_class VARCHAR, error_type VARCHAR,
  PRIMARY KEY (manifest_sha256, ordinal)
);
CREATE TABLE IF NOT EXISTS artifacts (
  manifest_sha256 VARCHAR NOT NULL, relative_path VARCHAR NOT NULL, name VARCHAR NOT NULL,
  media_type VARCHAR NOT NULL, artifact_sha256 VARCHAR NOT NULL, size_bytes BIGINT NOT NULL,
  row_count BIGINT, column_count BIGINT, required BOOLEAN NOT NULL,
  PRIMARY KEY (manifest_sha256, relative_path)
);
CREATE TABLE IF NOT EXISTS validation_checks (
  manifest_sha256 VARCHAR NOT NULL, stage_ordinal INTEGER NOT NULL, check_ordinal INTEGER NOT NULL,
  check_id VARCHAR NOT NULL, severity VARCHAR NOT NULL, passed BOOLEAN NOT NULL,
  required BOOLEAN NOT NULL, message VARCHAR NOT NULL,
  PRIMARY KEY (manifest_sha256, stage_ordinal, check_ordinal)
);
CREATE TABLE IF NOT EXISTS lineage_edges (
  child_manifest_sha256 VARCHAR NOT NULL, ordinal INTEGER NOT NULL,
  parent_manifest_sha256 VARCHAR NOT NULL, parent_run_id VARCHAR NOT NULL,
  parent_layer VARCHAR NOT NULL,
  PRIMARY KEY (child_manifest_sha256, ordinal)
);
CREATE TABLE IF NOT EXISTS features (
  manifest_sha256 VARCHAR NOT NULL, candidate_id VARCHAR NOT NULL, name VARCHAR NOT NULL,
  batch_index INTEGER, code_path VARCHAR, specification_json VARCHAR NOT NULL,
  PRIMARY KEY (manifest_sha256, candidate_id)
);
CREATE TABLE IF NOT EXISTS feature_decisions (
  manifest_sha256 VARCHAR NOT NULL, decision_scope VARCHAR NOT NULL,
  candidate_id VARCHAR NOT NULL, state VARCHAR NOT NULL, reason_code VARCHAR NOT NULL,
  selected BOOLEAN, feature_name VARCHAR, evidence_json VARCHAR NOT NULL,
  PRIMARY KEY (manifest_sha256, decision_scope, candidate_id)
);
CREATE TABLE IF NOT EXISTS fold_metrics (
  manifest_sha256 VARCHAR NOT NULL, ordinal BIGINT NOT NULL, fold BIGINT,
  arm VARCHAR, candidate_id VARCHAR, score DOUBLE, record_json VARCHAR NOT NULL,
  PRIMARY KEY (manifest_sha256, ordinal)
);
CREATE TABLE IF NOT EXISTS run_events (
  event_id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, case_id VARCHAR NOT NULL,
  event_type VARCHAR NOT NULL, occurred_at TIMESTAMPTZ NOT NULL, state VARCHAR,
  stage VARCHAR, layer VARCHAR, manifest_sha256 VARCHAR, disposition VARCHAR,
  attempt INTEGER NOT NULL, failure_json VARCHAR, fingerprint VARCHAR,
  details_json VARCHAR NOT NULL
);
"""

_DEPENDENT_TABLES = (
    "stages",
    "artifacts",
    "validation_checks",
    "lineage_edges",
    "features",
    "feature_decisions",
    "fold_metrics",
)


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "DuckDB catalog support is optional; run `uv sync --extra pipeline`"
        ) from exc
    return duckdb


class LocalCatalog:
    """Manifest-centric derived index; artifact packages remain authoritative."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connect(self, *, read_only: bool = False) -> Any:
        if read_only and not self.path.is_file():
            raise FileNotFoundError(self.path)
        return _duckdb().connect(str(self.path), read_only=read_only)

    @staticmethod
    def _initialize(connection: Any) -> None:
        connection.execute(CATALOG_DDL)
        rows = connection.execute(
            "SELECT schema_version FROM catalog_metadata WHERE singleton = TRUE"
        ).fetchall()
        if not rows:
            connection.execute(
                "INSERT INTO catalog_metadata VALUES (TRUE, ?, '1')",
                [CATALOG_SCHEMA_VERSION],
            )
        elif rows[0][0] != CATALOG_SCHEMA_VERSION:
            raise ValueError(f"unsupported catalog schema version: {rows[0][0]}")

    def index_package(self, store: LocalArtifactStore, ref: ManifestRef) -> None:
        """Verify and normalize first, then replace one package in one transaction."""
        report = store.verify(ref)
        if not report.valid:
            raise ValueError("package failed verification: " + "; ".join(report.errors))
        manifest = store.load_manifest(ref)
        for upstream in manifest.upstream_manifests:
            upstream_report = store.verify(upstream)
            if not upstream_report.valid:
                raise ValueError(
                    "upstream package failed verification: " + "; ".join(upstream_report.errors)
                )
        _semantic_verify(store, ref)
        package = store.package_path(ArtifactNamespace(layer=ref.layer, run_id=ref.run_id))
        normalized = _normalize_package(package, manifest, ref)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._writer_lock():
            connection = self._connect()
            try:
                self._initialize(connection)
                connection.execute("BEGIN TRANSACTION")
                existing = connection.execute(
                    "SELECT manifest_sha256 FROM manifests WHERE layer=? AND run_id=?",
                    [ref.layer.value, ref.run_id],
                ).fetchall()
                for (digest,) in existing:
                    for table in _DEPENDENT_TABLES:
                        connection.execute(f"DELETE FROM {table} WHERE manifest_sha256=?", [digest])
                    connection.execute("DELETE FROM manifests WHERE manifest_sha256=?", [digest])
                _insert_normalized(connection, normalized)
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                connection.close()

    def rebuild(
        self, store: LocalArtifactStore, *, lifecycle_root: str | Path | None = None
    ) -> CatalogReport:
        """Build a deterministic sibling database and atomically replace the catalog."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.rebuild-{uuid.uuid4().hex}")
        candidate = LocalCatalog(temporary)
        issues: list[CatalogIssue] = []
        count = 0
        with self._writer_lock():
            try:
                connection = candidate._connect()
                candidate._initialize(connection)
                connection.close()
                for layer, package in store.iter_package_paths():
                    if package.name.startswith("."):
                        issues.append(
                            CatalogIssue(
                                code="incomplete_staging",
                                message="Incomplete package candidate",
                                path=str(package),
                            )
                        )
                        continue
                    try:
                        marker = (package / "_SUCCESS").read_text(encoding="utf-8").strip()
                        ref = ManifestRef(layer=layer, run_id=package.name, sha256=marker)
                        candidate.index_package(store, ref)
                        count += 1
                    except (FeatureForgeError, OSError, ValueError) as exc:
                        issues.append(
                            CatalogIssue(
                                code="invalid_package",
                                message=str(exc),
                                path=str(package),
                                run_id=package.name,
                            )
                        )
                if lifecycle_root is not None:
                    candidate.index_run_events(lifecycle_root)
                digest = candidate.logical_digest()
                database_fd = os.open(temporary, os.O_RDONLY)
                try:
                    os.fsync(database_fd)
                finally:
                    os.close(database_fd)
                os.replace(temporary, self.path)
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return CatalogReport(
                    status=VerificationStatus.INVALID if issues else VerificationStatus.VALID,
                    indexed_manifests=count,
                    issues=issues,
                    logical_digest=digest,
                )
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

    def index_run_events(self, lifecycle_root: str | Path) -> int:
        """Replace the derived lifecycle event index from validated journals."""
        from feature_forge.contracts.orchestration import RunEvent

        root = Path(lifecycle_root) / "runs"
        events: list[RunEvent] = []
        if root.is_dir():
            for path in sorted(root.glob("*/events.jsonl"), key=lambda item: item.as_posix()):
                events.extend(
                    RunEvent.model_validate_json(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._writer_lock():
            connection = self._connect()
            try:
                self._initialize(connection)
                connection.execute("BEGIN TRANSACTION")
                connection.execute("DELETE FROM run_events")
                for event in sorted(events, key=lambda item: (item.occurred_at, item.event_id)):
                    connection.execute(
                        "INSERT INTO run_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        _run_event_row(event),
                    )
                connection.execute("COMMIT")
                return len(events)
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                connection.close()

    def manifest_rows(self) -> list[tuple[str, str, str]]:
        """Return ``(digest, layer, run_id)`` rows without changing the database."""
        connection = self._connect(read_only=True)
        try:
            self._check_version(connection)
            return cast(
                list[tuple[str, str, str]],
                connection.execute(
                    "SELECT manifest_sha256, layer, run_id FROM manifests ORDER BY layer, run_id"
                ).fetchall(),
            )
        finally:
            connection.close()

    def logical_dump(self) -> dict[str, list[list[Any]]]:
        connection = self._connect(read_only=True)
        try:
            self._check_version(connection)
            return _dump_connection(connection)
        finally:
            connection.close()

    def logical_digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.logical_dump()))

    def expected_package_dump(
        self, store: LocalArtifactStore, ref: ManifestRef
    ) -> dict[str, list[list[Any]]]:
        """Build the package's exact logical rows in an in-memory catalog."""
        report = store.verify(ref)
        if not report.valid:
            raise ValueError("package failed verification: " + "; ".join(report.errors))
        manifest = store.load_manifest(ref)
        _semantic_verify(store, ref)
        package = store.package_path(ArtifactNamespace(layer=ref.layer, run_id=ref.run_id))
        connection = _duckdb().connect(":memory:")
        try:
            self._initialize(connection)
            _insert_normalized(connection, _normalize_package(package, manifest, ref))
            return _dump_connection(connection)
        finally:
            connection.close()

    def expected_run_event_rows(self, lifecycle_root: str | Path) -> list[list[Any]]:
        """Return deterministic typed journal rows without touching the catalog."""
        root = Path(lifecycle_root) / "runs"
        events: list[RunEvent] = []
        if root.is_dir():
            for path in sorted(root.glob("*/events.jsonl"), key=lambda item: item.as_posix()):
                events.extend(
                    RunEvent.model_validate_json(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                )
        return [
            [_json_value(value) for value in _run_event_row(event)]
            for event in sorted(events, key=lambda item: (item.occurred_at, item.event_id))
        ]

    @contextmanager
    def _writer_lock(self) -> Iterator[None]:
        lock = self.path.with_name(f".{self.path.name}.write-lock")
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(f"catalog writer is already active: {self.path}") from exc
        try:
            yield
        finally:
            lock.rmdir()

    @staticmethod
    def _check_version(connection: Any) -> None:
        rows = connection.execute(
            "SELECT schema_version FROM catalog_metadata WHERE singleton = TRUE"
        ).fetchall()
        if not rows or rows[0][0] != CATALOG_SCHEMA_VERSION:
            version = None if not rows else rows[0][0]
            raise ValueError(f"unsupported catalog schema version: {version}")


def _normalize_package(package: Path, manifest: RunManifest, ref: ManifestRef) -> dict[str, Any]:
    manifest_uri = f"{manifest.layer.value}:{manifest.run_id}:manifest.json"
    data: dict[str, Any] = {"manifest": manifest, "ref": ref, "uri": manifest_uri}
    if manifest.layer is Layer.GOLD:
        data["features"] = _read_json(package / "candidates.json", [])
        data["gold_decisions"] = _read_json(package / "decisions.json", [])
    if manifest.layer is Layer.PLATINUM:
        data["platinum_decisions"] = _read_json(package / "selection_decisions.json", [])
        metrics_path = package / "fold_metrics.parquet"
        data["fold_metrics"] = (
            pd.read_parquet(metrics_path).to_dict(orient="records")
            if metrics_path.is_file()
            else []
        )
    return data


def _insert_normalized(connection: Any, data: dict[str, Any]) -> None:
    manifest: RunManifest = data["manifest"]
    ref: ManifestRef = data["ref"]
    digest = ref.sha256
    connection.execute(
        "INSERT INTO manifests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            digest,
            data["uri"],
            manifest.run_id,
            manifest.case_fingerprint,
            manifest.layer.value,
            manifest.layer_fingerprint,
            manifest.state.value,
            manifest.request.dataset,
            manifest.request.method,
            manifest.request.model,
            manifest.request.seed,
            manifest.created_at,
            manifest.completed_at,
        ],
    )
    for index, stage in enumerate(manifest.stages):
        connection.execute(
            "INSERT INTO stages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                digest,
                index,
                stage.stage,
                stage.state.value,
                stage.started_at,
                stage.finished_at,
                stage.duration_ms,
                stage.failure_class.value if stage.failure_class else None,
                stage.error_type,
            ],
        )
        for check_index, check in enumerate(stage.checks):
            connection.execute(
                "INSERT INTO validation_checks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    digest,
                    index,
                    check_index,
                    check.check_id,
                    check.severity.value,
                    check.passed,
                    check.required,
                    check.message,
                ],
            )
    for artifact in manifest.artifacts:
        connection.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                digest,
                artifact.relative_path,
                artifact.name,
                artifact.media_type,
                artifact.sha256,
                artifact.size_bytes,
                artifact.row_count,
                artifact.column_count,
                artifact.required,
            ],
        )
    for index, parent in enumerate(manifest.upstream_manifests):
        connection.execute(
            "INSERT INTO lineage_edges VALUES (?, ?, ?, ?, ?)",
            [digest, index, parent.sha256, parent.run_id, parent.layer.value],
        )
    for feature in data.get("features", []):
        connection.execute(
            "INSERT INTO features VALUES (?, ?, ?, ?, ?, ?)",
            [
                digest,
                feature["candidate_id"],
                feature["name"],
                feature.get("batch_index"),
                feature.get("code_path"),
                json.dumps(feature.get("specification", {}), sort_keys=True),
            ],
        )
    for scope, key in (("gold", "gold_decisions"), ("platinum", "platinum_decisions")):
        for decision in data.get(key, []):
            connection.execute(
                "INSERT INTO feature_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    digest,
                    scope,
                    decision["candidate_id"],
                    decision.get("state", "selected" if decision.get("selected") else "rejected"),
                    decision["reason_code"],
                    decision.get("selected"),
                    decision.get("feature_name"),
                    json.dumps(decision, sort_keys=True),
                ],
            )
    for index, record in enumerate(data.get("fold_metrics", [])):
        connection.execute(
            "INSERT INTO fold_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                digest,
                index,
                record.get("fold"),
                record.get("arm"),
                record.get("candidate_id"),
                record.get("score"),
                json.dumps(record, sort_keys=True, default=str),
            ],
        )


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _dump_connection(connection: Any) -> dict[str, list[list[Any]]]:
    result: dict[str, list[list[Any]]] = {}
    for table in ("manifests", *_DEPENDENT_TABLES, "run_events"):
        rows = connection.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
        result[table] = [[_json_value(value) for value in row] for row in rows]
    return result


def _run_event_row(event: RunEvent) -> list[Any]:
    return [
        event.event_id,
        event.run_id,
        event.case_id,
        event.event_type,
        event.occurred_at,
        event.state.value if event.state else None,
        event.stage,
        event.layer.value if event.layer else None,
        event.manifest_ref.sha256 if event.manifest_ref else None,
        event.disposition.value if event.disposition else None,
        event.attempt,
        json.dumps(event.failure.model_dump(mode="json"), sort_keys=True)
        if event.failure
        else None,
        event.fingerprint,
        json.dumps(event.details, sort_keys=True),
    ]


def _semantic_verify(store: LocalArtifactStore, ref: ManifestRef) -> None:
    """Authorize reuse/indexing through the same layer-specific semantic loaders."""
    if ref.layer is Layer.BRONZE:
        store.load_manifest(ref)
        return
    if ref.layer is Layer.SILVER:
        from feature_forge.dataflows.silver import load_silver_package

        load_silver_package(store, ref)
        return
    if ref.layer is Layer.GOLD:
        from feature_forge.dataflows.gold import load_gold_package
        from feature_forge.dataflows.silver import load_silver_package

        gold = load_gold_package(store, ref)
        silver = load_silver_package(store, gold.request.silver_manifest)
        if (
            gold.manifest.upstream_manifests != [gold.request.silver_manifest]
            or silver.manifest.layer_fingerprint != gold.request.silver_fingerprint
        ):
            raise FeatureForgeError("Gold package has invalid authoritative Silver lineage")
        return
    from feature_forge.dataflows.platinum import load_platinum_package

    load_platinum_package(store, ref)
