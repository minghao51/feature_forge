"""Read-only verification and catalog reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from feature_forge.contracts.artifacts import ArtifactNamespace, ArtifactRef, ManifestRef
from feature_forge.contracts.catalog import CatalogIssue, VerificationResult, VerificationStatus
from feature_forge.contracts.orchestration import RunEvent
from feature_forge.contracts.stages import Layer
from feature_forge.exceptions import FeatureForgeError
from feature_forge.storage.catalog import LocalCatalog
from feature_forge.storage.local import LocalArtifactStore


class VerificationService:
    """Verify authoritative evidence without constructing caches/providers or writers."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        catalog_path: str | Path,
        lifecycle_root: str | Path,
        stale_run_seconds: float = 3600.0,
    ) -> None:
        self.store = LocalArtifactStore(artifact_root)
        self.catalog = LocalCatalog(catalog_path)
        self.lifecycle_root = Path(lifecycle_root)
        self.stale_run_seconds = stale_run_seconds

    def verify_artifact(self, uri: str) -> VerificationResult:
        try:
            layer_text, run_id, relative_path = uri.split(":", maxsplit=2)
            layer = Layer(layer_text)
        except ValueError:
            return _invalid(uri, "invalid_uri", "Expected <layer>:<run-id>:<relative-path>")
        try:
            ref = ArtifactRef(layer=layer, run_id=run_id, relative_path=relative_path)
            path = self.store.resolve(ref)
        except FileNotFoundError as exc:
            return VerificationResult(
                subject=uri,
                status=VerificationStatus.MISSING,
                issues=[CatalogIssue(code="missing", message=str(exc), run_id=run_id)],
            )
        except ValueError as exc:
            return _invalid(uri, "invalid_artifact", str(exc), run_id=run_id)
        return VerificationResult(
            subject=uri,
            status=VerificationStatus.VALID,
            details={"resolved_path": str(path), "declared": True},
        )

    def verify_run(self, run_id: str) -> VerificationResult:
        issues: list[CatalogIssue] = []
        found = False
        for layer in Layer:
            namespace = ArtifactNamespace(layer=layer, run_id=run_id)
            if not self.store.package_path(namespace).exists():
                continue
            found = True
            result = self._verify_namespace(namespace, subject=run_id)
            issues.extend(result.issues)
        events = self._events(run_id)
        found = found or bool(events)
        if _is_stale(events, threshold=self.stale_run_seconds):
            issues.append(
                CatalogIssue(
                    code="stale_run",
                    message="Run has no case_terminal after case_running",
                    run_id=run_id,
                )
            )
        if not found:
            return VerificationResult(
                subject=run_id,
                status=VerificationStatus.MISSING,
                issues=[
                    CatalogIssue(code="missing_run", message="Run has no evidence", run_id=run_id)
                ],
            )
        return VerificationResult(
            subject=run_id,
            status=VerificationStatus.INVALID if issues else VerificationStatus.VALID,
            issues=issues,
            details={"event_count": len(events)},
        )

    def verify_catalog(self) -> VerificationResult:
        try:
            rows = self.catalog.manifest_rows()
        except FileNotFoundError:
            return VerificationResult(
                subject=str(self.catalog.path),
                status=VerificationStatus.MISSING,
                issues=[
                    CatalogIssue(
                        code="missing_catalog",
                        message="Catalog does not exist",
                        path=str(self.catalog.path),
                    )
                ],
            )
        except ValueError as exc:
            return VerificationResult(
                subject=str(self.catalog.path),
                status=VerificationStatus.INCOMPATIBLE,
                issues=[
                    CatalogIssue(
                        code="incompatible_catalog", message=str(exc), path=str(self.catalog.path)
                    )
                ],
            )
        issues: list[CatalogIssue] = []
        live_dump = self.catalog.logical_dump()
        expected_dump: dict[str, list[list[object]]] = {table: [] for table in live_dump}
        indexed = {(Layer(layer), run_id): digest for digest, layer, run_id in rows}
        disk: dict[tuple[Layer, str], str] = {}
        for layer, package in self.store.iter_package_paths():
            if package.name.startswith("."):
                issues.append(
                    CatalogIssue(
                        code="incomplete_staging",
                        message="Incomplete package candidate",
                        path=str(package),
                    )
                )
                continue
            namespace = ArtifactNamespace(layer=layer, run_id=package.name)
            result = self._verify_namespace(namespace, subject=package.name)
            issues.extend(result.issues)
            if result.status is VerificationStatus.VALID:
                marker = (package / "_SUCCESS").read_text(encoding="utf-8").strip()
                disk[(layer, package.name)] = marker
                ref = ManifestRef(layer=layer, run_id=package.name, sha256=marker)
                package_dump = self.catalog.expected_package_dump(self.store, ref)
                for table, values in package_dump.items():
                    if table != "run_events":
                        expected_dump[table].extend(values)
        for key, digest in disk.items():
            if indexed.get(key) != digest:
                issues.append(
                    CatalogIssue(
                        code="unindexed_package",
                        message="Verified package is absent or stale in catalog",
                        run_id=key[1],
                    )
                )
        for key in indexed.keys() - disk.keys():
            issues.append(
                CatalogIssue(
                    code="stale_catalog_row",
                    message="Catalog row has no verified package",
                    run_id=key[1],
                )
            )
        runs_root = self.lifecycle_root / "runs"
        journal_invalid = False
        if runs_root.is_dir():
            for run_dir in sorted(runs_root.iterdir(), key=lambda path: path.name):
                if not run_dir.is_dir() or run_dir.is_symlink():
                    continue
                try:
                    events = self._events(run_dir.name)
                except ValueError as exc:
                    issues.append(
                        CatalogIssue(
                            code="invalid_journal",
                            message=str(exc),
                            path=str(run_dir),
                            run_id=run_dir.name,
                        )
                    )
                    journal_invalid = True
                    continue
                if _is_stale(events, threshold=self.stale_run_seconds):
                    issues.append(
                        CatalogIssue(
                            code="stale_run",
                            message="Run has no case_terminal after case_running",
                            run_id=run_dir.name,
                        )
                    )
        if journal_invalid:
            return VerificationResult(
                subject=str(self.catalog.path),
                status=VerificationStatus.INVALID,
                issues=issues,
                details={"indexed_manifests": len(rows), "disk_manifests": len(disk)},
            )
        expected_dump["run_events"] = self.catalog.expected_run_event_rows(self.lifecycle_root)
        for table in live_dump:
            live_rows = sorted(live_dump[table], key=_row_key)
            expected_rows = sorted(expected_dump[table], key=_row_key)
            if live_rows != expected_rows:
                issues.append(
                    CatalogIssue(
                        code="catalog_row_mismatch",
                        message=f"Derived table differs from authoritative evidence: {table}",
                        path=str(self.catalog.path),
                    )
                )
        return VerificationResult(
            subject=str(self.catalog.path),
            status=VerificationStatus.INVALID if issues else VerificationStatus.VALID,
            issues=issues,
            details={"indexed_manifests": len(rows), "disk_manifests": len(disk)},
        )

    def _verify_namespace(
        self, namespace: ArtifactNamespace, *, subject: str
    ) -> VerificationResult:
        package = self.store.package_path(namespace)
        if not package.exists():
            return VerificationResult(
                subject=subject,
                status=VerificationStatus.MISSING,
                issues=[
                    CatalogIssue(
                        code="missing_package",
                        message="Package does not exist",
                        path=str(package),
                        run_id=namespace.run_id,
                    )
                ],
            )
        try:
            digest = (package / "_SUCCESS").read_text(encoding="utf-8").strip()
            ref = ManifestRef(layer=namespace.layer, run_id=namespace.run_id, sha256=digest)
            report = self.store.verify(ref)
        except (OSError, ValueError) as exc:
            return _invalid(
                subject, "invalid_package", str(exc), path=package, run_id=namespace.run_id
            )
        issues = [
            CatalogIssue(
                code="invalid_package", message=message, path=str(package), run_id=namespace.run_id
            )
            for message in report.errors
        ]
        if not issues:
            manifest = self.store.load_manifest(ref)
            try:
                from feature_forge.storage.catalog import _semantic_verify

                _semantic_verify(self.store, ref)
            except (FeatureForgeError, OSError, ValueError) as exc:
                issues.append(
                    CatalogIssue(
                        code="semantic_verification_failed",
                        message=str(exc),
                        path=str(package),
                        run_id=namespace.run_id,
                    )
                )
            declared = {item.relative_path for item in manifest.artifacts} | {
                "manifest.json",
                "_SUCCESS",
            }
            actual = {
                path.relative_to(package).as_posix()
                for path in package.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            for relative_path in sorted(actual - declared):
                issues.append(
                    CatalogIssue(
                        code="undeclared_artifact",
                        message="File is not declared by the manifest",
                        path=str(package / relative_path),
                        run_id=namespace.run_id,
                    )
                )
        return VerificationResult(
            subject=subject,
            status=VerificationStatus.INVALID if issues else VerificationStatus.VALID,
            issues=issues,
            details={"checked_artifacts": report.checked_artifacts, "warnings": report.warnings},
        )

    def _events(self, run_id: str) -> list[RunEvent]:
        path = self.lifecycle_root / "runs" / run_id / "events.jsonl"
        if not path.is_file():
            return []
        return [
            RunEvent.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]


def _invalid(
    subject: str, code: str, message: str, *, path: Path | None = None, run_id: str | None = None
) -> VerificationResult:
    return VerificationResult(
        subject=subject,
        status=VerificationStatus.INVALID,
        issues=[
            CatalogIssue(
                code=code, message=message, path=str(path) if path else None, run_id=run_id
            )
        ],
    )


def _is_stale(events: list[RunEvent], *, threshold: float) -> bool:
    running = [event for event in events if event.event_type == "case_running"]
    if not running:
        return False
    latest_running = running[-1]
    if any(
        event.event_type == "case_terminal" and event.occurred_at >= latest_running.occurred_at
        for event in events
    ):
        return False
    return (datetime.now(UTC) - latest_running.occurred_at).total_seconds() > threshold


def _row_key(row: list[object]) -> str:
    import json

    return json.dumps(row, sort_keys=True, default=str)
