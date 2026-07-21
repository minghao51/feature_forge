"""Local filesystem implementation of the durable artifact store."""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import pandas as pd

from feature_forge.contracts.artifacts import (
    ArtifactDescriptor,
    ArtifactNamespace,
    ArtifactRef,
    ManifestRef,
    validate_relative_path,
)
from feature_forge.contracts.runs import RunManifest
from feature_forge.contracts.stages import Layer, RunState, VerificationReport
from feature_forge.storage.atomic import (
    atomic_publish_directory,
    atomic_write_json,
    create_staging_directory,
    write_success_marker,
)
from feature_forge.storage.hashing import (
    canonical_json_bytes,
    fingerprint,
    sha256_bytes,
    sha256_file,
)

_SCHEMA_VERSION: Literal["1"] = "1"
_SHA256_LENGTH = 64
_LAYER_DIRECTORY = {
    Layer.BRONZE: "01_bronze",
    Layer.SILVER: "02_silver",
    Layer.GOLD: "03_gold",
    Layer.PLATINUM: "04_platinum",
}


def _relative_path(value: str, *, allow_reserved: bool = False) -> Path:
    try:
        validate_relative_path(value)
    except ValueError as exc:
        raise ValueError("artifact paths must be normalized and relative") from exc
    if not allow_reserved and value in {"manifest.json", "_SUCCESS"}:
        raise ValueError("reserved package paths cannot be written as artifacts")
    return Path(*PurePosixPath(value).parts)


def _success_manifest_sha256(package: Path) -> str:
    """Read and validate the manifest digest anchored by ``_SUCCESS``."""
    marker = package / "_SUCCESS"
    if not marker.is_file():
        raise FileNotFoundError("completion marker is missing")
    digest = marker.read_text(encoding="utf-8").strip()
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("completion marker does not contain a valid manifest SHA-256")
    return digest


class LocalStagingArea:
    """Write-once package staging area owned by a ``LocalArtifactStore``."""

    def __init__(
        self,
        path: Path,
        namespace: ArtifactNamespace,
        schema_version: Literal["1"],
    ) -> None:
        self.path = path
        self.namespace = namespace
        self.schema_version = schema_version
        self.manifest: RunManifest | None = None
        self._artifacts: dict[str, ArtifactDescriptor] = {}

    @property
    def artifacts(self) -> list[ArtifactDescriptor]:
        """Return descriptors registered by writes in path order."""
        return [self._artifacts[path] for path in sorted(self._artifacts)]

    def set_manifest(self, manifest: RunManifest) -> None:
        """Set the manifest metadata that will be written during commit."""
        if manifest.run_id != self.namespace.run_id:
            raise ValueError("manifest run_id does not match staging namespace")
        if manifest.layer is not self.namespace.layer:
            raise ValueError("manifest layer does not match staging namespace")
        self.manifest = manifest

    def write_bytes(
        self,
        relative_path: str,
        content: bytes,
        *,
        name: str | None = None,
        media_type: str = "application/octet-stream",
        required: bool = True,
    ) -> ArtifactDescriptor:
        """Write bytes and register their immutable descriptor."""
        path = _relative_path(relative_path)
        destination = self.path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return self._register(
            relative_path=relative_path,
            name=name or path.name,
            media_type=media_type,
            size_bytes=len(content),
            sha256=sha256_bytes(content),
            required=required,
        )

    def write_text(
        self,
        relative_path: str,
        content: str,
        *,
        name: str | None = None,
        media_type: str = "text/plain",
        required: bool = True,
    ) -> ArtifactDescriptor:
        """Write UTF-8 text and register its descriptor."""
        return self.write_bytes(
            relative_path,
            content.encode("utf-8"),
            name=name,
            media_type=media_type,
            required=required,
        )

    def write_json(
        self,
        relative_path: str,
        value: Any,
        *,
        name: str | None = None,
        required: bool = True,
    ) -> ArtifactDescriptor:
        """Write stable JSON and register its descriptor."""
        return self.write_bytes(
            relative_path,
            canonical_json_bytes(value) + b"\n",
            name=name,
            media_type="application/json",
            required=required,
        )

    def write_dataframe(
        self,
        relative_path: str,
        dataframe: pd.DataFrame,
        *,
        name: str | None = None,
        required: bool = True,
    ) -> ArtifactDescriptor:
        """Write a DataFrame as Parquet and register its descriptor."""
        path = _relative_path(relative_path)
        destination = self.path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(destination)
        return self._register(
            relative_path=relative_path,
            name=name or path.name,
            media_type="application/vnd.apache.parquet",
            size_bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            schema_fingerprint=fingerprint(
                {
                    "columns": list(dataframe.columns),
                    "dtypes": [str(dtype) for dtype in dataframe.dtypes],
                }
            ),
            required=required,
        )

    def _register(self, **kwargs: Any) -> ArtifactDescriptor:
        descriptor = ArtifactDescriptor(
            schema_version=self.schema_version,
            layer=self.namespace.layer,
            **kwargs,
        )
        self._artifacts[descriptor.relative_path] = descriptor
        return descriptor


class LocalArtifactStore:
    """Run-scoped local artifact store with atomic package publication."""

    def __init__(
        self,
        root: str | Path,
        *,
        schema_version: Literal["1"] = _SCHEMA_VERSION,
    ) -> None:
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema version: {schema_version}")
        self.root = Path(root)
        self.schema_version = schema_version

    def begin(self, namespace: ArtifactNamespace) -> LocalStagingArea:
        """Create a unique staging directory for a layer namespace."""
        parent = self.root / _LAYER_DIRECTORY[namespace.layer] / "runs"
        path = create_staging_directory(parent, prefix=f".{namespace.run_id}.tmp-")
        atomic_write_json(
            path / ".staging-owner.json",
            {
                "schema_version": "1",
                "token": uuid.uuid4().hex,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_at_epoch": time.time(),
            },
        )
        return LocalStagingArea(path, namespace, self.schema_version)

    def commit(self, staging: LocalStagingArea) -> ManifestRef:
        """Validate and atomically publish a staged package."""
        manifest = staging.manifest
        if manifest is None:
            raise ValueError("staging area has no manifest")
        if manifest.state is not RunState.SUCCEEDED:
            raise ValueError("only succeeded manifests can be committed")

        descriptors = self._merge_descriptors(manifest, staging.artifacts, staging.namespace.layer)
        for descriptor in descriptors:
            path = staging.path / _relative_path(descriptor.relative_path)
            if not path.is_file():
                if descriptor.required:
                    raise FileNotFoundError(
                        f"required artifact is missing: {descriptor.relative_path}"
                    )
                continue
            actual_size = path.stat().st_size
            actual_hash = sha256_file(path)
            if actual_size != descriptor.size_bytes or actual_hash != descriptor.sha256:
                raise ValueError(
                    f"artifact descriptor does not match file: {descriptor.relative_path}"
                )

        committed_manifest = manifest.model_copy(update={"artifacts": descriptors})
        staging.manifest = committed_manifest
        (staging.path / ".staging-owner.json").unlink(missing_ok=True)
        atomic_write_json(
            staging.path / "manifest.json", committed_manifest.model_dump(mode="json")
        )
        manifest_sha256 = sha256_file(staging.path / "manifest.json")
        write_success_marker(staging.path, manifest_sha256)
        destination = self._namespace_path(staging.namespace)
        atomic_publish_directory(staging.path, destination)
        return ManifestRef(
            layer=staging.namespace.layer,
            run_id=staging.namespace.run_id,
            sha256=manifest_sha256,
        )

    def find_reusable(
        self,
        fingerprint_value: str,
        layer: Layer,
        *,
        upstream: ManifestRef | None = None,
        preferred_run_id: str | None = None,
        allow_cross_run: bool = True,
    ) -> ManifestRef | None:
        """Find a verified package by fingerprint and exact upstream lineage."""
        runs = self.root / _LAYER_DIRECTORY[layer] / "runs"
        if not runs.is_dir():
            return None
        candidates = [path for path in runs.iterdir() if path.is_dir() and not path.is_symlink()]
        candidates.sort(key=lambda path: (path.name != preferred_run_id, path.name))
        for package in candidates:
            if not allow_cross_run and package.name != preferred_run_id:
                continue
            try:
                namespace = ArtifactNamespace(layer=layer, run_id=package.name)
                ref = self.get_manifest_ref(namespace)
                if ref is None:
                    continue
                manifest = self.load_manifest(ref)
            except (OSError, ValueError) as exc:
                if package.name == preferred_run_id:
                    raise ValueError(
                        f"preferred {layer.value} package is invalid: {package}"
                    ) from exc
                continue
            if manifest.layer_fingerprint != fingerprint_value:
                continue
            if upstream is not None and upstream not in manifest.upstream_manifests:
                continue
            if upstream is None and layer is not Layer.BRONZE:
                continue
            return ref
        return None

    def recover_abandoned(
        self,
        *,
        retain_staging_seconds: float,
        stale_lock_seconds: float,
    ) -> dict[str, list[str]]:
        """Remove only provably abandoned local staging areas and commit locks."""
        now = time.time()
        removed_staging: list[str] = []
        removed_locks: list[str] = []
        retained: list[str] = []
        hostname = socket.gethostname()
        for directory in _LAYER_DIRECTORY.values():
            runs = self.root / directory / "runs"
            if not runs.is_dir():
                continue
            for path in runs.iterdir():
                if path.is_symlink():
                    retained.append(str(path))
                    continue
                if path.is_dir() and ".tmp-" in path.name:
                    owner = _owner_payload(path / ".staging-owner.json")
                    age = now - path.stat().st_mtime
                    if age >= retain_staging_seconds and _owner_is_dead(owner, hostname):
                        shutil.rmtree(path)
                        removed_staging.append(str(path))
                    else:
                        retained.append(str(path))
                elif path.is_dir() and path.name.endswith(".commit-lock"):
                    owner = _owner_payload(path / "owner.json")
                    age = now - path.stat().st_mtime
                    if age >= stale_lock_seconds and _owner_is_dead(owner, hostname):
                        shutil.rmtree(path)
                        removed_locks.append(str(path))
                    else:
                        retained.append(str(path))
        return {
            "removed_staging": removed_staging,
            "removed_locks": removed_locks,
            "retained": retained,
        }

    def verify(self, ref: ManifestRef) -> VerificationReport:
        """Verify completion marker, manifest, and every required artifact hash."""
        package = self._namespace_path(ArtifactNamespace(layer=ref.layer, run_id=ref.run_id))
        manifest_path = package / _relative_path(ref.relative_path, allow_reserved=True)
        errors: list[str] = []
        warnings: list[str] = []
        checked: list[str] = []
        anchored_manifest_sha256: str | None = None
        try:
            anchored_manifest_sha256 = _success_manifest_sha256(package)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        if not manifest_path.is_file():
            errors.append("manifest is missing")
            return VerificationReport(
                valid=False,
                run_id=ref.run_id,
                layer=ref.layer,
                manifest_path=str(manifest_path),
                checked_artifacts=checked,
                errors=errors,
                warnings=warnings,
            )

        try:
            manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"manifest is invalid: {exc}")
            return VerificationReport(
                valid=False,
                run_id=ref.run_id,
                layer=ref.layer,
                manifest_path=str(manifest_path),
                checked_artifacts=checked,
                errors=errors,
                warnings=warnings,
            )

        if manifest.run_id != ref.run_id:
            errors.append("manifest run_id does not match reference")
        if manifest.layer is not ref.layer:
            errors.append("manifest layer does not match reference")
        if manifest.package_kind != ref.layer.value:
            errors.append("manifest package_kind does not match reference")
        if manifest.state is not RunState.SUCCEEDED:
            errors.append("manifest is not succeeded")
        if anchored_manifest_sha256 is not None and anchored_manifest_sha256 != ref.sha256:
            errors.append("manifest reference hash does not match completion marker")
        if sha256_file(manifest_path) != ref.sha256:
            errors.append("manifest hash does not match reference")

        artifact_paths = [descriptor.relative_path for descriptor in manifest.artifacts]
        if len(set(artifact_paths)) != len(artifact_paths):
            errors.append("manifest contains duplicate artifact paths")

        for descriptor in manifest.artifacts:
            if descriptor.layer is not ref.layer:
                errors.append(f"artifact layer mismatch: {descriptor.relative_path}")
            artifact_path = package / _relative_path(descriptor.relative_path)
            if not artifact_path.is_file():
                message = f"artifact is missing: {descriptor.relative_path}"
                (errors if descriptor.required else warnings).append(message)
                continue
            checked.append(descriptor.relative_path)
            issue_target = errors if descriptor.required else warnings
            if artifact_path.stat().st_size != descriptor.size_bytes:
                issue_target.append(f"artifact size mismatch: {descriptor.relative_path}")
            if sha256_file(artifact_path) != descriptor.sha256:
                issue_target.append(f"artifact hash mismatch: {descriptor.relative_path}")

        return VerificationReport(
            valid=not errors,
            run_id=ref.run_id,
            layer=ref.layer,
            manifest_path=str(manifest_path),
            checked_artifacts=checked,
            errors=errors,
            warnings=warnings,
        )

    def resolve(self, ref: ArtifactRef) -> Path:
        """Resolve a declared artifact only after checking its size and hash."""
        package = self._namespace_path(ArtifactNamespace(layer=ref.layer, run_id=ref.run_id))
        if not (package / "_SUCCESS").is_file() or not (package / "manifest.json").is_file():
            raise FileNotFoundError(f"artifact package is incomplete: {package}")
        if ref.relative_path in {"manifest.json", "_SUCCESS"}:
            raise ValueError("package metadata is not an artifact")
        manifest_path = package / "manifest.json"
        anchored_manifest_sha256 = _success_manifest_sha256(package)
        report = self.verify(
            ManifestRef(
                layer=ref.layer,
                run_id=ref.run_id,
                sha256=anchored_manifest_sha256,
            )
        )
        if not report.valid:
            raise ValueError(f"artifact package failed verification: {'; '.join(report.errors)}")
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.layer is not ref.layer or manifest.run_id != ref.run_id:
            raise ValueError("artifact manifest does not match reference namespace")
        descriptor = next(
            (
                item
                for item in manifest.artifacts
                if item.relative_path == ref.relative_path and item.layer is ref.layer
            ),
            None,
        )
        if descriptor is None:
            raise ValueError(f"artifact is not declared in manifest: {ref.relative_path}")
        path = package / _relative_path(ref.relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != descriptor.size_bytes or sha256_file(path) != descriptor.sha256:
            raise ValueError(f"artifact failed integrity verification: {ref.relative_path}")
        return path

    def get_manifest_ref(self, namespace: ArtifactNamespace) -> ManifestRef | None:
        """Return a verified reference for an existing committed namespace."""
        package = self._namespace_path(namespace)
        manifest_path = package / "manifest.json"
        if not package.exists():
            return None
        if not manifest_path.is_file() or not (package / "_SUCCESS").is_file():
            raise ValueError(f"artifact package is incomplete: {package}")
        try:
            anchored_manifest_sha256 = _success_manifest_sha256(package)
        except (OSError, ValueError) as exc:
            raise ValueError(f"artifact package is incomplete: {package}: {exc}") from exc
        ref = ManifestRef(
            layer=namespace.layer,
            run_id=namespace.run_id,
            sha256=anchored_manifest_sha256,
        )
        report = self.verify(ref)
        if not report.valid:
            raise ValueError(f"artifact package failed verification: {'; '.join(report.errors)}")
        return ref

    def load_manifest(self, ref: ManifestRef) -> RunManifest:
        """Load a manifest only after its complete package verifies."""
        report = self.verify(ref)
        if not report.valid:
            raise ValueError(f"artifact package failed verification: {'; '.join(report.errors)}")
        path = (
            self._namespace_path(ArtifactNamespace(layer=ref.layer, run_id=ref.run_id))
            / ref.relative_path
        )
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def iter_package_paths(self) -> list[tuple[Layer, Path]]:
        """Return deterministic package candidates without mutating the artifact root."""
        candidates: list[tuple[Layer, Path]] = []
        for layer, directory in _LAYER_DIRECTORY.items():
            runs = self.root / directory / "runs"
            if not runs.is_dir():
                continue
            candidates.extend(
                (layer, path) for path in runs.iterdir() if path.is_dir() and not path.is_symlink()
            )
        return sorted(candidates, key=lambda item: (list(Layer).index(item[0]), item[1].name))

    def package_path(self, namespace: ArtifactNamespace) -> Path:
        """Return the canonical local path for a namespace without creating it."""
        return self._namespace_path(namespace)

    def _namespace_path(self, namespace: ArtifactNamespace) -> Path:
        return self.root / _LAYER_DIRECTORY[namespace.layer] / "runs" / namespace.run_id

    @staticmethod
    def _merge_descriptors(
        manifest: RunManifest,
        staged: list[ArtifactDescriptor],
        layer: Layer,
    ) -> list[ArtifactDescriptor]:
        manifest_paths = [descriptor.relative_path for descriptor in manifest.artifacts]
        if len(set(manifest_paths)) != len(manifest_paths):
            raise ValueError("manifest contains duplicate artifact paths")
        for descriptor in [*manifest.artifacts, *staged]:
            if descriptor.layer is not layer:
                raise ValueError(
                    f"artifact layer does not match namespace: {descriptor.relative_path}"
                )
        by_path = {descriptor.relative_path: descriptor for descriptor in manifest.artifacts}
        for descriptor in staged:
            declared = by_path.get(descriptor.relative_path)
            if declared is not None and declared != descriptor:
                raise ValueError(
                    f"manifest descriptor conflicts with staged artifact: {descriptor.relative_path}"
                )
            by_path[descriptor.relative_path] = descriptor
        return [by_path[path] for path in sorted(by_path)]


def _owner_payload(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _owner_is_dead(owner: dict[str, Any] | None, hostname: str) -> bool:
    """Return true only for missing legacy metadata or a dead same-host PID."""
    if owner is None:
        return True
    if owner.get("hostname") != hostname:
        return False
    pid = owner.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return False
