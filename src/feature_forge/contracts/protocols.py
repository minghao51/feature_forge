"""Structural interfaces for artifact and run repositories."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from feature_forge.contracts.artifacts import (
    ArtifactDescriptor,
    ArtifactNamespace,
    ArtifactRef,
    ManifestRef,
)
from feature_forge.contracts.orchestration import RunEvent
from feature_forge.contracts.runs import RunManifest
from feature_forge.contracts.stages import VerificationReport


class StagingArea(Protocol):
    """Minimum write/manifest surface required by an artifact store."""

    path: Path
    namespace: ArtifactNamespace
    manifest: RunManifest | None

    @property
    def artifacts(self) -> list[ArtifactDescriptor]: ...

    def set_manifest(self, manifest: RunManifest) -> None: ...


class ArtifactStore(Protocol):
    """Protocol for durable, verifiable artifact packages."""

    def begin(self, namespace: ArtifactNamespace) -> StagingArea: ...

    def commit(self, staging: StagingArea) -> ManifestRef: ...

    def verify(self, ref: ManifestRef) -> VerificationReport: ...

    def resolve(self, ref: ArtifactRef) -> Path: ...


class RunRepository(Protocol):
    """Protocol for durable run lifecycle events."""

    def load_events(self, run_id: str) -> list[RunEvent]: ...


class CaseScheduler(Protocol):
    """Protocol for an outer case scheduler."""

    def run(self, cases: Sequence[object], *, fail_fast: bool = False) -> list[object]: ...
