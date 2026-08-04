"""Shared base for per-layer materialization result contracts."""

from __future__ import annotations

from pydantic import Field

from feature_forge.contracts.artifacts import ManifestRef
from feature_forge.contracts.base import ContractModel
from feature_forge.contracts.runs import RunManifest


class Materialization(ContractModel):
    """Common shape of a per-layer materialization result.

    Bronze/Silver/Platinum use this directly. ``GoldMaterialization``
    (in ``contracts.gold``) subclasses it to add the required ``counts`` field.
    ``extra="forbid"`` is inherited from ``ContractModel``.
    """

    manifest: RunManifest
    manifest_ref: ManifestRef | None = None
    persisted: bool
    artifact_paths: list[str] = Field(default_factory=list)
