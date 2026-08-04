"""Run identity and environment contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from feature_forge.contracts.artifacts import ArtifactDescriptor, ManifestRef, RunId
from feature_forge.contracts.base import ContractModel
from feature_forge.contracts.stages import Layer, RunState, StageResult

StageResult.model_rebuild(_types_namespace={"ArtifactDescriptor": ArtifactDescriptor})


class RunRequest(ContractModel):
    """Inputs that define one experiment case."""

    dataset: str = Field(min_length=1)
    method: str = Field(min_length=1)
    model: str = Field(min_length=1)
    seed: int
    mode: str | None = None
    cv_folds: int | None = Field(default=None, ge=2)
    options: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSnapshot(ContractModel):
    """Secret-free runtime information captured with a run."""

    python_version: str
    operating_system: str
    architecture: str
    feature_forge_version: str
    artifact_schema_versions: dict[str, str] = Field(default_factory=dict)
    random_seeds: dict[str, int] = Field(default_factory=dict)


class RunManifest(ContractModel):
    """Authoritative manifest for one committed layer package."""

    schema_version: Literal["1"] = "1"
    layer: Layer
    package_kind: Literal["bronze", "silver", "gold", "platinum"]
    layer_fingerprint: str = Field(min_length=1)
    run_id: RunId
    case_fingerprint: str = Field(min_length=1)
    state: RunState
    request: RunRequest
    environment: EnvironmentSnapshot
    stages: list[StageResult] = Field(default_factory=list)
    upstream_manifests: list[ManifestRef] = Field(default_factory=list)
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime | None = None

    @field_validator("package_kind")
    @classmethod
    def _package_kind_matches_layer(cls, value: str, info: Any) -> str:
        layer = info.data.get("layer")
        if layer is not None and value != layer.value:
            raise ValueError("package_kind must match manifest layer")
        return value
