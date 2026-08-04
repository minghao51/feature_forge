"""Layer, stage, and verification contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import Field

from feature_forge.contracts.base import ContractModel

if TYPE_CHECKING:
    from feature_forge.contracts.artifacts import ArtifactDescriptor


class Layer(StrEnum):
    """Durable maturity layer for an experiment artifact."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class RunState(StrEnum):
    """Lifecycle state of a run or stage."""

    PLANNED = "planned"
    RUNNING = "running"
    PARTIAL = "partial"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureClass(StrEnum):
    """Failure category used to decide whether recovery is safe."""

    TRANSIENT = "transient"
    DETERMINISTIC = "deterministic"
    POLICY = "policy"
    RESOURCE = "resource"
    CANCELLED = "cancelled"


class CheckSeverity(StrEnum):
    """Severity used to distinguish advisory and required checks."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CheckResult(ContractModel):
    """Result of one required or advisory validation check."""

    check_id: str = Field(min_length=1)
    severity: CheckSeverity = CheckSeverity.ERROR
    passed: bool
    required: bool = True
    observed: Any | None = None
    expected: Any | None = None
    message: str = Field(min_length=1)
    affected_artifacts: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class StageResult(ContractModel):
    """Persisted outcome for one executable stage."""

    stage: str = Field(min_length=1)
    state: RunState
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    failure_class: FailureClass | None = None
    error_type: str | None = None
    error_message: str | None = None
    skipped_reason: str | None = None


class VerificationReport(ContractModel):
    """Read-only verification outcome for a committed artifact namespace."""

    valid: bool
    run_id: str
    layer: Layer
    manifest_path: str
    checked_artifacts: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
