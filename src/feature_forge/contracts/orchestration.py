"""Strict contracts for scheduling, resources, recovery, and lifecycle events."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from feature_forge.contracts.artifacts import ManifestRef
from feature_forge.contracts.base import ContractModel
from feature_forge.contracts.stages import FailureClass, Layer, RunState


class StageDisposition(StrEnum):
    """How a stage contributed to the current run."""

    EXECUTED = "executed"
    REUSED = "reused"
    SKIPPED = "skipped"


class FailureRecord(ContractModel):
    """Serializable failure context shared by sequential and process execution."""

    schema_version: Literal["1"] = "1"
    failure_class: FailureClass
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stage: str = "case"
    retryable: bool = False
    attempt: int = Field(default=1, ge=1)
    cause_types: list[str] = Field(default_factory=list)


class ResourceConfig(ContractModel):
    """Requested global and per-case resource ceilings."""

    schema_version: Literal["1"] = "1"
    experiment_workers: int = Field(default=1, ge=1)
    llm_concurrency: int = Field(default=3, ge=1)
    sandbox_workers: int = Field(default=1, ge=1)
    candidate_batch_size: int = Field(default=1, ge=1)
    cv_workers: int = Field(default=1, ge=1)
    cv_backend: Literal["threading", "loky"] = "threading"
    model_threads: int = Field(default=1, ge=1)
    blas_threads: int = Field(default=1, ge=1)
    memory_budget_mb: int | None = Field(default=None, ge=128)
    max_attempts: int = Field(default=1, ge=1)
    backoff_base_seconds: float = Field(default=1.0, gt=0)
    backoff_max_seconds: float = Field(default=30.0, gt=0)


class EffectiveResourcePlan(ContractModel):
    """Deterministic safe resource plan applied to every case."""

    schema_version: Literal["1"] = "1"
    experiment_workers: int = Field(ge=1)
    llm_concurrency: int = Field(ge=1)
    sandbox_workers: int = Field(ge=1)
    candidate_batch_size: int = Field(ge=1)
    cv_workers: int = Field(ge=1)
    cv_backend: Literal["threading", "loky"]
    model_threads: int = Field(ge=1)
    blas_threads: int = Field(ge=1)
    memory_budget_mb: int | None = Field(default=None, ge=128)
    max_attempts: int = Field(ge=1)
    backoff_base_seconds: float = Field(gt=0)
    backoff_max_seconds: float = Field(gt=0)
    heavy_parallel_layer: Literal["outer", "llm", "cv", "none"]
    adjustments: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _only_one_heavy_layer(self) -> EffectiveResourcePlan:
        heavy = sum(
            value > 1 for value in (self.experiment_workers, self.llm_concurrency, self.cv_workers)
        )
        if heavy > 1:
            raise ValueError("effective resource plan enables multiple heavy parallel layers")
        if self.experiment_workers > 1 and (
            self.sandbox_workers > 1 or self.candidate_batch_size > 1
        ):
            raise ValueError("outer parallelism requires serial per-case execution")
        if (self.experiment_workers > 1 or self.cv_workers > 1) and (
            self.model_threads > 1 or self.blas_threads > 1
        ):
            raise ValueError("outer/CV parallelism requires single-threaded model and BLAS work")
        nested_products = {
            "model": self.experiment_workers
            * self.cv_workers
            * self.model_threads
            * self.blas_threads,
            "sandbox": self.experiment_workers * self.sandbox_workers * self.candidate_batch_size,
        }
        ceilings = {
            "model": max(
                self.experiment_workers,
                self.cv_workers,
                self.model_threads,
                self.blas_threads,
            ),
            "sandbox": max(
                self.experiment_workers,
                self.sandbox_workers,
                self.candidate_batch_size,
            ),
        }
        invalid = [name for name, product in nested_products.items() if product > ceilings[name]]
        if invalid:
            raise ValueError(f"effective resource plan multiplies nested limits: {invalid}")
        return self


class ResumePolicy(ContractModel):
    """Fail-closed policy for reusable and damaged artifact packages."""

    schema_version: Literal["1"] = "1"
    enabled: bool = False
    recompute_invalid: bool = False
    allow_cross_run_reuse: bool = True


class ResumeStageDecision(ContractModel):
    """Verified decision for one layer in dependency order."""

    schema_version: Literal["1"] = "1"
    layer: Layer
    expected_fingerprint: str = Field(min_length=1)
    disposition: StageDisposition
    state: RunState | None = None
    manifest_ref: ManifestRef | None = None
    reason: str = Field(min_length=1)


class ResumePlan(ContractModel):
    """Side-effect-free contiguous resume plan."""

    schema_version: Literal["1"] = "1"
    run_id: str = Field(min_length=1)
    dry_run: bool = False
    decisions: list[ResumeStageDecision]
    next_layer: Layer | None = None
    complete: bool = False


class RunEvent(ContractModel):
    """Append-only control-plane event consumed by the future catalog."""

    schema_version: Literal["1"] = "1"
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    occurred_at: datetime
    state: RunState | None = None
    stage: str | None = None
    layer: Layer | None = None
    disposition: StageDisposition | None = None
    failure: FailureRecord | None = None
    manifest_ref: ManifestRef | None = None
    fingerprint: str | None = None
    attempt: int = Field(default=1, ge=1)
    details: dict[str, Any] = Field(default_factory=dict)
