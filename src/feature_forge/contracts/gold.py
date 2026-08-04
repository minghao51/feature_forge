"""Strict contracts for Gold feature evidence and offline replay."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

import pandas as pd
from pydantic import ConfigDict, Field, model_validator

from feature_forge.contracts.artifacts import (
    ManifestRef,
    RelativePath,
    RunId,
    Sha256,
)
from feature_forge.contracts.base import ContractModel
from feature_forge.contracts.materialization import Materialization
from feature_forge.contracts.runs import RunManifest
from feature_forge.contracts.stages import CheckResult


class FeatureDecisionState(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


class FeatureCandidate(ContractModel):
    schema_version: Literal["1"] = "1"
    candidate_id: RunId
    name: str = Field(min_length=1)
    batch_index: int = Field(ge=0)
    code_path: RelativePath
    specification: dict[str, Any] = Field(default_factory=dict)


class FeatureProvenance(ContractModel):
    schema_version: Literal["1"] = "1"
    candidate_id: RunId
    method: str = Field(min_length=1)
    method_version: str = Field(min_length=1)
    round_index: int | None = Field(default=None, ge=0)
    prompt_bundle_fingerprint: str = Field(min_length=1)
    code_sha256: Sha256
    dependencies: list[str] = Field(default_factory=list)
    source_columns: list[str] = Field(default_factory=list)


class FeatureDecision(ContractModel):
    schema_version: Literal["1"] = "1"
    candidate_id: RunId
    state: FeatureDecisionState
    reason_code: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GoldRequest(ContractModel):
    schema_version: Literal["1"] = "1"
    run_id: RunId
    case_fingerprint: str = Field(min_length=1)
    silver_manifest: ManifestRef
    silver_fingerprint: str = Field(min_length=1)
    gold_input_fingerprint: str = Field(min_length=1)
    method_name: str = Field(min_length=1)
    method_version: str = Field(min_length=1)
    method_config: dict[str, Any] = Field(default_factory=dict)
    prompt_bundle_fingerprint: str = Field(min_length=1)
    generated_contract_version: str = "1"
    selection_policy: dict[str, Any] = Field(default_factory=lambda: {"policy": "validation"})
    persist_candidates: bool = True
    value_rtol: float = Field(default=1e-9, ge=0)
    value_atol: float = Field(default=1e-12, ge=0)

    @model_validator(mode="after")
    def _valid_input_fingerprint(self) -> Self:
        from feature_forge.contracts.identity import gold_input_fingerprint

        expected = gold_input_fingerprint(
            silver_fingerprint_value=self.silver_fingerprint,
            method_name=self.method_name,
            method_version=self.method_version,
            method_config=self.method_config,
            prompt_bundle_fingerprint=self.prompt_bundle_fingerprint,
            generated_contract_version=self.generated_contract_version,
            selection_policy=self.selection_policy,
        )
        if self.gold_input_fingerprint != expected:
            raise ValueError("gold_input_fingerprint does not match Gold request fields")
        return self


class GoldFeatureCounts(ContractModel):
    schema_version: Literal["1"] = "1"
    candidates_proposed: int = Field(ge=0)
    candidates_executed: int = Field(ge=0)
    candidates_accepted: int = Field(ge=0)
    accepted_output_columns: int = Field(ge=0)
    failure_counts: dict[str, int] = Field(default_factory=dict)


class GoldMaterialization(Materialization):
    counts: GoldFeatureCounts


class GoldPackage(ContractModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    manifest: RunManifest
    request: GoldRequest
    candidates: list[FeatureCandidate]
    provenance: list[FeatureProvenance]
    decisions: list[FeatureDecision]
    candidate_features: pd.DataFrame | None = None
    accepted_features: pd.DataFrame
    checks: list[CheckResult]
    code: dict[str, str]
    dependencies: dict[str, Any]
    counts: GoldFeatureCounts


class GoldExecutionResult(ContractModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    materialization: GoldMaterialization
    package: GoldPackage
    counts: GoldFeatureCounts
