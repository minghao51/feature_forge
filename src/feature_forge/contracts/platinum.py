"""Strict contracts for reconstructable Platinum evaluation evidence."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import ConfigDict, Field, model_validator

from feature_forge.contracts.artifacts import ManifestRef, RunId
from feature_forge.contracts.base import ContractModel, EvaluationProtocol
from feature_forge.contracts.materialization import Materialization
from feature_forge.contracts.runs import RunManifest
from feature_forge.contracts.stages import CheckResult
from feature_forge.evaluation.metrics import MetricDirection


class ModelSpecification(ContractModel):
    schema_version: Literal["1"] = "1"
    name: str = Field(min_length=1)
    estimator_class: str = Field(min_length=1)
    distribution: str = Field(min_length=1)
    distribution_version: str = Field(min_length=1)
    resolved_params: dict[str, Any] = Field(default_factory=dict)
    task: Literal["classification", "regression"]
    seed: int


class EvaluationPolicy(ContractModel):
    schema_version: Literal["1"] = "1"
    alignment: Literal["normalize_by_row_id", "strict_order"] = "normalize_by_row_id"
    aggregation: Literal["unweighted_mean_of_fold_metrics"] = "unweighted_mean_of_fold_metrics"
    preprocessing: Literal["fit_on_training_fold"] = "fit_on_training_fold"
    minimum_successful_folds: int = Field(default=2, ge=2)
    estimator_threads: int = Field(default=1, ge=1)
    blas_threads: int = Field(default=1, ge=1)
    selection_partition: Literal["discovery", "evaluation", "all"] = "discovery"
    # Partition provenance, ADR 0018; selection_partition/protocol combinations
    # are enforced at PlatinumRequest validation.
    evaluation_protocol: EvaluationProtocol = "holdout"


class UncertaintyPolicy(ContractModel):
    schema_version: Literal["1"] = "1"
    method: Literal["paired_t"] = "paired_t"
    confidence_level: float = Field(default=0.95, gt=0, lt=1)


class SelectionPolicy(ContractModel):
    schema_version: Literal["1"] = "1"
    profile: Literal["compatibility", "recommended"] = "compatibility"
    # Enforced by greedy selection on discovery folds (plan 23 PR 3).
    minimum_practical_gain: float = Field(default=0.0, ge=0)
    # Enforced by greedy selection on discovery folds (plan 23 PR 3).
    require_positive_lower_bound: bool = False
    # Enforced by greedy selection on discovery folds (plan 23 PR 3).
    max_selected_features: int | None = Field(default=None, ge=1)


class PlatinumRequest(ContractModel):
    schema_version: Literal["1"] = "1"
    run_id: RunId
    case_fingerprint: str = Field(min_length=1)
    silver_manifest: ManifestRef
    gold_manifest: ManifestRef
    gold_fingerprint: str = Field(min_length=1)
    fold_fingerprint: str = Field(min_length=1)
    platinum_input_fingerprint: str = Field(min_length=1)
    model: ModelSpecification
    metric: str = Field(min_length=1)
    metric_direction: MetricDirection
    evaluation_policy: EvaluationPolicy = Field(default_factory=EvaluationPolicy)
    uncertainty_policy: UncertaintyPolicy = Field(default_factory=UncertaintyPolicy)
    selection_policy: SelectionPolicy = Field(default_factory=SelectionPolicy)

    @model_validator(mode="after")
    def validate_fingerprint(self) -> PlatinumRequest:
        from feature_forge.contracts.identity import platinum_input_fingerprint

        expected = platinum_input_fingerprint(
            gold_fingerprint_value=self.gold_fingerprint,
            model_name=self.model.name,
            model_version=self.model.distribution_version,
            model_config=self.model.model_dump(mode="json"),
            metric=self.metric,
            fold_fingerprint=self.fold_fingerprint,
            evaluation_policy={
                **self.evaluation_policy.model_dump(mode="json"),
                "metric_direction": self.metric_direction.value,
                "selection_policy": self.selection_policy.model_dump(mode="json"),
            },
            uncertainty_policy=self.uncertainty_policy.model_dump(mode="json"),
        )
        if self.platinum_input_fingerprint != expected:
            raise ValueError("platinum_input_fingerprint does not match request fields")
        return self

    @model_validator(mode="after")
    def validate_selection_partition_profile(self) -> PlatinumRequest:
        """Reject unsupported selection_partition/protocol combinations (ADR 0018 §5)."""
        partition = self.evaluation_policy.selection_partition
        protocol = self.evaluation_policy.evaluation_protocol
        if partition == "evaluation":
            raise ValueError(
                "selection_partition='evaluation' selects features on the reported "
                "evaluation folds, which is leakage in every profile; use 'discovery' "
                "(ADR 0018 decision 5)"
            )
        if protocol == "holdout" and partition != "discovery":
            raise ValueError(
                f"selection_partition={partition!r} is unsupported under "
                "evaluation_protocol='holdout', which selects on 'discovery' folds only; "
                "'all' is legal only under the explicit 'compatibility' protocol "
                "(ADR 0018 decision 5)"
            )
        return self


class AggregateMetric(ContractModel):
    schema_version: Literal["1"] = "1"
    metric: str
    metric_direction: MetricDirection
    aggregation: Literal["unweighted_mean_of_fold_metrics"]
    baseline_score: float
    enhanced_score: float
    legacy_gain: float
    directional_gain: float
    successful_folds: int = Field(ge=2)


class UncertaintySummary(ContractModel):
    schema_version: Literal["1"] = "1"
    method: Literal["paired_t"] = "paired_t"
    confidence_level: float
    pair_count: int = Field(ge=2)
    mean_directional_gain: float
    standard_deviation: float
    standard_error: float
    lower_bound: float
    upper_bound: float


class PlatinumSelectionDecision(ContractModel):
    schema_version: Literal["1"] = "1"
    candidate_id: str
    feature_name: str
    selected: bool
    reason_code: str
    reason: str
    legacy_gain: float
    directional_gain: float
    # Carries the evaluation-aggregate directional interval bound, not a
    # per-step discovery bound; the bounds that actually gated greedy
    # selection are persisted in selection_steps.json (ranking entries'
    # lower_bound), and offline reconstruction of the gating recomputes them
    # from the persisted discovery fold metrics.
    lower_bound: float


class PlatinumEvidenceIndex(ContractModel):
    """v2 evidence-set marker binding the durable Platinum evidence set.

    Per-artifact ``schema_version`` fields stay at ``"1"`` (the artifact store
    pins them); the evidence-set schema bump lives only in this persisted
    marker (ADR 0018 decision 8).
    """

    evidence_schema_version: Literal["2"] = "2"
    selected: list[str]
    selection_partition: str
    evaluation_protocol: EvaluationProtocol
    confidence_level: float


class PlatinumMaterialization(Materialization):
    pass


class PlatinumPackage(ContractModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    manifest: RunManifest
    request: PlatinumRequest
    fold_assignments: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    aggregate: AggregateMetric
    uncertainty: UncertaintySummary
    decisions: list[PlatinumSelectionDecision]
    checks: list[CheckResult]
