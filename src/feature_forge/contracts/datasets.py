"""Contracts shared by the deterministic Bronze and Silver dataflow."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import ConfigDict, Field

from feature_forge.contracts.artifacts import RunId
from feature_forge.contracts.base import ContractModel
from feature_forge.contracts.materialization import Materialization
from feature_forge.contracts.runs import RunManifest
from feature_forge.contracts.stages import CheckResult


class DatasetRequest(ContractModel):
    """Resolved dataset inputs for one Silver materialization."""

    name: str = Field(min_length=1)
    target: str | None = None
    task: Literal["classification", "regression"]
    run_id: RunId
    case_fingerprint: str = Field(min_length=1)
    split_seed: int
    cv_folds: int = Field(ge=2)
    # Fraction of rows reserved as an evaluation-only partition. Feature
    # selection sees the complementary discovery rows; the reported Platinum
    # score uses the evaluation rows only. 0.0 disables the partition.
    evaluation_holdout_fraction: float = Field(default=0.25, ge=0.0, lt=0.5)
    source_policy: Literal["reference", "snapshot"] = "reference"
    canonicalization_config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BronzeRecord(ContractModel):
    """Reproducibility record for the dataset entering the experiment."""

    schema_version: Literal["1"] = "1"
    source: str = Field(min_length=1)
    source_checksum: str = Field(min_length=1)
    target: str = Field(min_length=1)
    task: Literal["classification", "regression"]
    snapshot_mode: Literal["reference", "snapshot"]
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SilverMaterialization(Materialization):
    """Result of a Silver boundary, including dry-run outcomes."""


class BronzeMaterialization(Materialization):
    """Result of the Bronze reference/snapshot boundary."""


class SilverPackage(ContractModel):
    """Verified Silver package loaded for deterministic offline reuse."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    manifest: RunManifest
    bronze: BronzeRecord
    canonical_features: pd.DataFrame
    canonical_target: pd.DataFrame
    row_ids: pd.DataFrame
    fold_assignments: pd.DataFrame
    profile: dict[str, Any]
    checks: list[CheckResult]
