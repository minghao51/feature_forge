"""Shared strict configuration for versioned persisted contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Protocol profiles from ADR 0018: "holdout" reserves deterministic
# discovery/evaluation partitions and fails closed on small data;
# "compatibility" reproduces the explicit legacy all-row behavior, whose
# estimates are selection-biased.
# NOTE: feature_forge.config.EvaluationConfig.protocol intentionally duplicates
# this Literal (config must stay dependency-free); keep the two in sync.
EvaluationProtocol = Literal["holdout", "compatibility"]


class ContractModel(BaseModel):
    """Base model that rejects unknown fields instead of hiding schema drift."""

    model_config = ConfigDict(extra="forbid")
