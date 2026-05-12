"""Pydantic schema models for artifact validation and type safety.

Provides structured models for artifact metadata, feature records,
iteration logs, and full artifact bundles.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ArtifactConfigSchema(BaseModel):
    """Pydantic model for artifact storage configuration.

    Mirrors ``ArtifactConfig`` dataclass with validation.
    """

    model_config = ConfigDict(frozen=True)

    storage_mode: Literal["memory", "disk", "hybrid"] = "memory"
    storage_format: Literal["parquet", "csv", "feather"] = "parquet"
    spill_threshold_bytes: int = Field(default=50 * 1024 * 1024, ge=0)
    storage_dir: str = ".feature_forge_artifacts"

    @field_validator("storage_dir")
    @classmethod
    def _ensure_dir(cls, v: str) -> str:
        if not v:
            msg = "storage_dir cannot be empty"
            raise ValueError(msg)
        return v


class FeatureMetadata(BaseModel):
    """Metadata for a single engineered feature.

    Attributes:
        name: Feature column name.
        method: Method that produced it (e.g. 'llmfe', 'caafe', 'malmas').
        round: Round number (for multi-round methods).
        agent: Agent name (for multi-agent methods).
        iteration: Iteration number (for iterative methods).
        code: Code snippet that generated the feature.
        gain: Cross-validation gain vs baseline.
        importance: Feature importance score (if available).
        provenance: Arbitrary provenance metadata.
    """

    model_config = ConfigDict(frozen=False)

    name: str
    method: str = "unknown"
    round: int | None = None
    agent: str | None = None
    iteration: int | None = None
    code: str | None = None
    gain: float | None = None
    importance: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class IterationRecord(BaseModel):
    """Schema for a single iteration in iterative feature engineering.

    Attributes:
        iteration: Iteration index.
        generated_code: Code block produced by the LLM.
        kept_features: List of feature names that were kept.
        gains: Dict mapping feature name to gain value.
        error: Error message if iteration failed.
        prompt: Prompt sent to LLM (optional).
    """

    model_config = ConfigDict(frozen=False)

    iteration: int
    generated_code: str | None = None
    kept_features: list[str] = Field(default_factory=list)
    gains: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    prompt: str | None = None


class ProvenanceRecord(BaseModel):
    """Structured provenance for a feature's full lineage.

    Attributes:
        feature_name: The generated feature's column name.
        source_method: Baseline or pipeline name (e.g. 'malmas').
        source_agent: Agent that proposed the feature.
        round_index: Round in which feature was generated.
        iteration_index: Iteration within the round.
        prompt: Prompt used to generate the feature.
        generated_code: Full code block for the feature.
        cv_gain: Cross-validated gain vs original baseline.
        timestamp: When the feature was generated.
    """

    model_config = ConfigDict(frozen=False)

    feature_name: str
    source_method: str
    source_agent: str | None = None
    round_index: int | None = None
    iteration_index: int | None = None
    prompt: str | None = None
    generated_code: str | None = None
    cv_gain: float | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


class ArtifactBundle(BaseModel):
    """Validated container for a complete artifact bundle.

    Attributes:
        method_name: Name of the method that produced these artifacts.
        generated_scripts: List of code blocks.
        feature_metadata: Structured metadata for each feature.
        iteration_records: Per-iteration records (for iterative methods).
        provenance_records: Per-feature provenance records.
        custom: Arbitrary additional artifacts.
    """

    model_config = ConfigDict(frozen=False)

    method_name: str
    generated_scripts: list[str] = Field(default_factory=list)
    feature_metadata: list[FeatureMetadata] = Field(default_factory=list)
    iteration_records: list[IterationRecord] = Field(default_factory=list)
    provenance_records: list[ProvenanceRecord] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generated_scripts")
    @classmethod
    def _dedup_scripts(cls, v: list[str]) -> list[str]:
        seen: set[int] = set()
        out: list[str] = []
        for s in v:
            h = hash(s)
            if h not in seen:
                seen.add(h)
                out.append(s)
        return out

    def to_feature_dataframe(self) -> pd.DataFrame:
        """Export feature metadata as a flat DataFrame."""
        if not self.feature_metadata:
            return pd.DataFrame()
        rows = [f.model_dump() for f in self.feature_metadata]
        return pd.DataFrame(rows)

    def to_provenance_dataframe(self) -> pd.DataFrame:
        """Export provenance records as a flat DataFrame."""
        if not self.provenance_records:
            return pd.DataFrame()
        rows = [p.model_dump() for p in self.provenance_records]
        return pd.DataFrame(rows)
