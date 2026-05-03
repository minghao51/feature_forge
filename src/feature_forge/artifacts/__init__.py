"""Unified artifact access for all feature engineering methods."""

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter
from feature_forge.artifacts.comparison import compare_methods
from feature_forge.artifacts.dashboard import ArtifactDashboard
from feature_forge.artifacts.diff import ArtifactDiff
from feature_forge.artifacts.schema import (
    ArtifactBundle,
    ArtifactConfigSchema,
    FeatureMetadata,
    IterationRecord,
    ProvenanceRecord,
)
from feature_forge.artifacts.storage import DataFrameStorage, LazyDataFrameRef

__all__ = [
    "ArtifactBundle",
    "ArtifactConfig",
    "ArtifactConfigSchema",
    "ArtifactDashboard",
    "ArtifactDiff",
    "ArtifactExporter",
    "DataFrameStorage",
    "FeatureMetadata",
    "IterationRecord",
    "LazyDataFrameRef",
    "ProvenanceRecord",
    "compare_methods",
]
