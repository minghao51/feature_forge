"""ArtifactExporter ABC and ArtifactConfig for unified artifact access.

All feature engineering methods implement ArtifactExporter to provide
consistent access to generated scripts, intermediate DataFrames,
and feature metadata.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from feature_forge.contracts.artifacts import validate_identifier


@dataclass
class ArtifactConfig:
    """Configuration for artifact storage behavior.

    Attributes:
        storage_mode: Where to store DataFrames.
            - 'memory': Keep everything in RAM.
            - 'disk': Write to parquet/csv/feather on disk, return LazyDataFrameRef.
            - 'hybrid': Keep small DataFrames in memory, spill large ones to disk.
        storage_format: File format for disk storage ('parquet', 'csv', 'feather').
        spill_threshold_bytes: In hybrid mode, DataFrames larger than this are
            written to disk. Ignored for 'memory' and 'disk' modes.
        storage_dir: Directory for disk-based artifacts.
        run_id: Optional safe identifier used to isolate disk artifacts by run.
    """

    storage_mode: Literal["memory", "disk", "hybrid"] = "memory"
    storage_format: Literal["parquet", "csv", "feather"] = "parquet"
    spill_threshold_bytes: int = 50 * 1024 * 1024  # 50 MB
    storage_dir: str = ".feature_forge_artifacts"
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.run_id is not None:
            validate_identifier(self.run_id)
        if self.storage_mode == "memory":
            return
        os.makedirs(self.storage_dir, exist_ok=True)


class ArtifactExporter(ABC):
    """Abstract base class for unified artifact access.

    All feature engineering methods (MALMAS, LLMFE, CAAFE, OpenFE)
    implement this mixin to expose generated code, intermediate DataFrames,
    and feature metadata through a consistent interface.
    """

    def __init__(self, artifact_config: ArtifactConfig | None = None) -> None:
        self.artifact_config = artifact_config or ArtifactConfig()

    @property
    @abstractmethod
    def generated_scripts(self) -> list[str]:
        """Return all generated code blocks as strings."""

    @property
    def intermediate_dataframes(self) -> dict[str, pd.DataFrame]:
        """Return intermediate DataFrames produced during fit/transform.

        Subclasses should override this to return method-specific artifacts.
        The default implementation filters get_artifacts() for DataFrame or
        LazyDataFrameRef values, resolving refs automatically.
        """
        from feature_forge.artifacts.storage import LazyDataFrameRef

        artifacts = self.get_artifacts()
        result: dict[str, pd.DataFrame] = {}
        for k, v in artifacts.items():
            if isinstance(v, pd.DataFrame):
                result[k] = v
            elif isinstance(v, LazyDataFrameRef):
                result[k] = v.load()
        return result

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        """Return feature specifications/metadata.

        Subclasses should override this for method-specific metadata.
        The default returns an empty list.
        """
        return []

    @abstractmethod
    def get_artifacts(self) -> dict[str, Any]:
        """Return all artifacts as a flat dictionary.

        Keys should be descriptive and prefixed with the method name where
        appropriate (e.g. 'round_1_generated_code').
        """
