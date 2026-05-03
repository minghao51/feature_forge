"""ArtifactExporter ABC and ArtifactConfig for unified artifact access.

All feature engineering methods implement ArtifactExporter to provide
consistent access to generated scripts, intermediate DataFrames,
and feature metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from feature_forge.artifacts.schema import (
    ArtifactBundle,
    ArtifactConfigSchema,
    FeatureMetadata,
)


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
    """

    storage_mode: Literal["memory", "disk", "hybrid"] = "memory"
    storage_format: Literal["parquet", "csv", "feather"] = "parquet"
    spill_threshold_bytes: int = 50 * 1024 * 1024  # 50 MB
    storage_dir: str = ".feature_forge_artifacts"

    def __post_init__(self) -> None:
        if self.storage_mode == "memory":
            return
        os.makedirs(self.storage_dir, exist_ok=True)

    def to_schema(self) -> ArtifactConfigSchema:
        """Convert to Pydantic schema for validation."""
        return ArtifactConfigSchema(
            storage_mode=self.storage_mode,
            storage_format=self.storage_format,
            spill_threshold_bytes=self.spill_threshold_bytes,
            storage_dir=self.storage_dir,
        )


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

    def get_artifact_bundle(self) -> ArtifactBundle:
        """Return artifacts as a validated Pydantic model.

        Subclasses may override to populate structured fields.
        The default infers from get_artifacts() and feature_metadata().
        """
        return ArtifactBundle(
            method_name=getattr(self, "name", "unknown"),
            generated_scripts=self.generated_scripts,
            feature_metadata=[
                FeatureMetadata(**m) if isinstance(m, dict) else m
                for m in self.feature_metadata
            ],
            custom=self.get_artifacts(),
        )

    def save_artifacts(self, path: str | Path) -> None:
        """Persist the full artifact bundle to disk.

        Writes a JSON metadata file plus individual parquet files for
        DataFrame values found in get_artifacts().

        Args:
            path: Directory or file stem. If a directory, writes
                ``artifacts.json`` inside it. If a file path, uses it
                as the JSON path and creates a ``_dataframes`` sibling
                directory.
        """
        path = Path(path)
        if path.suffix == ".json":
            json_path = path
            df_dir = path.with_suffix("")
        else:
            path.mkdir(parents=True, exist_ok=True)
            json_path = path / "artifacts.json"
            df_dir = path / "dataframes"

        df_dir.mkdir(parents=True, exist_ok=True)

        from feature_forge.artifacts.storage import LazyDataFrameRef

        artifacts = self.get_artifacts()
        serializable: dict[str, Any] = {}

        for key, value in artifacts.items():
            if isinstance(value, pd.DataFrame):
                safe_key = key.replace("/", "_").replace(" ", "_")
                df_path = df_dir / f"{safe_key}.parquet"
                value.to_parquet(df_path)
                serializable[key] = {"__type__": "dataframe", "path": str(df_path)}
            elif isinstance(value, LazyDataFrameRef):
                safe_key = key.replace("/", "_").replace(" ", "_")
                dest = df_dir / f"{safe_key}.parquet"
                import shutil
                shutil.copy(value.path, dest)
                serializable[key] = {"__type__": "lazy_dataframe", "path": str(dest), "fmt": value.fmt}
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                serializable[key] = value
            else:
                serializable[key] = {"__type__": "str", "value": str(value)}

        # Also store structured bundle (exclude custom since raw artifacts
        # are stored separately and may contain non-serializable objects)
        bundle = self.get_artifact_bundle()
        bundle_dict = bundle.model_dump(mode="json", exclude={"custom"})
        serializable["__bundle__"] = bundle_dict

        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    @classmethod
    def load_artifacts(cls, path: str | Path) -> dict[str, Any]:
        """Load an artifact bundle from disk.

        Args:
            path: Path to the JSON file or parent directory.

        Returns:
            Dictionary with DataFrames restored from parquet and all other
            values as their native types.
        """
        path = Path(path)
        if path.is_dir():
            json_path = path / "artifacts.json"
        else:
            json_path = path

        with open(json_path) as f:
            data: dict[str, Any] = json.load(f)

        from feature_forge.artifacts.storage import LazyDataFrameRef

        restored: dict[str, Any] = {}
        for key, value in data.items():
            if key == "__bundle__":
                restored[key] = ArtifactBundle(**value)
                continue
            if isinstance(value, dict) and value.get("__type__") == "dataframe":
                restored[key] = pd.read_parquet(value["path"])
            elif isinstance(value, dict) and value.get("__type__") == "lazy_dataframe":
                restored[key] = LazyDataFrameRef(value["path"], value.get("fmt", "parquet"))
            else:
                restored[key] = value

        return restored

    def log_artifacts(
        self,
        tracker: Any,
        prefix: str = "",
    ) -> None:
        """Log all artifacts to an experiment tracker.

        Args:
            tracker: An ExperimentTracker instance with log_artifacts_dict support.
            prefix: Optional prefix prepended to all artifact keys.
        """
        if not hasattr(tracker, "log_artifacts_dict"):
            return

        artifacts = self.get_artifacts()
        deduped: dict[str, Any] = {}
        seen_hashes: set[str] = set()

        for key, value in artifacts.items():
            # Deduplicate identical code strings
            if isinstance(value, str) and value.strip():
                h = hashlib.sha256(value.encode()).hexdigest()[:16]
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
            deduped[key] = value

        tracker.log_artifacts_dict(deduped, prefix=prefix)

    def provenance_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with per-feature provenance records.

        Subclasses should override to populate ProvenanceRecord objects.
        Default returns an empty DataFrame.
        """
        bundle = self.get_artifact_bundle()
        return bundle.to_provenance_dataframe()
