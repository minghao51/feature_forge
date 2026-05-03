"""Lazy DataFrame loading and storage management.

Provides LazyDataFrameRef for on-demand loading and DataFrameStorage
for managing where DataFrames live (memory vs disk).
"""

from __future__ import annotations

import os

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig


class LazyDataFrameRef:
    """Lazy-loading wrapper for disk-backed DataFrames.

    Instead of loading a DataFrame into memory at creation time,
    this wrapper holds the path and format, and loads the DataFrame
    only when accessed via .load().

    Supports pickling/serialization for multiprocessing.
    """

    def __init__(self, path: str, fmt: str = "parquet") -> None:
        self.path = path
        self.fmt = fmt
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Load and cache the DataFrame from disk."""
        if self._df is None:
            self._df = self._read(self.path, self.fmt)
        return self._df

    def invalidate(self) -> None:
        """Drop the cached DataFrame, forcing re-read on next load()."""
        self._df = None

    @staticmethod
    def _read(path: str, fmt: str) -> pd.DataFrame:
        """Read a DataFrame from disk."""
        if fmt == "parquet":
            return pd.read_parquet(path)
        if fmt == "csv":
            return pd.read_csv(path, index_col=0)
        if fmt == "feather":
            return pd.read_feather(path)
        raise ValueError(f"Unsupported format: {fmt}")

    def __repr__(self) -> str:
        loaded = "loaded" if self._df is not None else "not loaded"
        return f"LazyDataFrameRef(path={self.path!r}, fmt={self.fmt!r}, {loaded})"


class DataFrameStorage:
    """Manages DataFrame persistence based on ArtifactConfig.

    In 'memory' mode, DataFrames are returned as-is.
    In 'disk' mode, DataFrames are written to disk and a LazyDataFrameRef
        is returned.
    In 'hybrid' mode, small DataFrames stay in memory while large ones
        are spilled to disk.
    """

    def __init__(self, config: ArtifactConfig | None = None) -> None:
        self.config = config or ArtifactConfig()

    def store(
        self,
        key: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame | LazyDataFrameRef:
        """Store a DataFrame according to the configured storage mode.

        Args:
            key: A descriptive key used as the filename stem.
            df: The DataFrame to store.

        Returns:
            The original DataFrame (memory), or a LazyDataFrameRef (disk).
        """
        if self.config.storage_mode == "memory":
            return df

        if self.config.storage_mode == "hybrid":
            size = df.memory_usage(deep=True).sum()
            if size < self.config.spill_threshold_bytes:
                return df

        path = self._make_path(key)
        self._write(df, path, self.config.storage_format)
        return LazyDataFrameRef(path, self.config.storage_format)

    def _make_path(self, key: str) -> str:
        """Build a file path for a given key."""
        ext = self.config.storage_format
        safe_key = key.replace("/", "_").replace(" ", "_")
        return os.path.join(self.config.storage_dir, f"{safe_key}.{ext}")

    @staticmethod
    def _write(df: pd.DataFrame, path: str, fmt: str) -> None:
        """Write a DataFrame to disk."""
        if fmt == "parquet":
            df.to_parquet(path)
        elif fmt == "csv":
            df.to_csv(path)
        elif fmt == "feather":
            df.to_feather(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
