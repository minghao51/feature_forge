"""Dataset registry for managing available datasets."""

from __future__ import annotations

import importlib.metadata
import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd


class DatasetRegistry:
    """Registry of available datasets with loaders.

    Supports Kaggle datasets, local sample datasets, and entry-point
    registered datasets (discovered via ``feature_forge.datasets``).
    """

    SAMPLE_DATASETS: ClassVar[dict[str, dict[str, Any]]] = {
        "titanic": {
            "source": "kaggle",
            "slug": "titanic",
            "target": "Survived",
            "task": "classification",
        },
        "house_prices": {
            "source": "kaggle",
            "slug": "house-prices-advanced-regression-techniques",
            "target": "SalePrice",
            "task": "regression",
        },
    }

    ENTRY_POINT_GROUP = "feature_forge.datasets"

    def __init__(self, sample_dir: str = "data/samples") -> None:
        self.sample_dir = Path(sample_dir)
        self._datasets: dict[str, dict[str, Any]] = dict(self.SAMPLE_DATASETS)
        self._entry_point_loaders: dict[str, Callable[[], dict[str, Any]]] = {}
        self._entry_points_loaded = False
        self._load_local_samples()

    def _ensure_entry_points_loaded(self) -> None:
        """Lazy-load entry point datasets on first access."""
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for ep in importlib.metadata.entry_points(group=self.ENTRY_POINT_GROUP):
            if ep.name in self._datasets:
                continue
            try:
                loader = ep.load()
            except Exception as exc:
                warnings.warn(
                    f"Failed to load dataset entry point '{ep.name}': {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue
            self._entry_point_loaders[ep.name] = loader
            self._datasets[ep.name] = {
                "source": "entry_point",
                "target": None,
                "task": "classification",
                "_metadata_resolved": False,
            }

    def _load_local_samples(self) -> None:
        """Auto-register local sample datasets."""
        if not self.sample_dir.exists():
            return
        for meta_path in self.sample_dir.glob("*/metadata.json"):
            name = meta_path.parent.name
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._datasets[name] = {
                "source": "local",
                "path": str(meta_path.parent),
                **meta,
            }

    def list(self) -> list[str]:
        """Return list of available dataset names."""
        self._ensure_entry_points_loaded()
        return sorted(self._datasets.keys())

    def info(self, name: str) -> dict[str, Any]:
        """Get metadata for a dataset."""
        self._ensure_entry_points_loaded()
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {self.list()}")
        return self._datasets[name]

    def load(self, name: str) -> dict[str, Any]:
        self._ensure_entry_points_loaded()
        info = self.info(name)
        if info["source"] == "entry_point":
            loader = self._entry_point_loaders.get(name)
            if loader is None:
                raise ValueError(f"Entry point loader for '{name}' is not available")
            if not info.get("_metadata_resolved", True):
                try:
                    sample = loader()
                except Exception as exc:
                    warnings.warn(
                        f"Failed to resolve metadata for dataset entry point '{name}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    raise
                info["target"] = sample.get("target")
                info["task"] = sample.get("metadata", {}).get("task", "classification")
                info["_metadata_resolved"] = True
                self._datasets[name] = info
                return sample
            return loader()
        if info["source"] == "local":
            return self._load_local(info)
        if info["source"] == "kaggle":
            from feature_forge.data.ingestion import KaggleFetcher

            fetcher = KaggleFetcher()
            return fetcher.load_with_metadata(info["slug"], info.get("target"))
        raise ValueError(f"Unknown source: {info['source']}")

    def _load_local(self, info: dict[str, Any]) -> dict[str, Any]:
        """Load a local sample dataset."""
        path = Path(info["path"])
        train_path = path / "train.csv"
        test_path = path / "test.csv"
        train_df = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
        test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
        return {
            "train": train_df,
            "test": test_df,
            "target": info.get("target"),
            "metadata": info,
        }

    def register(self, name: str, info: dict[str, Any]) -> None:
        """Register a new dataset."""
        if name in self._datasets:
            warnings.warn(
                f"Dataset '{name}' already registered. Overwriting.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._datasets[name] = info


_DEFAULT_REGISTRY: DatasetRegistry | None = None


def _get_default_registry() -> DatasetRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = DatasetRegistry()
    return _DEFAULT_REGISTRY


def titanic_loader() -> dict[str, Any]:
    """Load the titanic dataset via the standard registry mechanism."""
    return _get_default_registry().load("titanic")


def house_prices_loader() -> dict[str, Any]:
    """Load the house_prices dataset via the standard registry mechanism."""
    return _get_default_registry().load("house_prices")
