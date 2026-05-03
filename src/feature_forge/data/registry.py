"""Dataset registry for managing available datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd


class DatasetRegistry:
    """Registry of available datasets with loaders.

    Supports both Kaggle datasets and local sample datasets.
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

    def __init__(self, sample_dir: str = "data/samples") -> None:
        self.sample_dir = Path(sample_dir)
        self._datasets: dict[str, dict[str, Any]] = dict(self.SAMPLE_DATASETS)
        self._load_local_samples()

    def _load_local_samples(self) -> None:
        """Auto-register local sample datasets."""
        if not self.sample_dir.exists():
            return
        for meta_path in self.sample_dir.glob("*/metadata.json"):
            name = meta_path.parent.name
            with open(meta_path, encoding="utf-8") as f:
                meta = __import__("json").load(f)
            self._datasets[name] = {
                "source": "local",
                "path": str(meta_path.parent),
                **meta,
            }

    def list(self) -> list[str]:
        """Return list of available dataset names."""
        return list(self._datasets.keys())

    def info(self, name: str) -> dict[str, Any]:
        """Get metadata for a dataset."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found. Available: {self.list()}")
        return self._datasets[name]

    def load(self, name: str) -> dict[str, Any]:
        """Load a dataset by name.

        Returns:
            Dict with keys: train, test, target, metadata.
        """
        info = self.info(name)
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
        self._datasets[name] = info
