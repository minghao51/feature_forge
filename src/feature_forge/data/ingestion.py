"""Data ingestion utilities.

Supports Kaggle dataset downloads and local sample datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from feature_forge.exceptions import DatasetError


class KaggleFetcher:
    """Fetch datasets from Kaggle.

    Requires kaggle.json credentials or KAGGLE_USERNAME/KAGGLE_KEY env vars.
    """

    def __init__(self, cache_dir: str = "data/kaggle") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, dataset_slug: str, files: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Download and load a Kaggle dataset.

        Args:
            dataset_slug: Kaggle dataset identifier, e.g. 'titanic' or 'username/dataset'.
            files: Specific files to load (None = load all CSVs).

        Returns:
            Dictionary mapping filename to DataFrame.
        """
        try:
            import kaggle
        except ImportError as exc:
            raise DatasetError("kaggle package not installed. Run: uv pip install kaggle") from exc

        dataset_path = self.cache_dir / dataset_slug.replace("/", "_")
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Download if not cached
        if not any(dataset_path.iterdir()):
            kaggle.api.dataset_download_files(dataset_slug, path=str(dataset_path), unzip=True)

        # Load CSVs
        result: dict[str, pd.DataFrame] = {}
        for csv_path in dataset_path.glob("*.csv"):
            if files is not None and csv_path.name not in files:
                continue
            result[csv_path.name] = pd.read_csv(csv_path)

        if not result:
            raise DatasetError(f"No CSV files found for dataset {dataset_slug}")

        return result

    def load_with_metadata(
        self, dataset_slug: str, target_column: str | None = None
    ) -> dict[str, Any]:
        """Load dataset with metadata.

        Returns:
            Dict with keys: train, test, target, metadata.
        """
        data = self.fetch(dataset_slug)
        train_df = data.get("train.csv") or next(iter(data.values()))
        test_df = data.get("test.csv", pd.DataFrame())

        metadata = {
            "dataset": dataset_slug,
            "num_rows": len(train_df),
            "num_features": len(train_df.columns),
            "target": target_column,
        }

        return {
            "train": train_df,
            "test": test_df,
            "target": target_column,
            "metadata": metadata,
        }
