"""Tests for data ingestion — KaggleFetcher and DatasetRegistry edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from feature_forge.data.ingestion import KaggleFetcher
from feature_forge.exceptions import DatasetError


class TestKaggleFetcher:
    """Cover KaggleFetcher edge cases (lines 22-84)."""

    def _make_fetcher_for_cache(self, cache_dir) -> KaggleFetcher:
        """Create a fetcher that skips the kaggle import by patching."""
        fetcher = KaggleFetcher(cache_dir=str(cache_dir))
        # Simulate already-cached directory (no download needed)
        fetcher.cache_dir = cache_dir
        return fetcher

    def test_init_creates_cache_dir(self, tmp_path):
        cache = str(tmp_path / "kaggle_cache")
        KaggleFetcher(cache_dir=cache)
        assert tmp_path.joinpath("kaggle_cache").exists()

    def test_fetch_missing_kaggle_package(self):
        fetcher = KaggleFetcher(cache_dir="/tmp/test_cache")
        # kaggle is not installed in this venv, so this naturally hits ImportError
        with pytest.raises(DatasetError, match="kaggle package not installed"):
            fetcher.fetch("titanic")

    def test_fetch_no_csv_found(self, tmp_path):
        empty_dir = tmp_path / "empty_ds"
        empty_dir.mkdir()
        # Bypass kaggle import by patching it
        mock_kaggle_mod = MagicMock()
        with patch.dict("sys.modules", {"kaggle": mock_kaggle_mod}):
            fetcher = KaggleFetcher(cache_dir=str(tmp_path))
            with patch.object(fetcher, "cache_dir", empty_dir):
                with pytest.raises(DatasetError, match="No CSV files found"):
                    fetcher.fetch("titanic")

    def test_fetch_cache_hit(self, tmp_path):
        base_dir = tmp_path / "kaggle_base"
        base_dir.mkdir(parents=True)
        fetcher = KaggleFetcher(cache_dir=str(base_dir))
        ds_dir = base_dir / "titanic"
        ds_dir.mkdir()
        pd.DataFrame({"a": [1]}).to_csv(ds_dir / "train.csv", index=False)
        mock_kaggle_mod = MagicMock()
        with patch.dict("sys.modules", {"kaggle": mock_kaggle_mod}):
            result = fetcher.fetch("titanic")
            assert "train.csv" in result

    def test_fetch_cache_miss(self, tmp_path):
        base_dir = tmp_path / "kaggle_miss_base"
        fetcher = KaggleFetcher(cache_dir=str(base_dir))
        ds_dir = base_dir / "username_dataset"
        ds_dir.mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_csv(ds_dir / "data.csv", index=False)
        mock_kaggle_mod = MagicMock()
        with patch.dict("sys.modules", {"kaggle": mock_kaggle_mod}):
            result = fetcher.fetch("username/dataset")
            assert len(result) >= 1

    def test_fetch_file_filtering(self, tmp_path):
        base_dir = tmp_path / "kaggle_filter_base"
        fetcher = KaggleFetcher(cache_dir=str(base_dir))
        ds_dir = base_dir / "test_ds"
        ds_dir.mkdir(parents=True)
        pd.DataFrame({"a": [1]}).to_csv(ds_dir / "train.csv", index=False)
        pd.DataFrame({"b": [2]}).to_csv(ds_dir / "test.csv", index=False)
        mock_kaggle_mod = MagicMock()
        with patch.dict("sys.modules", {"kaggle": mock_kaggle_mod}):
            result = fetcher.fetch("test_ds", files=["train.csv"])
            assert "train.csv" in result
            assert "test.csv" not in result

    def test_load_with_metadata(self, tmp_path):
        base_dir = tmp_path / "kaggle_meta_base"
        fetcher = KaggleFetcher(cache_dir=str(base_dir))
        ds_dir = base_dir / "test_ds"
        ds_dir.mkdir(parents=True)
        pd.DataFrame({"a": [1], "y": [0]}).to_csv(ds_dir / "train.csv", index=False)
        mock_kaggle_mod = MagicMock()
        with patch.dict("sys.modules", {"kaggle": mock_kaggle_mod}):
            result = fetcher.load_with_metadata("test_ds", target_column="y")
            assert "train" in result
            assert result["target"] == "y"
            assert "metadata" in result
            assert result["metadata"]["dataset"] == "test_ds"
