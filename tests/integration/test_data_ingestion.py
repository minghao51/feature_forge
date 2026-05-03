"""Tests for data layer."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_forge.data import DatasetRegistry


class TestDatasetRegistry:
    def test_list_builtin(self):
        reg = DatasetRegistry()
        datasets = reg.list()
        assert "titanic" in datasets
        assert "house_prices" in datasets

    def test_info(self):
        reg = DatasetRegistry()
        info = reg.info("titanic")
        assert info["source"] == "kaggle"
        assert info["target"] == "Survived"

    def test_info_missing_raises(self):
        reg = DatasetRegistry()
        with pytest.raises(KeyError):
            reg.info("nonexistent")

    def test_register(self):
        reg = DatasetRegistry()
        reg.register("custom", {"source": "local", "path": "/tmp", "target": "y"})
        assert "custom" in reg.list()

    def test_load_local(self, tmp_path):
        # Create a temporary sample dataset
        sample_dir = tmp_path / "samples" / "dummy"
        sample_dir.mkdir(parents=True)
        train_df = pd.DataFrame({"a": [1, 2], "y": [0, 1]})
        train_df.to_csv(sample_dir / "train.csv", index=False)
        meta = {"target": "y", "task": "classification"}
        import json
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

        reg = DatasetRegistry(sample_dir=str(tmp_path / "samples"))
        result = reg.load("dummy")
        assert "train" in result
        assert list(result["train"].columns) == ["a", "y"]
