"""Tests for DatasetRegistry entry point discovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from feature_forge.data.registry import DatasetRegistry


class TestDatasetRegistryDiscovery:
    """Verify DatasetRegistry handles entry point discovery."""

    def test_builtin_datasets_available(self):
        registry = DatasetRegistry(sample_dir="/nonexistent")
        names = registry.list()
        assert "titanic" in names
        assert "house_prices" in names

    def test_register_adds_dataset(self):
        registry = DatasetRegistry(sample_dir="/nonexistent")
        registry.register("custom", {"source": "local", "target": "y"})
        assert "custom" in registry.list()

    def test_info_returns_metadata(self):
        registry = DatasetRegistry(sample_dir="/nonexistent")
        info = registry.info("titanic")
        assert info["target"] == "Survived"
        assert info["task"] == "classification"

    def test_info_raises_on_unknown(self):
        registry = DatasetRegistry(sample_dir="/nonexistent")
        with pytest.raises(KeyError):
            registry.info("nonexistent")

    def test_list_returns_sorted(self):
        registry = DatasetRegistry(sample_dir="/nonexistent")
        names = registry.list()
        assert names == sorted(names)

    @patch("importlib.metadata.entry_points")
    def test_entry_point_discovery_merges_with_builtin(self, mock_entry_points):
        mock_ep = MagicMock()
        mock_ep.name = "test_ep_dataset"

        def dummy_loader():
            return {"train": None, "test": None, "target": "y", "metadata": {}}

        mock_ep.load.return_value = dummy_loader
        mock_entry_points.return_value = [mock_ep]

        registry = DatasetRegistry(sample_dir="/nonexistent")
        names = registry.list()
        assert "test_ep_dataset" in names
        assert "titanic" in names  # builtin still present

    @patch("importlib.metadata.entry_points")
    def test_entry_point_does_not_override_builtin(self, mock_entry_points):
        mock_ep = MagicMock()
        mock_ep.name = "titanic"
        mock_ep.load.return_value = dict
        mock_entry_points.return_value = [mock_ep]

        registry = DatasetRegistry(sample_dir="/nonexistent")
        info = registry.info("titanic")
        # Built-in metadata preserved (not overridden by entry point)
        assert info["source"] == "kaggle"

    @patch("importlib.metadata.entry_points")
    def test_entry_point_discovery_empty(self, mock_entry_points):
        mock_entry_points.return_value = []
        registry = DatasetRegistry(sample_dir="/nonexistent")
        names = registry.list()
        assert len(names) == 2  # only built-in datasets
