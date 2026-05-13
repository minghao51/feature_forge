"""Integration tests for plugin discovery via entry points."""

from __future__ import annotations

from feature_forge.baselines.base import BaselineRegistry
from feature_forge.data.registry import DatasetRegistry
from feature_forge.evaluation.metrics import MetricRegistry
from feature_forge.evaluation.model_factory import ModelRegistry


class TestPluginDiscovery:
    """Verify that built-in entry points are discoverable."""

    def test_baseline_entry_points_discovered(self):
        baselines = BaselineRegistry.get_all_baselines()
        assert "openfe" in baselines
        assert "caafe" in baselines
        assert "llmfe" in baselines
        assert "malmus" in baselines

    def test_metric_entry_points_discovered(self):
        metrics = MetricRegistry.get_all()
        assert "auc" in metrics
        assert "acc" in metrics
        assert "f1" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "nrmse" in metrics

    def test_model_entry_points_discovered(self):
        models = ModelRegistry.get_all()
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "catboost" in models
        assert "random_forest" in models
        assert "mlp" in models

    def test_dataset_entry_points_discovered(self):
        registry = DatasetRegistry(sample_dir="/nonexistent")
        datasets = registry.list()
        assert "titanic" in datasets
        assert "house_prices" in datasets

    def test_baseline_registry_caches_discovery(self):
        BaselineRegistry._discovered = None  # reset cache
        BaselineRegistry.get_all_baselines()
        first = BaselineRegistry._discovered
        BaselineRegistry.get_all_baselines()
        second = BaselineRegistry._discovered
        assert first is second  # cached discovered reference

    def test_metric_registry_caches_discovery(self):
        MetricRegistry._discovered = None  # reset cache
        MetricRegistry.get_all()
        first = MetricRegistry._discovered
        MetricRegistry.get_all()
        second = MetricRegistry._discovered
        assert first is second  # cached discovered reference

    def test_model_registry_caches_discovery(self):
        ModelRegistry._discovered = None  # reset cache
        ModelRegistry.get_all()
        first = ModelRegistry._discovered
        ModelRegistry.get_all()
        second = ModelRegistry._discovered
        assert first is second  # cached discovered reference
