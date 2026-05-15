"""Integration tests for plugin discovery via entry points."""

from __future__ import annotations

from feature_forge.data.registry import DatasetRegistry
from feature_forge.evaluation.metrics import MetricRegistry
from feature_forge.evaluation.model_factory import ModelRegistry
from feature_forge.methods.base import MethodRegistry


class TestPluginDiscovery:
    """Verify that built-in entry points are discoverable."""

    def test_method_entry_points_discovered(self):
        methods = MethodRegistry.get_all_methods()
        assert "malmas" in methods
        assert "openfe" in methods
        assert "caafe" in methods
        assert "llmfe" in methods
        assert "malmus" in methods

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

    def test_method_registry_caches_discovery(self):
        MethodRegistry._discovered = None  # reset cache
        MethodRegistry.get_all_methods()
        first = MethodRegistry._discovered
        MethodRegistry.get_all_methods()
        second = MethodRegistry._discovered
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
