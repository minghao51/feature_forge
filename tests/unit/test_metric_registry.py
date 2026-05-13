"""Tests for MetricRegistry with entry point discovery."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from feature_forge.evaluation.metrics import METRIC_REGISTRY, MetricRegistry, get_metric


class TestMetricRegistry:
    """Verify MetricRegistry built-in + entry point discovery."""

    def test_get_builtin_returns_all_metrics(self):
        builtin = MetricRegistry.get_builtin()
        assert "auc" in builtin
        assert "acc" in builtin
        assert "f1" in builtin
        assert "rmse" in builtin
        assert "mae" in builtin
        assert "r2" in builtin
        assert "nrmse" in builtin
        assert len(builtin) >= 7

    def test_get_all_includes_builtin(self):
        all_metrics = MetricRegistry.get_all()
        for name in METRIC_REGISTRY:
            assert name in all_metrics

    def test_get_returns_builtin_metric(self):
        fn = MetricRegistry.get("auc")
        assert callable(fn)

    def test_get_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricRegistry.get("nonexistent_metric")

    def test_register_adds_metric(self):
        MetricRegistry.register("test_metric", lambda y, p: 1.0)
        all_metrics = MetricRegistry.get_all()
        assert "test_metric" in all_metrics

    def test_get_metric_delegates_to_registry(self):
        fn = get_metric("auc")
        assert callable(fn)

    def test_metrics_are_callable(self):
        import numpy as np

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        fn = MetricRegistry.get("auc")
        score = fn(y_true, y_pred)
        assert isinstance(score, float)

    @patch("importlib.metadata.entry_points")
    def test_entry_point_discovery(self, mock_entry_points):
        mock_entry_points.return_value.select.return_value = []

        discovered = MetricRegistry.discover()
        assert isinstance(discovered, dict)
