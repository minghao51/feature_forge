"""Tests for MetricRegistry with entry point discovery."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from feature_forge.evaluation.metrics import (
    METRIC_REGISTRY,
    MetricDirection,
    MetricRegistry,
    get_metric,
    get_metric_direction,
)
from feature_forge.exceptions import EvaluationError


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
        with pytest.raises(EvaluationError, match="Unknown metric"):
            MetricRegistry.get("nonexistent_metric")

    def test_register_adds_metric(self):
        MetricRegistry.register("test_metric", lambda y, p: 1.0)
        all_metrics = MetricRegistry.get_all()
        assert "test_metric" in all_metrics

    def test_register_and_reset(self):
        MetricRegistry.register("temp_metric", lambda y, p: 99.0)
        assert "temp_metric" in MetricRegistry.get_all()
        MetricRegistry.reset()
        assert "temp_metric" not in MetricRegistry.get_all()

    def test_clear_cache_and_refresh(self):
        MetricRegistry.get_all()
        first = MetricRegistry._discovered
        MetricRegistry.clear_cache()
        assert MetricRegistry._discovered is None
        MetricRegistry.refresh()
        assert MetricRegistry._discovered is not None
        assert MetricRegistry._discovered is not first

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


class TestMetricDirection:
    """Direction metadata for built-in and plugin metrics."""

    def test_maximize_metrics(self):
        for name in ("auc", "acc", "f1", "r2"):
            assert MetricRegistry.get_direction(name) is MetricDirection.MAXIMIZE, name

    def test_minimize_metrics(self):
        for name in ("rmse", "mae", "nrmse"):
            assert MetricRegistry.get_direction(name) is MetricDirection.MINIMIZE, name

    def test_get_metric_direction_module_helper(self):
        assert get_metric_direction("auc") is MetricDirection.MAXIMIZE
        assert get_metric_direction("rmse") is MetricDirection.MINIMIZE

    def test_register_with_direction_string(self):
        MetricRegistry.register("custom_max", lambda y, p: 1.0, direction="maximize")
        MetricRegistry.register("custom_min", lambda y, p: 1.0, direction="minimize")
        try:
            assert MetricRegistry.get_direction("custom_max") is MetricDirection.MAXIMIZE
            assert MetricRegistry.get_direction("custom_min") is MetricDirection.MINIMIZE
        finally:
            MetricRegistry.reset()

    def test_register_with_direction_enum(self):
        MetricRegistry.register("custom_enum", lambda y, p: 1.0, direction=MetricDirection.MINIMIZE)
        try:
            assert MetricRegistry.get_direction("custom_enum") is MetricDirection.MINIMIZE
        finally:
            MetricRegistry.reset()

    def test_direction_fails_closed_for_directionless_plugin(self):
        MetricRegistry.register("directionless", lambda y, p: 1.0)
        try:
            with pytest.raises(EvaluationError, match="no direction metadata"):
                MetricRegistry.get_direction("directionless")
        finally:
            MetricRegistry.reset()

    def test_direction_fails_closed_for_unknown_metric(self):
        with pytest.raises(EvaluationError, match="Unknown metric"):
            MetricRegistry.get_direction("does_not_exist")
