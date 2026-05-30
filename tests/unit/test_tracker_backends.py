"""Contract tests for experiment tracker backends.

Tests WandBTracker and MLflowTracker for:
- Import error handling when SDKs not installed
- Contract compliance with ExperimentTracker ABC
- Type dispatch in _log_artifact_item
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from feature_forge.exceptions import TrackingError
from feature_forge.experiment.mlflow_backend import MLflowTracker
from feature_forge.experiment.tracker import ExperimentTracker, NoOpTracker
from feature_forge.experiment.wandb_backend import WandBTracker

pytestmark = pytest.mark.contract


class TestTrackerContract:
    def test_wandb_implements_tracker(self):
        assert issubclass(WandBTracker, ExperimentTracker)

    def test_mlflow_implements_tracker(self):
        assert issubclass(MLflowTracker, ExperimentTracker)

    def test_noop_implements_tracker(self):
        assert issubclass(NoOpTracker, ExperimentTracker)

    def test_tracker_has_required_methods(self):
        required = {"init_run", "log_metrics", "log_params", "log_artifact", "finish"}
        for cls in (WandBTracker, MLflowTracker, NoOpTracker):
            instance = cls(project="test")
            for method in required:
                assert hasattr(instance, method), f"{cls.__name__} missing {method}"


class TestWandBTrackerInit:
    def test_init_sets_defaults(self):
        tracker = WandBTracker(project="test")
        assert tracker.project == "test"
        assert tracker.log_code_to_artifact is True
        assert tracker.log_code_to_table is True
        assert tracker._run is None

    def test_init_custom_flags(self):
        tracker = WandBTracker(project="test", log_code_to_artifact=False, log_code_to_table=False)
        assert not tracker.log_code_to_artifact
        assert not tracker.log_code_to_table

    def test_init_run_no_wandb_raises(self):
        tracker = WandBTracker(project="test")
        with patch.dict("sys.modules", {"wandb": None}):
            with pytest.raises(TrackingError, match="wandb not installed"):
                tracker.init_run("run1", {"x": 1})

    def test_log_metrics_no_run_no_error(self):
        tracker = WandBTracker(project="test")
        tracker.log_metrics({"acc": 0.9})

    def test_log_params_no_run_no_error(self):
        tracker = WandBTracker(project="test")
        tracker.log_params({"lr": 0.01})

    def test_log_artifact_no_run_no_error(self):
        tracker = WandBTracker(project="test")
        tracker.log_artifact("path/to/file")

    def test_finish_no_run_no_error(self):
        tracker = WandBTracker(project="test")
        tracker.finish()


class TestMLflowTrackerInit:
    def test_init_sets_defaults(self):
        tracker = MLflowTracker(project="test")
        assert tracker.project == "test"
        assert tracker._run_id is None

    def test_init_with_tracking_uri(self):
        tracker = MLflowTracker(project="test", tracking_uri="http://localhost:5000")
        assert tracker.tracking_uri == "http://localhost:5000"

    def test_init_run_no_mlflow_raises(self):
        tracker = MLflowTracker(project="test")
        with patch.dict("sys.modules", {"mlflow": None}):
            with pytest.raises(TrackingError, match="mlflow not installed"):
                tracker.init_run("run1", {"x": 1})

    def test_log_metrics_no_mlflow_no_error(self):
        tracker = MLflowTracker(project="test")
        with patch.dict("sys.modules", {"mlflow": None}):
            tracker.log_metrics({"acc": 0.9})

    def test_log_params_no_mlflow_no_error(self):
        tracker = MLflowTracker(project="test")
        with patch.dict("sys.modules", {"mlflow": None}):
            tracker.log_params({"lr": 0.01})

    def test_log_artifact_no_mlflow_no_error(self):
        tracker = MLflowTracker(project="test")
        with patch.dict("sys.modules", {"mlflow": None}):
            tracker.log_artifact("path/to/file")

    def test_finish_no_mlflow_no_error(self):
        tracker = MLflowTracker(project="test")
        with patch.dict("sys.modules", {"mlflow": None}):
            tracker.finish()


class TestLogArtifactItem:
    """Cover _log_artifact_item type dispatch in the base ExperimentTracker."""

    def test_none_logs_as_param(self):
        tracker = NoOpTracker(project="test")
        with patch.object(tracker, "log_params") as mock_log:
            tracker._log_artifact_item("key", None)
            mock_log.assert_called_once_with({"key": "None"})

    def test_bool_logs_as_string_param(self):
        tracker = NoOpTracker(project="test")
        with patch.object(tracker, "log_params") as mock_log:
            tracker._log_artifact_item("flag", True)
            mock_log.assert_called_once_with({"flag": "True"})

    def test_int_metric(self):
        tracker = NoOpTracker(project="test")
        with patch.object(tracker, "log_metrics") as mock_log:
            tracker._log_artifact_item("count", 42)
            mock_log.assert_called_once_with({"count": 42.0})

    def test_float_metric(self):
        tracker = NoOpTracker(project="test")
        with patch.object(tracker, "log_metrics") as mock_log:
            tracker._log_artifact_item("score", 0.95)
            mock_log.assert_called_once_with({"score": 0.95})

    def test_dataframe_delegates(self):
        tracker = NoOpTracker(project="test")
        df = pd.DataFrame({"x": [1, 2]})
        with patch.object(tracker, "_log_dataframe") as mock_log:
            tracker._log_artifact_item("df", df)
            mock_log.assert_called_once_with("df", df)

    def test_code_string_logs_as_code(self):
        tracker = NoOpTracker(project="test")
        with patch.object(tracker, "_log_code") as mock_log:
            tracker._log_artifact_item("script", "def foo(): pass")
            mock_log.assert_called_once_with("script", "def foo(): pass")

    def test_code_with_code_in_key(self):
        tracker = NoOpTracker(project="test")
        with patch.object(tracker, "_log_code") as mock_log:
            tracker._log_artifact_item("generated_code", "import pandas")
            mock_log.assert_called_once_with("generated_code", "import pandas")

    def test_list_logs_as_json_param(self):
        tracker = NoOpTracker(project="test")
        with patch.object(tracker, "log_params") as mock_log:
            tracker._log_artifact_item("items", [1, 2, 3])
            call_key = mock_log.call_args[0][0]
            assert "items" in call_key

    def test_dict_logs_as_json_param(self):
        tracker = NoOpTracker(project="test")
        with patch.object(tracker, "log_params") as mock_log:
            tracker._log_artifact_item("config", {"a": 1})
            call_key = mock_log.call_args[0][0]
            assert "config" in call_key

    def test_string_logs_as_param(self):
        tracker = NoOpTracker(project="test")
        with patch.object(tracker, "log_params") as mock_log:
            tracker._log_artifact_item("name", "hello")
            mock_log.assert_called_once_with({"name": "hello"})

    def test_lazy_ref_with_load(self):
        tracker = NoOpTracker(project="test")

        class FakeLazyRef:
            def load(self):
                return "loaded_string"

        with patch.object(tracker, "_log_dataframe") as mock_log:
            tracker._log_artifact_item("ref", FakeLazyRef())
            mock_log.assert_called_once_with("ref", "loaded_string")

    def test_lazy_ref_load_fails(self):
        tracker = NoOpTracker(project="test")

        class BrokenLazyRef:
            def load(self):
                raise ValueError("broken")

        with patch.object(tracker, "log_params") as mock_log:
            tracker._log_artifact_item("broken", BrokenLazyRef())
            mock_log.assert_called_once()
