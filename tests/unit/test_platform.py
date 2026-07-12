"""Tests for ExperimentalPlatform."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from feature_forge import ExperimentalPlatform
from feature_forge.config import Settings
from feature_forge.methods import BaseMethod


class DummyBaseline(BaseMethod):
    """A minimal baseline for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="dummy")
        self._mode = kwargs.get("mode")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> DummyBaseline:
        self.feature_names = list(X_train.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()

    def get_artifacts(self) -> dict[str, Any]:
        return {"generated_code": "x = 1"}

    @property
    def generated_scripts(self) -> list[str]:
        return ["x = 1"]


@pytest.fixture
def sample_X() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def sample_y() -> pd.Series:
    return pd.Series([0, 1, 0], name="target")


class TestExperimentalPlatform:
    """Verify ExperimentalPlatform initialization, registration, listing, and run."""

    def test_init_no_config(self):
        platform = ExperimentalPlatform()
        assert platform._config is None

    def test_init_with_dict_config(self):
        platform = ExperimentalPlatform(config={"task": "regression", "metric": "rmse"})
        assert isinstance(platform._config, dict)

    def test_init_with_settings(self):
        settings = Settings(task="regression")
        platform = ExperimentalPlatform(config=settings)
        assert platform._config is settings

    def test_list_methods(self):
        platform = ExperimentalPlatform()
        names = platform.list_methods()
        assert "openfe" in names
        assert "caafe" in names
        assert "llmfe" in names
        assert "malmus" in names

    def test_list_datasets(self):
        platform = ExperimentalPlatform()
        names = platform.list_datasets()
        assert "titanic" in names
        assert "house_prices" in names

    def test_list_models(self):
        platform = ExperimentalPlatform()
        names = platform.list_models()
        assert "xgboost" in names

    def test_list_metrics(self):
        platform = ExperimentalPlatform()
        names = platform.list_metrics()
        assert "auc" in names

    def test_register_method(self):
        platform = ExperimentalPlatform()
        platform.register_method("dummy", DummyBaseline)
        names = platform.list_methods()
        assert "dummy" in names

    def test_register_dataset(self):
        platform = ExperimentalPlatform()
        platform.register_dataset("custom_ds", {"source": "local", "target": "y"})
        names = platform.list_datasets()
        assert "custom_ds" in names

    def test_register_model(self):
        platform = ExperimentalPlatform()
        platform.register_model("custom_model", lambda task, rs: None)
        names = platform.list_models()
        assert "custom_model" in names

    def test_register_metric(self):
        platform = ExperimentalPlatform()
        platform.register_metric("custom_metric", lambda y, p: 1.0)
        names = platform.list_metrics()
        assert "custom_metric" in names

    def test_to_dataframe(self):
        results = [
            {"dataset": "d1", "method": "b1", "cv_score": 0.9},
            {"dataset": "d1", "method": "b2", "cv_score": 0.8},
        ]
        df = ExperimentalPlatform.to_dataframe(results)
        assert len(df) == 2
        assert list(df.columns) == ["dataset", "method", "cv_score"]

    def test_report_empty(self):
        platform = ExperimentalPlatform()
        report = platform.report([])
        assert "No results" in report

    def test_report_best(self):
        results = [
            {"dataset": "d1", "method": "b1", "cv_score": 0.9},
            {"dataset": "d1", "method": "b2", "cv_score": 0.8},
        ]
        platform = ExperimentalPlatform()
        best = platform.report_best(results)
        assert len(best) == 1
        assert best.iloc[0]["method"] == "b1"

    @patch("feature_forge.platform.ExperimentCaseExecutor")
    def test_run_basic(self, mock_executor_cls, sample_X, sample_y):
        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor
        mock_executor.execute.return_value = MagicMock(
            dataset="titanic",
            method="dummy",
            model="xgboost",
            seed=42,
            cv_score=0.9,
            gain=0.1,
            baseline_score=0.8,
            num_features_generated=1,
            error=None,
        )
        platform = ExperimentalPlatform()
        platform._dataset_registry = MagicMock()
        platform._dataset_registry.list.return_value = ["titanic"]
        platform._dataset_registry.info.return_value = {
            "source": "local",
            "target": "Survived",
            "task": "classification",
        }
        platform._dataset_registry.load.return_value = {
            "train": sample_X,
            "target": "target",
            "test": pd.DataFrame(),
            "metadata": {"task": "classification"},
        }

        sample_X["target"] = sample_y
        results = platform.run(datasets=["titanic"], methods=["dummy"])
        assert len(results) == 1
        assert results[0]["method"] == "dummy"

    @patch("feature_forge.platform.MethodRegistry")
    def test_run_with_extra_method(self, mock_registry):
        mock_registry.get_all_methods.return_value = {}

        platform = ExperimentalPlatform()
        platform.register_method("custom", DummyBaseline)

        platform._dataset_registry = MagicMock()
        platform._dataset_registry.list.return_value = []
        platform._dataset_registry.info.return_value = {
            "source": "local",
            "target": "y",
            "task": "classification",
        }
        platform._dataset_registry.load.return_value = {
            "train": pd.DataFrame({"a": [1, 2], "b": [3, 4], "y": [0, 1]}),
            "target": "y",
            "test": pd.DataFrame(),
            "metadata": {},
        }

        names = platform.list_methods()
        assert "custom" in names

    @patch("feature_forge.platform.ExperimentCaseExecutor")
    @patch("feature_forge.platform.ProcessPoolExecutionAdapter")
    def test_run_parallel_uses_pool_backend(self, mock_pool_cls, mock_executor_cls):
        mock_executor_cls.return_value = MagicMock()
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_pool.run.return_value = [
            MagicMock(
                dataset="demo",
                method="dummy",
                model="xgboost",
                seed=42,
                cv_score=0.7,
                gain=0.1,
                baseline_score=0.6,
                num_features_generated=1,
                error=None,
            )
        ]
        platform = ExperimentalPlatform()
        results = platform.run(
            datasets=["demo"],
            methods=["dummy"],
            parallel=True,
            progress=False,
        )
        assert len(results) == 1
        assert results[0]["dataset"] == "demo"

    def test_run_parallel_rejects_extra_methods(self):
        platform = ExperimentalPlatform()
        platform.register_method("custom", DummyBaseline)
        with pytest.raises(ValueError, match="parallel=True currently supports only"):
            platform.run(
                datasets=["demo"],
                methods=["custom"],
                parallel=True,
                progress=False,
            )

    @patch("feature_forge.platform.ExperimentCaseExecutor")
    def test_run_uses_config_tracker_when_no_override(self, mock_executor_cls):
        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor
        mock_executor.execute.return_value = MagicMock(
            dataset="demo",
            method="dummy",
            model="xgboost",
            seed=42,
            cv_score=0.7,
            gain=0.1,
            baseline_score=0.6,
            num_features_generated=1,
            error=None,
        )
        from feature_forge.experiment import NoOpTracker

        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
        platform.run(datasets=["demo"], methods=["dummy"], progress=False)
        kwargs = mock_executor_cls.call_args.kwargs
        assert isinstance(kwargs["tracker"], NoOpTracker)

    @patch("feature_forge.platform.ExperimentCaseExecutor")
    def test_run_explicit_tracker_overrides_config(self, mock_executor_cls):
        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor
        mock_executor.execute.return_value = MagicMock(
            dataset="demo",
            method="dummy",
            model="xgboost",
            seed=42,
            cv_score=0.7,
            gain=0.1,
            baseline_score=0.6,
            num_features_generated=1,
            error=None,
        )
        from feature_forge.experiment import NoOpTracker

        explicit = NoOpTracker(project="override")
        platform = ExperimentalPlatform(config={"tracker": {"backend": "wandb"}})
        platform.run(datasets=["demo"], methods=["dummy"], tracker=explicit, progress=False)
        kwargs = mock_executor_cls.call_args.kwargs
        assert kwargs["tracker"] is explicit
