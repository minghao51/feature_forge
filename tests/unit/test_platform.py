"""Tests for ExperimentalPlatform."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from feature_forge import ExperimentalPlatform
from feature_forge.config import Settings
from feature_forge.contracts import (
    CaseExecutionPlan,
    Layer,
    ManifestRef,
    StageDisposition,
    StageExecution,
)
from feature_forge.experiment.execution import ExperimentResult
from feature_forge.experiment.hamilton_executor import run_hamilton_case
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


class TestExperimentalPlatform:
    """Verify ExperimentalPlatform initialization, registration, listing, and run."""

    def test_init_no_config(self) -> None:
        platform = ExperimentalPlatform()
        assert platform._config is None

    def test_init_with_dict_config(self) -> None:
        platform = ExperimentalPlatform(config={"task": "regression", "metric": "rmse"})
        assert isinstance(platform._config, dict)

    def test_init_with_settings(self) -> None:
        settings = Settings(task="regression")
        platform = ExperimentalPlatform(config=settings)
        assert platform._config is settings

    def test_list_methods(self) -> None:
        platform = ExperimentalPlatform()
        names = platform.list_methods()
        assert "openfe" in names
        assert "caafe" in names
        assert "llmfe" in names
        assert "malmus" in names

    def test_list_datasets(self) -> None:
        platform = ExperimentalPlatform()
        names = platform.list_datasets()
        assert "titanic" in names
        assert "house_prices" in names

    def test_list_models(self) -> None:
        platform = ExperimentalPlatform()
        names = platform.list_models()
        assert "xgboost" in names

    def test_list_metrics(self) -> None:
        platform = ExperimentalPlatform()
        names = platform.list_metrics()
        assert "auc" in names

    def test_register_method(self) -> None:
        platform = ExperimentalPlatform()
        platform.register_method("dummy", DummyBaseline)
        names = platform.list_methods()
        assert "dummy" in names

    def test_register_dataset(self) -> None:
        platform = ExperimentalPlatform()
        platform.register_dataset("custom_ds", {"source": "local", "target": "y"})
        names = platform.list_datasets()
        assert "custom_ds" in names

    def test_register_model(self) -> None:
        platform = ExperimentalPlatform()
        platform.register_model("custom_model", lambda task, rs: None)
        names = platform.list_models()
        assert "custom_model" in names

    def test_register_metric(self) -> None:
        platform = ExperimentalPlatform()
        platform.register_metric("custom_metric", lambda y, p: 1.0)
        names = platform.list_metrics()
        assert "custom_metric" in names

    def test_to_dataframe(self) -> None:
        results = [
            {"dataset": "d1", "method": "b1", "cv_score": 0.9},
            {"dataset": "d1", "method": "b2", "cv_score": 0.8},
        ]
        df = ExperimentalPlatform.to_dataframe(results)
        assert len(df) == 2
        assert list(df.columns) == ["dataset", "method", "cv_score"]

    def test_report_empty(self) -> None:
        platform = ExperimentalPlatform()
        report = platform.report([])
        assert "No results" in report

    def test_report_best(self) -> None:
        results = [
            {"dataset": "d1", "method": "b1", "cv_score": 0.9},
            {"dataset": "d1", "method": "b2", "cv_score": 0.8},
        ]
        platform = ExperimentalPlatform()
        best = platform.report_best(results)
        assert len(best) == 1
        assert best.iloc[0]["method"] == "b1"

    @patch("feature_forge.platform.HamiltonLayerExecutor")
    def test_run_basic(self, mock_hamilton_cls: MagicMock) -> None:
        mock_executor = MagicMock()
        mock_hamilton_cls.return_value = mock_executor
        mock_executor.plan_case.return_value = CaseExecutionPlan(
            case_key="casekey",
            attempt_id="attempt-1",
            dataset="titanic",
            method="dummy",
            model="xgboost",
            unresolved_layers=list(Layer),
        )
        mock_executor.execute_case.return_value = ExperimentResult(
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
        results = platform.run(datasets=["titanic"], methods=["dummy"], progress=False)
        assert len(results) == 1
        assert results[0]["method"] == "dummy"

    @patch("feature_forge.platform.MethodRegistry")
    def test_run_with_extra_method(self, mock_registry: MagicMock) -> None:
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

    def test_run_rejects_stale_legacy_config(self) -> None:
        """Stale engine=legacy fails at Settings construction with migration guidance."""
        platform = ExperimentalPlatform(config={"dataflow": {"engine": "legacy"}})
        with pytest.raises(ValidationError, match="FF_DATAFLOW__ENGINE") as excinfo:
            platform.run(datasets=["demo"], methods=["dummy"], progress=False)
        message = str(excinfo.value)
        assert "legacy" in message
        assert "FF_DATAFLOW__ENGINE" in message

    @patch("feature_forge.platform.HamiltonLayerExecutor")
    @patch("feature_forge.platform.create_tracker_from_config")
    def test_run_uses_config_tracker_when_no_override(
        self, mock_create_tracker: MagicMock, mock_hamilton_cls: MagicMock
    ) -> None:
        mock_executor = MagicMock()
        mock_hamilton_cls.return_value = mock_executor
        mock_executor.plan_case.return_value = CaseExecutionPlan(
            case_key="casekey",
            attempt_id="attempt-1",
            dataset="demo",
            method="dummy",
            model="xgboost",
            unresolved_layers=list(Layer),
        )
        mock_executor.execute_case.return_value = ExperimentResult(
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
        config_tracker = MagicMock()
        mock_create_tracker.return_value = config_tracker

        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
        platform.run(datasets=["demo"], methods=["dummy"], progress=False)

        mock_create_tracker.assert_called_once()
        # The parent owns tracker effects on the Hamilton path.
        config_tracker.init_run.assert_called_once()
        config_tracker.log_metrics.assert_called_once()
        config_tracker.finish.assert_called_once()

    @patch("feature_forge.platform.HamiltonLayerExecutor")
    @patch("feature_forge.platform.create_tracker_from_config")
    def test_run_explicit_tracker_overrides_config(
        self, mock_create_tracker: MagicMock, mock_hamilton_cls: MagicMock
    ) -> None:
        mock_executor = MagicMock()
        mock_hamilton_cls.return_value = mock_executor
        mock_executor.plan_case.return_value = CaseExecutionPlan(
            case_key="casekey",
            attempt_id="attempt-1",
            dataset="demo",
            method="dummy",
            model="xgboost",
            unresolved_layers=list(Layer),
        )
        mock_executor.execute_case.return_value = ExperimentResult(
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
        explicit = MagicMock()
        platform = ExperimentalPlatform(config={"tracker": {"backend": "wandb"}})
        platform.run(datasets=["demo"], methods=["dummy"], tracker=explicit, progress=False)

        # The explicit tracker is used and the config-built tracker is never created.
        mock_create_tracker.assert_not_called()
        assert explicit.init_run.call_count == 1
        assert explicit.log_metrics.call_count == 1
        assert explicit.finish.call_count == 1

    @patch("feature_forge.platform.HamiltonLayerExecutor")
    def test_run_accepts_policy_and_token_kwargs(self, mock_hamilton_cls: MagicMock) -> None:
        """run() plumbing accepts the ADR 0017 kwargs without changing defaults."""
        from feature_forge.config import FailurePolicy
        from feature_forge.experiment.execution import CancellationToken

        mock_executor = MagicMock()
        mock_hamilton_cls.return_value = mock_executor
        mock_executor.plan_case.return_value = CaseExecutionPlan(
            case_key="casekey",
            attempt_id="attempt-1",
            dataset="demo",
            method="dummy",
            model="xgboost",
            unresolved_layers=list(Layer),
        )
        mock_executor.execute_case.return_value = ExperimentResult(
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

        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
        results = platform.run(
            datasets=["demo"],
            methods=["dummy"],
            progress=False,
            failure_policy=FailurePolicy.CONTINUE,
            cancellation_token=CancellationToken(),
        )

        assert len(results) == 1
        assert results[0]["state"] == "succeeded"
        mock_executor.execute_case.assert_called_once()


class TestExperimentalPlatformHamiltonDefault:
    """Hamilton is the sole execution engine (ADR 0016); stale legacy settings are rejected."""

    def _patch_executor(self, mock_hamilton_cls: MagicMock) -> MagicMock:
        mock_executor = MagicMock()
        mock_hamilton_cls.return_value = mock_executor
        mock_executor.plan_case.return_value = CaseExecutionPlan(
            case_key="casekey",
            attempt_id="attempt-1",
            dataset="titanic",
            method="dummy",
            model="xgboost",
            unresolved_layers=list(Layer),
        )
        return mock_executor

    def _staged_result(self) -> ExperimentResult:
        manifest_ref = ManifestRef(layer=Layer.BRONZE, run_id="attempt-1", sha256="a" * 64)
        stage = StageExecution(
            layer=Layer.BRONZE,
            disposition=StageDisposition.EXECUTED,
            manifest_ref=manifest_ref,
            reuse_fingerprint="reuse",
            layer_fingerprint="layerfp",
        )
        return ExperimentResult(
            dataset="titanic",
            method="dummy",
            model="xgboost",
            seed=42,
            cv_score=0.9,
            gain=0.1,
            baseline_score=0.8,
            num_features_generated=1,
            run_id="attempt-1",
            case_fingerprint="casekey",
            stages=[stage],
        )

    @patch("feature_forge.platform.HamiltonLayerExecutor")
    def test_run_default_dispatches_hamilton_sequential(self, mock_hamilton_cls: MagicMock) -> None:
        mock_executor = self._patch_executor(mock_hamilton_cls)
        mock_executor.execute_case.return_value = self._staged_result()

        platform = ExperimentalPlatform()
        results = platform.run(datasets=["titanic"], methods=["dummy"], progress=False)

        mock_hamilton_cls.assert_called_once()
        mock_executor.plan_case.assert_called_once()
        mock_executor.execute_case.assert_called_once()
        assert len(results) == 1
        assert results[0]["method"] == "dummy"
        # Nested StageExecution serializes to a plain dict via dataclasses.asdict.
        assert results[0]["stages"] == [self._staged_result().stages[0].model_dump()]
        assert results[0]["stages"][0]["layer"] == "bronze"
        assert results[0]["stages"][0]["disposition"] == "executed"

    @patch("feature_forge.platform.ProcessPoolExecutionAdapter")
    @patch("feature_forge.platform.HamiltonLayerExecutor")
    def test_run_default_dispatches_hamilton_process(
        self, mock_hamilton_cls: MagicMock, mock_pool_cls: MagicMock
    ) -> None:
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_executor = MagicMock()
        mock_hamilton_cls.return_value = mock_executor
        mock_executor.plan_case.return_value = CaseExecutionPlan(
            case_key="casekey2",
            attempt_id="attempt-2",
            dataset="demo",
            method="dummy",
            model="xgboost",
            unresolved_layers=list(Layer),
        )
        mock_pool.run.return_value = [
            ExperimentResult(
                dataset="demo",
                method="dummy",
                model="xgboost",
                seed=42,
                cv_score=0.7,
                gain=0.1,
                baseline_score=0.6,
                num_features_generated=1,
                run_id="attempt-2",
                case_fingerprint="casekey2",
            )
        ]

        platform = ExperimentalPlatform()
        results = platform.run(datasets=["demo"], methods=["dummy"], parallel=True, progress=False)

        # The worker is the Hamilton process entry point and payloads are serializable.
        call_args = mock_pool.run.call_args
        payloads = call_args.args[0]
        worker = call_args.args[1]
        assert worker is run_hamilton_case
        assert len(payloads) == 1
        assert isinstance(payloads[0].settings_data, dict)
        assert payloads[0].case.case_key is not None
        assert payloads[0].case.attempt_id is not None
        assert len(results) == 1
        assert results[0]["method"] == "dummy"

    @patch("feature_forge.platform.HamiltonCacheManager")
    def test_cache_retention_runs_once_after_hamilton_workers(
        self, manager_class: MagicMock
    ) -> None:
        settings = Settings(
            dataflow={
                "artifact_root": "experiments/test-artifacts",
                "cache": {"max_age_days": 14, "max_size_mb": 512},
            }
        )

        ExperimentalPlatform._maintain_hamilton_cache(settings)

        manager_class.assert_called_once()
        manager_class.return_value.collect.assert_called_once_with(
            max_age_days=14,
            max_size_mb=512,
        )

    def test_run_rejects_stale_legacy_config(self) -> None:
        """Stale engine=legacy fails at Settings construction with migration guidance."""
        platform = ExperimentalPlatform(config={"dataflow": {"engine": "legacy"}})
        with pytest.raises(ValidationError, match="FF_DATAFLOW__ENGINE") as excinfo:
            platform.run(datasets=["demo"], methods=["dummy"], progress=False)
        message = str(excinfo.value)
        assert "legacy" in message
        assert "FF_DATAFLOW__ENGINE" in message
