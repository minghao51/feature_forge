"""End-to-end integration tests for ExperimentalPlatform."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge import ExperimentalPlatform
from feature_forge.experiment.execution import ExperimentResult
from feature_forge.methods import BaseMethod


class DummyBaseline(BaseMethod):
    """A minimal baseline for integration testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="dummy")

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


def _deterministic_result(dataset: str, method: str, model: str, seed: int) -> ExperimentResult:
    baseline = 0.7 + (seed % 10) * 0.001
    gain = 0.05 if method == "openfe" else 0.04
    return ExperimentResult(
        dataset=dataset,
        method=method,
        model=model,
        seed=seed,
        cv_score=baseline + gain,
        gain=gain,
        baseline_score=baseline,
        num_features_generated=1,
    )


def _fake_execute(self, case):
    return _deterministic_result(case.dataset, case.method, case.model, case.seed)


def _fake_run_case(payload):
    case = payload.case
    return _deterministic_result(case.dataset, case.method, case.model, case.seed)


# These tests exercise platform plumbing against deliberately tiny synthetic
# frames; the discovery holdout is disabled because the fixtures cannot host a
# stratified holdout partition. Leakage-safe evaluation is covered separately
# in tests/unit/test_discovery_holdout.py.
_TINY_DATA_CONFIG: dict[str, Any] = {"evaluation": {"evaluation_holdout_fraction": 0.0}}


class TestPlatformE2E:
    """Run a real experiment end-to-end with mock data."""

    def test_run_with_synthetic_data(self, tmp_path):
        platform = ExperimentalPlatform(config=_TINY_DATA_CONFIG)
        platform.register_method("dummy", DummyBaseline)

        # Create a minimal synthetic CSV dataset
        train_df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
                "target": [0, 1, 0, 1, 0],
            }
        )
        sample_dir = tmp_path / "synth"
        sample_dir.mkdir()
        train_df.to_csv(sample_dir / "train.csv", index=False)
        (sample_dir / "metadata.json").write_text('{"target": "target", "task": "classification"}')

        platform.register_dataset(
            "synth",
            {
                "source": "local",
                "path": str(sample_dir),
                "target": "target",
                "task": "classification",
            },
        )

        results = platform.run(
            datasets=["synth"],
            methods=["dummy"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )

        assert len(results) == 1
        result = results[0]
        assert result["dataset"] == "synth"
        assert result["method"] == "dummy"
        assert "cv_score" in result
        assert "gain" in result
        assert "baseline_score" in result
        assert result["error"] is None

    def test_run_with_missing_dataset(self):
        platform = ExperimentalPlatform()
        platform.register_method("dummy", DummyBaseline)

        results = platform.run(
            datasets=["nonexistent"],
            methods=["dummy"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )

        assert len(results) == 1
        assert "error" in results[0]
        assert "nonexistent" in results[0]["error"]

    def test_run_with_missing_method(self, tmp_path):
        platform = ExperimentalPlatform()

        train_df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )
        sample_dir = tmp_path / "synth2"
        sample_dir.mkdir()
        train_df.to_csv(sample_dir / "train.csv", index=False)
        (sample_dir / "metadata.json").write_text('{"target": "target", "task": "classification"}')
        platform.register_dataset(
            "synth2",
            {
                "source": "local",
                "path": str(sample_dir),
                "target": "target",
                "task": "classification",
            },
        )

        results = platform.run(
            datasets=["nonexistent"],
            methods=["nonexistent"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )

        assert len(results) == 1
        assert "error" in results[0]
        assert "nonexistent" in results[0]["error"]

    def test_run_returns_dataframe(self, tmp_path):
        platform = ExperimentalPlatform()
        platform.register_method("dummy", DummyBaseline)

        train_df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )
        sample_dir = tmp_path / "synth3"
        sample_dir.mkdir()
        train_df.to_csv(sample_dir / "train.csv", index=False)
        (sample_dir / "metadata.json").write_text('{"target": "target", "task": "classification"}')
        platform.register_dataset(
            "synth3",
            {
                "source": "local",
                "path": str(sample_dir),
                "target": "target",
                "task": "classification",
            },
        )

        results = platform.run(
            datasets=["synth3"],
            methods=["dummy"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )

        df = platform.to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "cv_score" in df.columns

    def test_run_with_no_progress(self, tmp_path):
        platform = ExperimentalPlatform(config=_TINY_DATA_CONFIG)
        platform.register_method("dummy", DummyBaseline)

        train_df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )
        sample_dir = tmp_path / "synth4"
        sample_dir.mkdir()
        train_df.to_csv(sample_dir / "train.csv", index=False)
        (sample_dir / "metadata.json").write_text('{"target": "target", "task": "classification"}')
        platform.register_dataset(
            "synth4",
            {
                "source": "local",
                "path": str(sample_dir),
                "target": "target",
                "task": "classification",
            },
        )

        results = platform.run(
            datasets=["synth4"],
            methods=["dummy"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )

        assert len(results) == 1
        assert results[0]["error"] is None

    def test_report_best_integration(self, tmp_path):
        platform = ExperimentalPlatform(config=_TINY_DATA_CONFIG)
        platform.register_method("dummy", DummyBaseline)

        train_df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )
        sample_dir = tmp_path / "synth5"
        sample_dir.mkdir()
        train_df.to_csv(sample_dir / "train.csv", index=False)
        (sample_dir / "metadata.json").write_text('{"target": "target", "task": "classification"}')
        platform.register_dataset(
            "synth5",
            {
                "source": "local",
                "path": str(sample_dir),
                "target": "target",
                "task": "classification",
            },
        )

        results = platform.run(
            datasets=["synth5"],
            methods=["dummy"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )

        best = platform.report_best(results)
        assert len(best) >= 1

    def test_to_dataframe_with_errors(self, tmp_path):
        platform = ExperimentalPlatform()

        results = platform.run(
            datasets=["nonexistent"],
            methods=["dummy"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )

        df = platform.to_dataframe(results)
        assert isinstance(df, __import__("pandas").DataFrame)
        assert "error" in df.columns

    def test_run_with_empty_methods(self, tmp_path):
        platform = ExperimentalPlatform()
        results = platform.run(
            datasets=["nonexistent"],
            methods=[],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )
        assert len(results) == 0

    def test_run_with_tracker(self, tmp_path):
        from feature_forge.experiment import NoOpTracker

        platform = ExperimentalPlatform()
        platform.register_method("dummy", DummyBaseline)

        train_df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )
        sample_dir = tmp_path / "synth6"
        sample_dir.mkdir()
        train_df.to_csv(sample_dir / "train.csv", index=False)
        (sample_dir / "metadata.json").write_text('{"target": "target", "task": "classification"}')
        platform.register_dataset(
            "synth6",
            {
                "source": "local",
                "path": str(sample_dir),
                "target": "target",
                "task": "classification",
            },
        )

        tracker = NoOpTracker(project="test-platform")
        results = platform.run(
            datasets=["synth6"],
            methods=["dummy"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            tracker=tracker,
            progress=False,
        )
        assert len(results) == 1

    def test_list_models_includes_xgboost_and_more(self):
        platform = ExperimentalPlatform()
        models = platform.list_models()
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "random_forest" in models

    def test_parallel_and_sequential_semantic_equality(self, monkeypatch):
        monkeypatch.setattr(
            "feature_forge.experiment.case_executor.ExperimentCaseExecutor.execute", _fake_execute
        )
        monkeypatch.setattr("feature_forge.platform.run_case", _fake_run_case)

        platform = ExperimentalPlatform()
        kwargs = {
            "datasets": ["titanic"],
            "methods": ["openfe"],
            "models": ["xgboost"],
            "seeds": [1, 2],
            "progress": False,
        }

        seq_results = platform.run(parallel=False, **kwargs)
        par_results = platform.run(parallel=True, max_workers=2, **kwargs)

        seq_sorted = sorted(
            seq_results,
            key=lambda x: (x["dataset"], x["method"], x["model"], x["seed"]),
        )
        par_sorted = sorted(
            par_results,
            key=lambda x: (x["dataset"], x["method"], x["model"], x["seed"]),
        )
        assert seq_sorted == par_sorted

    def test_parallel_supports_portable_instance_local_registered_method(self, tmp_path):
        platform = ExperimentalPlatform(config=_TINY_DATA_CONFIG)
        platform.register_method("dummy", DummyBaseline)
        sample_dir = tmp_path / "portable"
        sample_dir.mkdir()
        pd.DataFrame({"a": list(range(12)), "target": [0, 1] * 6}).to_csv(
            sample_dir / "train.csv", index=False
        )
        platform.register_dataset(
            "portable",
            {
                "source": "local",
                "path": str(sample_dir),
                "target": "target",
                "task": "classification",
            },
        )
        results = platform.run(
            datasets=["portable"],
            methods=["dummy"],
            models=["random_forest"],
            cv_folds=2,
            parallel=True,
            progress=False,
        )
        assert results[0]["error"] is None
