"""End-to-end integration tests for ExperimentalPlatform."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge import ExperimentalPlatform
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


class TestPlatformE2E:
    """Run a real experiment end-to-end with mock data."""

    def test_run_with_synthetic_data(self, tmp_path):
        platform = ExperimentalPlatform()
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
        assert "error" not in result

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
        platform = ExperimentalPlatform()
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
        assert "error" not in results[0]

    def test_report_best_integration(self, tmp_path):
        platform = ExperimentalPlatform()
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
