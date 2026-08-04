"""Tests for experiment harness."""

from __future__ import annotations

from feature_forge.experiment import (
    NoOpTracker,
    Reporter,
)


class TestNoOpTracker:
    def test_all_methods_noop(self):
        tracker = NoOpTracker(project="test")
        tracker.init_run("run1", {"x": 1})
        tracker.log_metrics({"acc": 0.9})
        tracker.log_params({"lr": 0.01})
        tracker.log_artifact("path/to/file")
        tracker.finish()

    def test_log_artifacts_dict_with_values(self):
        import pandas as pd

        tracker = NoOpTracker(project="test-artifacts")
        tracker.log_artifacts_dict(
            {
                "score": 0.9,
                "enabled": True,
                "count": 42,
                "code_snippet": "def foo(): pass",
                "params_list": [1, 2, 3],
                "params_dict": {"a": 1},
                "plain_string": "hello",
                "dataframe": pd.DataFrame({"x": [1]}),
            }
        )

    def test_log_artifacts_dict_with_prefix(self):
        tracker = NoOpTracker(project="test-prefix")
        tracker.log_artifacts_dict({"val": 1.0}, prefix="round_0_")

    def test_log_artifacts_dict_with_lazy_ref(self):
        tracker = NoOpTracker(project="test-lazy")

        class FakeLazyRef:
            def load(self):
                return "loaded_value"

        tracker.log_artifacts_dict({"ref": FakeLazyRef()})

    def test_log_artifacts_dict_with_none(self):
        tracker = NoOpTracker(project="test-none")
        tracker.log_artifacts_dict({"nothing": None})


class TestReporter:
    def test_to_markdown(self):
        results = [
            {"dataset": "a", "method": "m1", "score": 0.8},
            {"dataset": "a", "method": "m2", "score": 0.9},
            {"dataset": "b", "method": "m1", "score": 0.7},
        ]
        reporter = Reporter(results)
        md = reporter.to_markdown()
        assert "m1" in md or "0.8" in md or "dataset" in md

    def test_get_best(self):
        results = [
            {"dataset": "a", "score": 0.8},
            {"dataset": "a", "score": 0.9},
            {"dataset": "b", "score": 0.7},
        ]
        reporter = Reporter(results)
        best = reporter.get_best(metric="score", group_by="dataset")
        assert len(best) == 2

    def test_empty_results(self):
        reporter = Reporter([])
        assert "No results" in reporter.to_markdown()

    def test_get_best_invalid_metric(self):
        results = [{"dataset": "a", "score": 0.8}]
        reporter = Reporter(results)
        best = reporter.get_best(metric="nonexistent", group_by="dataset")
        assert len(best) == 0

    def test_get_best_empty_results(self):
        reporter = Reporter([])
        best = reporter.get_best(metric="score", group_by="dataset")
        assert len(best) == 0

    def test_get_best_can_minimize_metric(self):
        reporter = Reporter(
            [
                {"dataset": "a", "metric": "rmse", "score": 2.0},
                {"dataset": "a", "metric": "rmse", "score": 1.0},
            ]
        )
        best = reporter.get_best(metric="score", group_by="dataset")
        assert best.iloc[0]["score"] == 1.0

    def test_mixed_legacy_and_enriched_results_render(self):
        reporter = Reporter(
            [
                {"dataset": "a", "method": "m", "model": "rf", "cv_score": 0.8},
                {
                    "dataset": "a",
                    "method": "m",
                    "model": "rf",
                    "cv_score": 0.9,
                    "directional_gain": 0.1,
                    "uncertainty": {"lower_bound": 0.01},
                    "platinum_manifest_uri": "04_platinum/runs/r/manifest.json",
                },
            ]
        )
        assert "cv_score" in reporter.to_markdown()

    def test_get_best_default_metric_is_cv_score(self):
        # The default metric matches ExperimentalPlatform.report_best and the
        # ExperimentResult.cv_score column produced by the platform path.
        reporter = Reporter(
            [
                {"dataset": "a", "method": "m1", "model": "rf", "cv_score": 0.8},
                {"dataset": "a", "method": "m2", "model": "rf", "cv_score": 0.9},
            ]
        )
        best = reporter.get_best(group_by="dataset")
        assert len(best) == 1
        assert best.iloc[0]["method"] == "m2"  # maximize by default

    def test_get_best_minimizes_when_metric_column_names_rmse(self):
        # With a single minimize metric in the `metric` column and the default
        # cv_score metric, the reporter must pick the LOWEST score, not highest.
        reporter = Reporter(
            [
                {"dataset": "a", "metric": "rmse", "cv_score": 2.0},
                {"dataset": "a", "metric": "rmse", "cv_score": 1.0},
            ]
        )
        best = reporter.get_best(group_by="dataset")
        assert len(best) == 1
        assert best.iloc[0]["cv_score"] == 1.0


def test_experiment_case_effective_run_id_uses_explicit_when_set():
    from feature_forge.experiment.execution import ExperimentCase

    explicit = ExperimentCase(
        dataset="titanic", method="m", model="rf", seed=1, run_id="custom-run"
    )
    assert explicit.effective_run_id == "custom-run"


def test_experiment_case_effective_run_id_falls_back_deterministically():
    from feature_forge.experiment.execution import ExperimentCase

    case = ExperimentCase(dataset="titanic", method="openfe", model="xgboost", seed=42)
    assert case.effective_run_id == "run_titanic_openfe_xgboost_42"
