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

    def test_summary_stats(self):
        results = [
            {"dataset": "a", "score": 0.8},
            {"dataset": "a", "score": 0.9, "error": "fail"},
        ]
        reporter = Reporter(results)
        stats = reporter.summary_stats()
        assert stats["total_runs"] == 2
        assert stats["successful_runs"] == 1
        assert stats["failed_runs"] == 1

    def test_empty_results(self):
        reporter = Reporter([])
        assert "No results" in reporter.to_markdown()

    def test_to_html(self):
        results = [{"dataset": "a", "score": 0.8}]
        reporter = Reporter(results)
        html = reporter.to_html()
        assert "<table" in html or "No results" in html

    def test_to_html_empty(self):
        reporter = Reporter([])
        html = reporter.to_html()
        assert "No results" in html

    def test_get_best_invalid_metric(self):
        results = [{"dataset": "a", "score": 0.8}]
        reporter = Reporter(results)
        best = reporter.get_best(metric="nonexistent", group_by="dataset")
        assert len(best) == 0

    def test_get_best_empty_results(self):
        reporter = Reporter([])
        best = reporter.get_best(metric="score", group_by="dataset")
        assert len(best) == 0

    def test_summary_stats_empty(self):
        reporter = Reporter([])
        stats = reporter.summary_stats()
        assert stats["total_runs"] == 0

    def test_summary_stats_treats_missing_error_as_success(self):
        reporter = Reporter([{"dataset": "a", "score": 0.8}])
        stats = reporter.summary_stats()
        assert stats["successful_runs"] == 1
        assert stats["failed_runs"] == 0

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
