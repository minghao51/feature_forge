"""Tests for experiment harness."""

from __future__ import annotations

from feature_forge.experiment import (
    ExperimentMatrix,
    ExperimentRunner,
    NoOpTracker,
    Reporter,
)


class TestExperimentMatrix:
    def test_generate_combinations(self):
        matrix = ExperimentMatrix().datasets(["titanic"]).seeds([0, 1]).models(["xgb"])
        configs = matrix.generate()
        assert len(configs) == 2
        assert configs[0]["dataset"] == "titanic"
        assert configs[0]["seed"] == 0

    def test_len(self):
        matrix = ExperimentMatrix().datasets(["a", "b"]).seeds([0, 1, 2])
        assert len(matrix) == 6

    def test_empty_params(self):
        matrix = ExperimentMatrix()
        assert matrix.generate() == []
        assert len(matrix) == 0


class TestNoOpTracker:
    def test_all_methods_noop(self):
        tracker = NoOpTracker(project="test")
        tracker.init_run("run1", {"x": 1})
        tracker.log_metrics({"acc": 0.9})
        tracker.log_params({"lr": 0.01})
        tracker.log_artifact("path/to/file")
        tracker.finish()


class TestExperimentRunner:
    def test_run_sequential(self):
        runner = ExperimentRunner()
        configs = [{"x": 1}, {"x": 2}]
        results = runner.run(configs, lambda cfg: {"y": cfg["x"] * 2})
        assert len(results) == 2
        assert results[0]["y"] == 2
        assert results[1]["y"] == 4

    def test_run_with_error(self):
        runner = ExperimentRunner()
        configs = [{"x": 1}, {"x": 2}]

        def fail_on_2(cfg):
            if cfg["x"] == 2:
                raise ValueError("fail")
            return {"y": cfg["x"]}

        results = runner.run(configs, fail_on_2)
        assert "error" in results[1]
        assert results[0]["y"] == 1


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
