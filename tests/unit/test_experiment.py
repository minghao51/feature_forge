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

    def test_methods_with_dict(self):
        matrix = ExperimentMatrix().datasets(["a"]).methods({"malmus": ["single_shot"]})
        configs = matrix.generate()
        assert len(configs) == 1
        assert configs[0]["method"] == "malmus"

    def test_rounds(self):
        matrix = ExperimentMatrix().datasets(["a"]).rounds([1, 2]).seeds([42])
        configs = matrix.generate()
        assert len(configs) == 2
        assert configs[0]["n_rounds"] == 1
        assert configs[1]["n_rounds"] == 2

    def test_add_param(self):
        matrix = ExperimentMatrix().add_param("custom", ["x", "y"]).seeds([42])
        configs = matrix.generate()
        assert len(configs) == 2
        assert configs[0]["custom"] == "x"

    def test_full_cartesian(self):
        matrix = (
            ExperimentMatrix().datasets(["a", "b"]).seeds([0, 1]).models(["xgb", "rf"]).rounds([1])
        )
        configs = matrix.generate()
        assert len(configs) == 8  # 2 * 2 * 2 * 1


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


def _parallel_double(cfg: dict) -> dict:
    return {"y": cfg["x"] * 2}


def _parallel_fail_on_2(cfg: dict) -> dict:
    if cfg["x"] == 2:
        raise ValueError("parallel fail")
    return {"y": cfg["x"]}


class TestExperimentRunnerEdgeCases:
    """Cover ExperimentRunner uncovered paths: tracker, no progress, run_parallel."""

    def test_run_with_tracker(self):
        from feature_forge.experiment import NoOpTracker

        tracker = NoOpTracker(project="test-runner")
        runner = ExperimentRunner(tracker=tracker)
        configs = [{"x": 1}, {"x": 2}]
        results = runner.run(configs, lambda cfg: {"y": cfg["x"] * 2})
        assert len(results) == 2

    def test_run_no_progress(self):
        runner = ExperimentRunner()
        configs = [{"x": 1}, {"x": 2}]
        results = runner.run(configs, lambda cfg: {"y": cfg["x"] * 2}, progress=False)
        assert len(results) == 2

    def test_run_parallel_basic(self):
        runner = ExperimentRunner()
        configs = [{"x": 1}, {"x": 2}]
        results = runner.run_parallel(configs, _parallel_double, max_workers=2, progress=False)
        assert len(results) == 2
        scores = {r["x"]: r["y"] for r in results}
        assert scores[1] == 2
        assert scores[2] == 4

    def test_run_parallel_with_error(self):
        configs = [{"x": 1}, {"x": 2}]
        results = ExperimentRunner().run_parallel(
            configs, _parallel_fail_on_2, max_workers=2, progress=False
        )
        assert "error" in results[1]
        assert results[0]["y"] == 1

    def test_run_parallel_no_progress(self):
        runner = ExperimentRunner()
        configs = [{"x": 1}]
        results = runner.run_parallel(configs, _parallel_double, max_workers=1, progress=False)
        assert len(results) == 1

    def test_run_parallel_with_tracker(self):
        from feature_forge.experiment import NoOpTracker

        tracker = NoOpTracker(project="test-parallel")
        runner = ExperimentRunner(tracker=tracker, max_workers=1)
        configs = [{"x": 1}, {"x": 2}]
        results = runner.run_parallel(configs, _parallel_double, progress=False)
        assert len(results) == 2


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
