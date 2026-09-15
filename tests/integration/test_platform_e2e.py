"""End-to-end integration tests for ExperimentalPlatform."""

from __future__ import annotations

import json
import pathlib
from hashlib import sha256
from typing import Any, TypedDict

import pandas as pd
import pytest

from feature_forge import ExperimentalPlatform
from feature_forge.config import FailurePolicy
from feature_forge.contracts import CaseExecutionPlan
from feature_forge.contracts.stages import FailureClass
from feature_forge.experiment.execution import (
    CaseComputationInput,
    ExperimentCase,
    ExperimentResult,
)
from feature_forge.experiment.hamilton_executor import HamiltonLayerExecutor
from feature_forge.experiment.lifecycle import LocalRunRepository
from feature_forge.methods import BaseMethod

# Fork-context pools (plan 22 dimension) fork from a multi-threaded pytest
# parent, which Python 3.12+ warns about; expected here, see
# tests/unit/test_fail_fast_process.py for the full rationale.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*multi-threaded.*fork:DeprecationWarning"
)


class HamiltonDeterministicMethod(BaseMethod):
    """Offline method whose generated feature follows the sandbox contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="hamilton_deterministic")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> HamiltonDeterministicMethod:
        self._artifacts["generated_code"] = (
            "import pandas as pd\n"
            "def generate_features(df):\n"
            "    return pd.DataFrame({'double_a': df['a'] * 2}, index=df.index)\n"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(double_a=X["a"] * 2)

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"name": "double_a", "base_columns": ["a"]}]


class FailingMethod(BaseMethod):
    """Method whose fit always raises — yields a terminal failed case."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="failing")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> FailingMethod:
        raise RuntimeError("deterministic boom")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()

    def get_artifacts(self) -> dict[str, Any]:
        return {}

    @property
    def generated_scripts(self) -> list[str]:
        return []


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


def _case_key_stub(case: ExperimentCase) -> str:
    return f"{case.dataset}-{case.method}-{case.model}-{case.seed}"


def _fake_plan_case(self: HamiltonLayerExecutor, case: ExperimentCase) -> CaseExecutionPlan:
    """Mirror the planning seam: allocate stable case identity without side effects."""
    key = _case_key_stub(case)
    return CaseExecutionPlan(
        case_key=f"case-{key}",
        attempt_id=f"attempt-{key}",
        dataset=case.dataset,
        method=case.method,
        model=case.model,
        unresolved_layers=[],
    )


def _fake_execute_case(self: HamiltonLayerExecutor, case: ExperimentCase) -> ExperimentResult:
    """Mirror the execution seam contract: per-case errors become error results."""
    if case.dataset not in self.dataset_registry.list():
        return ExperimentResult(
            dataset=case.dataset,
            method=case.method,
            model=case.model,
            seed=case.seed,
            error=f"Dataset '{case.dataset}' not found",
        )
    if case.method not in self.method_classes:
        return ExperimentResult(
            dataset=case.dataset,
            method=case.method,
            model=case.model,
            seed=case.seed,
            error=f"Method '{case.method}' not found",
        )
    return _deterministic_result(case.dataset, case.method, case.model, case.seed)


def _fake_run_case(payload: CaseComputationInput) -> ExperimentResult:
    case = payload.case
    return _deterministic_result(case.dataset, case.method, case.model, case.seed)


class _RunKwargs(TypedDict):
    """Keyword arguments shared by the sequential and parallel platform.run calls."""

    datasets: list[str]
    methods: list[str]
    models: list[str]
    seeds: list[int]
    progress: bool


class TestPlatformE2E:
    """Platform orchestration end-to-end over faked Hamilton seams.

    ADR 0016 removed the legacy engine; the former imperative-engine tests now
    drive the same orchestration through faked ``HamiltonLayerExecutor``
    planning/execution seams and the ``run_hamilton_case`` worker seam. The
    real four-layer Hamilton chain is covered by ``TestPlatformHamiltonFullCase``.
    """

    @pytest.fixture(autouse=True)
    def _hamilton_seam_fakes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Route every test in this class through the Hamilton seam fakes."""
        monkeypatch.setattr(
            "feature_forge.experiment.hamilton_executor.HamiltonLayerExecutor.plan_case",
            _fake_plan_case,
        )
        monkeypatch.setattr(
            "feature_forge.experiment.hamilton_executor.HamiltonLayerExecutor.execute_case",
            _fake_execute_case,
        )
        monkeypatch.setattr("feature_forge.platform.run_hamilton_case", _fake_run_case)

    def test_run_with_synthetic_data(self, tmp_path: pathlib.Path) -> None:
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
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

    def test_run_with_missing_dataset(self) -> None:
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
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

    def test_run_with_missing_method(self, tmp_path: pathlib.Path) -> None:
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})

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

    def test_run_returns_dataframe(self, tmp_path: pathlib.Path) -> None:
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
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

    def test_run_with_no_progress(self, tmp_path: pathlib.Path) -> None:
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
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

    def test_report_best_integration(self, tmp_path: pathlib.Path) -> None:
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
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

    def test_to_dataframe_with_errors(self, tmp_path: pathlib.Path) -> None:
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})

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

    def test_run_with_empty_methods(self, tmp_path: pathlib.Path) -> None:
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
        results = platform.run(
            datasets=["nonexistent"],
            methods=[],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )
        assert len(results) == 0

    def test_run_with_tracker(self, tmp_path: pathlib.Path) -> None:
        from feature_forge.experiment import NoOpTracker

        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
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

    def test_list_models_includes_xgboost_and_more(self) -> None:
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
        models = platform.list_models()
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "random_forest" in models

    def test_parallel_and_sequential_semantic_equality(self) -> None:
        """Sequential and Hamilton process-worker paths agree semantically."""
        platform = ExperimentalPlatform(config={"tracker": {"backend": "none"}})
        kwargs = _RunKwargs(
            datasets=["titanic"],
            methods=["openfe"],
            models=["xgboost"],
            seeds=[1, 2],
            progress=False,
        )

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


class TestPlatformHamiltonFullCase:
    """Public API produces and then reuses a verified four-layer chain."""

    def test_public_default_executes_and_reuses_all_layers(self, tmp_path: pathlib.Path) -> None:
        sample_dir = tmp_path / "hamilton-synth"
        sample_dir.mkdir()
        pd.DataFrame(
            {
                "a": list(range(8)),
                "b": list(range(10, 18)),
                "target": [0, 1] * 4,
            }
        ).to_csv(sample_dir / "train.csv", index=False)
        (sample_dir / "metadata.json").write_text('{"target":"target","task":"classification"}')
        artifact_root = tmp_path / "artifacts"
        platform = ExperimentalPlatform(
            config={
                "metric": "acc",
                "dataflow": {
                    "artifact_root": artifact_root,
                    "cache": {"path": tmp_path / "cache"},
                },
            }
        )
        platform.register_dataset(
            "hamilton-synth",
            {
                "source": "local",
                "path": str(sample_dir),
                "target": "target",
                "task": "classification",
            },
        )
        platform.register_method("hamilton_deterministic", HamiltonDeterministicMethod)

        kwargs: dict[str, Any] = {
            "datasets": ["hamilton-synth"],
            "methods": ["hamilton_deterministic"],
            "models": ["random_forest"],
            "cv_folds": 2,
            "seeds": [42],
            "progress": False,
        }
        first = platform.run(**kwargs)[0]
        second = platform.run(**kwargs, parallel=True, max_workers=1)[0]
        platform._get_settings().llm.temperature = 0.7
        identity_changed = platform.run(**kwargs)[0]

        assert first["error"] is None
        assert [stage["layer"] for stage in first["stages"]] == [
            "bronze",
            "silver",
            "gold",
            "platinum",
        ]
        assert [stage["disposition"] for stage in first["stages"]] == ["executed"] * 4
        assert [stage["disposition"] for stage in second["stages"]] == ["reused"] * 4
        assert [stage["disposition"] for stage in identity_changed["stages"]] == [
            "reused",
            "reused",
            "executed",
            "executed",
        ]
        assert first["run_id"] != second["run_id"]
        for directory in ("01_bronze", "02_silver", "03_gold", "04_platinum"):
            assert any((artifact_root / directory / "runs").iterdir())
        lifecycle_text = (
            artifact_root / "control" / "lifecycle" / "runs" / first["run_id"] / "events.jsonl"
        ).read_text(encoding="utf-8")
        assert '"event_type": "case_running"' in lifecycle_text
        assert lifecycle_text.count('"event_type": "stage_succeeded"') == 4
        assert '"event_type": "case_succeeded"' in lifecycle_text
        lifecycle_events = [json.loads(line) for line in lifecycle_text.splitlines()]
        cache_events = [
            event for event in lifecycle_events if event["event_type"] == "hamilton_node_cache"
        ]
        assert cache_events
        assert {event["layer"] for event in cache_events} <= {
            "bronze",
            "silver",
            "gold",
            "platinum",
        }
        assert all(
            {"cache_outcome", "duration_ms", "serialized_bytes"} <= event["details"].keys()
            for event in cache_events
        )
        assert all(
            forbidden not in lifecycle_text
            for forbidden in ("node_input", "node_result", "prompt_body", "api_key")
        )


class TestFailFastResumeTruthfulness:
    """Sequential fail-fast on the real chain (plan 22 PR 2, ADR 0017).

    Cancellation must not delete or modify any verified package, and a later
    invocation must be able to resume from the verified prefix produced by a
    case that completed before the stop.
    """

    @staticmethod
    def _snapshot(root: pathlib.Path) -> dict[str, str]:
        """Map every file below ``root`` to its content digest."""
        return {
            str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_fail_fast_preserves_packages_and_resumes_completed_prefix(
        self, tmp_path: pathlib.Path
    ) -> None:
        sample_dir = tmp_path / "hamilton-synth"
        sample_dir.mkdir()
        pd.DataFrame(
            {
                "a": list(range(8)),
                "b": list(range(10, 18)),
                "target": [0, 1] * 4,
            }
        ).to_csv(sample_dir / "train.csv", index=False)
        (sample_dir / "metadata.json").write_text('{"target":"target","task":"classification"}')
        artifact_root = tmp_path / "artifacts"
        platform = ExperimentalPlatform(
            config={
                "metric": "acc",
                "dataflow": {
                    "artifact_root": artifact_root,
                    "cache": {"path": tmp_path / "cache"},
                },
            }
        )
        platform.register_dataset(
            "hamilton-synth",
            {
                "source": "local",
                "path": str(sample_dir),
                "target": "target",
                "task": "classification",
            },
        )
        platform.register_method("hamilton_deterministic", HamiltonDeterministicMethod)
        platform.register_method("failing", FailingMethod)

        first = platform.run(
            datasets=["hamilton-synth"],
            methods=["hamilton_deterministic"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
        )[0]
        assert first["error"] is None
        before = self._snapshot(artifact_root)
        assert before, "the completed case must produce durable packages"

        results = platform.run(
            datasets=["hamilton-synth"],
            methods=["hamilton_deterministic", "failing", "hamilton_deterministic"],
            models=["random_forest"],
            cv_folds=2,
            seeds=[42],
            progress=False,
            failure_policy=FailurePolicy.FAIL_FAST,
        )

        # success, terminal failure, then exactly one cancelled never-started case.
        assert [row["state"] for row in results] == ["succeeded", "failed", "cancelled"]
        cancelled = results[2]
        assert cancelled["cv_score"] is None
        assert cancelled["stages"] == []
        # The first case resumed from the verified prefix of the completed run.
        dispositions = [stage["disposition"] for stage in results[0]["stages"]]
        assert dispositions[:2] == ["reused", "reused"]

        # Every file that existed before the cancelled run is byte-identical.
        after = self._snapshot(artifact_root)
        assert set(before) <= set(after)
        for relative_path, digest in before.items():
            assert after[relative_path] == digest, relative_path

        # One redacted parent-side case_cancelled event for the unstarted case.
        events = LocalRunRepository(artifact_root / "control" / "lifecycle").load_events(
            cancelled["run_id"]
        )
        cancelled_events = [event for event in events if event.event_type == "case_cancelled"]
        assert len(cancelled_events) == 1
        assert cancelled_events[0].state is not None
        assert cancelled_events[0].state.value == "cancelled"
        assert cancelled_events[0].failure is not None
        assert cancelled_events[0].failure.failure_class is FailureClass.CANCELLED
        # Redaction: no triggering exception text in the cancellation event.
        assert "boom" not in cancelled_events[0].failure.message


class TestPlatformHamiltonDefaultDispatch:
    """Default engine is Hamilton (ADR 0016); stale legacy settings are rejected."""

    def test_run_default_dispatches_hamilton_with_four_layers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock, patch

        from feature_forge.contracts import (
            CaseExecutionPlan,
            Layer,
            ManifestRef,
            StageDisposition,
            StageExecution,
        )

        manifest_ref = ManifestRef(layer=Layer.BRONZE, run_id="attempt-1", sha256="a" * 64)

        def _stages() -> list[StageExecution]:
            return [
                StageExecution(
                    layer=layer,
                    disposition=StageDisposition.EXECUTED,
                    manifest_ref=manifest_ref,
                    reuse_fingerprint=f"reuse-{layer.value}",
                    layer_fingerprint=f"layer-{layer.value}",
                )
                for layer in (Layer.BRONZE, Layer.SILVER, Layer.GOLD, Layer.PLATINUM)
            ]

        mock_executor = MagicMock()
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
            num_features_generated=3,
            run_id="attempt-1",
            case_fingerprint="casekey",
            stages=_stages(),
        )

        with patch("feature_forge.platform.HamiltonLayerExecutor", return_value=mock_executor):
            platform = ExperimentalPlatform()
            results = platform.run(datasets=["titanic"], methods=["dummy"], progress=False)

        assert len(results) == 1
        result = results[0]
        assert result["method"] == "dummy"
        # All four medallion layers are present and in dependency order.
        layers = [stage["layer"] for stage in result["stages"]]
        assert layers == ["bronze", "silver", "gold", "platinum"]
        # Nested StageExecution instances serialized to plain dicts.
        assert result["stages"][0]["manifest_ref"]["layer"] == "bronze"
