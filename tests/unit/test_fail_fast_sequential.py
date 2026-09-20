"""Sequential fail-fast and cancellation scheduling (plan 22 PR 2, ADR 0017).

Covers the sequential runtime slice of the failure/cancellation contract:
policy resolution on ``ExperimentalPlatform.run()`` (run override >
``settings.execution.failure_policy``, cached settings never mutated),
fail-fast stopping after the first terminal failed case, cooperative token
cancellation at case boundaries, typed cancelled results with
parent-allocated identity, the redacted parent-side ``case_cancelled``
lifecycle events, tracker exactly-once rules, artifact truthfulness, and
default-behavior backward compatibility. Bounded process submission belongs
to plan 22 PR 3 (`tests/unit/test_fail_fast_process.py`); only the platform
wiring into the process adapter is asserted here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from feature_forge import ExperimentalPlatform
from feature_forge import platform as platform_module
from feature_forge.config import FailurePolicy, Settings
from feature_forge.contracts import CaseExecutionPlan, Layer
from feature_forge.contracts.orchestration import FailureRecord
from feature_forge.contracts.stages import FailureClass, RunState
from feature_forge.experiment.execution import (
    CancellationToken,
    ExperimentCase,
    ExperimentResult,
)
from feature_forge.experiment.lifecycle import LocalRunRepository

# ── Fakes and helpers ──────────────────────────────────────────


def _success(method: str = "m") -> ExperimentResult:
    return ExperimentResult(
        dataset="d",
        method=method,
        model="random_forest",
        seed=42,
        cv_score=0.9,
        gain=0.1,
        baseline_score=0.8,
        num_features_generated=2,
    )


def _failure(method: str = "m", message: str = "terminal boom") -> ExperimentResult:
    return ExperimentResult(
        dataset="d",
        method=method,
        model="random_forest",
        seed=42,
        error=message,
        failure=FailureRecord(
            failure_class=FailureClass.DETERMINISTIC,
            error_type="ValueError",
            message=message,
            stage="case",
        ),
    )


def _cancelled_row(method: str = "m") -> ExperimentResult:
    return ExperimentResult(
        dataset="d",
        method=method,
        model="random_forest",
        seed=42,
        run_id=f"attempt-{method}",
        case_fingerprint=f"casekey-{method}",
        error="Case did not run: cancelled before start (fail_fast).",
        failure=FailureRecord(
            failure_class=FailureClass.CANCELLED,
            error_type="CaseCancelled",
            message="Case did not run: cancelled before start (fail_fast).",
            stage="case",
        ),
        state=RunState.CANCELLED,
    )


class _ScriptedExecutor:
    """Stand-in for the ``HamiltonLayerExecutor`` planning/execution seams.

    ``plan_case`` mirrors the real seam: read-only identity allocation. Each
    ``execute_case`` call consumes the next scripted outcome. ``on_execute``
    is invoked with the zero-based case index just before the outcome is
    returned (used to cancel the token mid-case).
    """

    def __init__(
        self,
        outcomes: Sequence[ExperimentResult],
        on_execute: Callable[[int], None] | None = None,
    ) -> None:
        self._outcomes = list(outcomes)
        self._on_execute = on_execute
        self.planned: list[str] = []
        self.executed: list[str] = []

    def plan_case(self, case: ExperimentCase) -> CaseExecutionPlan:
        self.planned.append(case.method)
        return CaseExecutionPlan(
            case_key=f"casekey-{case.method}",
            attempt_id=f"attempt-{case.method}",
            dataset=case.dataset,
            method=case.method,
            model=case.model,
            unresolved_layers=list(Layer),
        )

    def execute_case(self, case: ExperimentCase) -> ExperimentResult:
        index = len(self.executed)
        self.executed.append(case.method)
        if self._on_execute is not None:
            self._on_execute(index)
        return replace(
            self._outcomes[index],
            dataset=case.dataset,
            method=case.method,
            model=case.model,
            seed=case.seed,
            run_id=case.attempt_id,
            case_fingerprint=case.case_key,
        )


def _install_executor(monkeypatch: pytest.MonkeyPatch, executor: _ScriptedExecutor) -> None:
    monkeypatch.setattr(platform_module, "HamiltonLayerExecutor", lambda **_: executor)


def _cases(count: int) -> list[ExperimentCase]:
    return [
        ExperimentCase(dataset="d", method=f"m{index}", model="random_forest", seed=42)
        for index in range(1, count + 1)
    ]


def _platform(tmp_path: Path, **config: Any) -> ExperimentalPlatform:
    base: dict[str, Any] = {
        "tracker": {"backend": "none"},
        "dataflow": {
            "artifact_root": tmp_path / "artifacts",
            "cache": {"path": tmp_path / "cache"},
        },
    }
    base.update(config)
    return ExperimentalPlatform(config=base)


def _run(platform: ExperimentalPlatform, count: int, **kwargs: Any) -> list[dict[str, Any]]:
    return platform.run(
        datasets=["d"],
        methods=[f"m{index}" for index in range(1, count + 1)],
        progress=False,
        **kwargs,
    )


# ── Continue policy (plan 22 §6 test 1) ────────────────────────


class TestContinuePolicy:
    @pytest.mark.parametrize("policy", [None, FailurePolicy.CONTINUE], ids=["default", "explicit"])
    def test_all_cases_execute_despite_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: FailurePolicy | None
    ) -> None:
        executor = _ScriptedExecutor([_success(), _failure(), _success()])
        _install_executor(monkeypatch, executor)
        tracker = MagicMock()

        kwargs: dict[str, Any] = {"tracker": tracker}
        if policy is not None:
            kwargs["failure_policy"] = policy
        results = _run(_platform(tmp_path), 3, **kwargs)

        assert executor.executed == ["m1", "m2", "m3"]
        assert [row["state"] for row in results] == ["succeeded", "failed", "succeeded"]
        assert tracker.init_run.call_count == 3

    def test_continue_records_no_cancelled_lifecycle_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_success(), _failure()])
        _install_executor(monkeypatch, executor)
        artifact_root = tmp_path / "artifacts"

        _run(_platform(tmp_path), 2, failure_policy=FailurePolicy.CONTINUE)

        runs_root = artifact_root / "control" / "lifecycle" / "runs"
        # The scripted executor journals nothing worker-side and nothing was
        # cancelled, so the parent must not have created any run journals.
        assert not runs_root.exists() or list(runs_root.iterdir()) == []


# ── Sequential fail-fast (plan 22 §6 test 2) ───────────────────


class TestSequentialFailFast:
    def test_stops_after_first_terminal_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_success(), _failure(), _success(), _success()])
        _install_executor(monkeypatch, executor)

        results = _run(_platform(tmp_path), 4, failure_policy=FailurePolicy.FAIL_FAST)

        # Matrix order, exact cardinality: one result per requested case.
        assert [row["state"] for row in results] == [
            "succeeded",
            "failed",
            "cancelled",
            "cancelled",
        ]
        assert executor.executed == ["m1", "m2"]
        # Identity is allocated for every case, executed or not (read-only).
        assert executor.planned == ["m1", "m2", "m3", "m4"]

    def test_cancelled_rows_carry_no_packages_or_fabricated_scores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_failure(), _success(), _success()])
        _install_executor(monkeypatch, executor)

        results = _run(_platform(tmp_path), 3, failure_policy=FailurePolicy.FAIL_FAST)
        cancelled = results[1:]

        assert len(cancelled) == 2
        for row in cancelled:
            assert row["cv_score"] is None
            assert row["gain"] is None
            assert row["baseline_score"] is None
            assert row["num_features_generated"] is None
            assert row["stages"] == []
            assert row["failure"] is not None
            assert row["failure"]["failure_class"] == "cancelled"
            assert row["failure"]["error_type"] == "CaseCancelled"
            assert row["state"] == "cancelled"
            assert "cancelled before start" in row["error"]
            assert row["run_id"]
            assert row["case_fingerprint"]
        # Each cancelled case carries its own allocated attempt identity.
        assert cancelled[0]["run_id"] != cancelled[1]["run_id"]
        assert cancelled[0]["case_fingerprint"] != cancelled[1]["case_fingerprint"]

    def test_succeeded_cases_never_stop_fail_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_success(), _success(), _success()])
        _install_executor(monkeypatch, executor)

        results = _run(_platform(tmp_path), 3, failure_policy=FailurePolicy.FAIL_FAST)

        assert executor.executed == ["m1", "m2", "m3"]
        assert [row["state"] for row in results] == ["succeeded", "succeeded", "succeeded"]

    def test_settings_policy_applies_when_param_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_success(), _failure(), _success()])
        _install_executor(monkeypatch, executor)
        platform = _platform(tmp_path, execution={"failure_policy": "fail_fast"})

        results = _run(platform, 3)

        assert executor.executed == ["m1", "m2"]
        assert [row["state"] for row in results] == [
            "succeeded",
            "failed",
            "cancelled",
        ]

    def test_run_override_beats_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_success(), _failure(), _success()])
        _install_executor(monkeypatch, executor)
        platform = _platform(tmp_path, execution={"failure_policy": "fail_fast"})

        results = _run(platform, 3, failure_policy=FailurePolicy.CONTINUE)

        assert executor.executed == ["m1", "m2", "m3"]
        assert [row["state"] for row in results] == ["succeeded", "failed", "succeeded"]

    def test_invalid_override_fails_immediately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_success()])
        _install_executor(monkeypatch, executor)

        with pytest.raises(ValueError, match="FailurePolicy"):
            _platform(tmp_path).run(
                datasets=["d"],
                methods=["m1"],
                failure_policy="abort",  # type: ignore[arg-type]
                progress=False,
            )
        assert executor.executed == []


# ── Cooperative token cancellation (plan 22 §6 tests 3, 4) ─────


class TestCancellationTokenScheduling:
    def test_pre_cancelled_token_executes_zero_cases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_success(), _success(), _success()])
        _install_executor(monkeypatch, executor)
        tracker = MagicMock()
        token = CancellationToken.cancelled("stop_early")

        results = _run(_platform(tmp_path), 3, tracker=tracker, cancellation_token=token)

        assert executor.executed == []
        assert len(executor.planned) == 3  # identity allocation only
        assert [row["state"] for row in results] == ["cancelled", "cancelled", "cancelled"]
        assert "(stop_early)" in results[0]["error"]
        tracker.init_run.assert_not_called()
        tracker.finish.assert_not_called()

    def test_token_cancelled_during_case_finishes_it_then_cancels_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = CancellationToken()

        def cancel_during_first_case(index: int) -> None:
            if index == 0:
                token.cancel("mid_run_stop")

        executor = _ScriptedExecutor(
            [_success(), _success(), _success()], on_execute=cancel_during_first_case
        )
        _install_executor(monkeypatch, executor)

        results = _run(_platform(tmp_path), 3, cancellation_token=token)

        # The in-flight case keeps its real result; the rest are cancelled.
        assert executor.executed == ["m1"]
        assert [row["state"] for row in results] == ["succeeded", "cancelled", "cancelled"]
        assert results[0]["cv_score"] == pytest.approx(0.9)
        assert "(mid_run_stop)" in results[1]["error"]

    def test_fail_fast_and_token_stop_reasons_do_not_mix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token = CancellationToken()

        def cancel_during_second_case(index: int) -> None:
            if index == 1:
                token.cancel("token_reason")

        executor = _ScriptedExecutor(
            [_failure(), _success(), _success()], on_execute=cancel_during_second_case
        )
        _install_executor(monkeypatch, executor)

        results = _run(_platform(tmp_path), 3, failure_policy=FailurePolicy.FAIL_FAST)

        # The fail-fast stop happens first; the token never gets to drain.
        assert executor.executed == ["m1"]
        assert [row["state"] for row in results] == ["failed", "cancelled", "cancelled"]
        assert "(fail_fast)" in results[1]["error"]


# ── Tracker rules (plan 22 §6 test 8) ──────────────────────────


class TestTrackerRules:
    def test_exactly_once_for_completions_never_for_cancelled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_success(), _failure(), _success(), _success()])
        _install_executor(monkeypatch, executor)
        tracker = MagicMock()

        _run(_platform(tmp_path), 4, tracker=tracker, failure_policy=FailurePolicy.FAIL_FAST)

        # Only the two cases that actually started get tracker runs.
        assert tracker.init_run.call_count == 2
        assert tracker.finish.call_count == 2
        # The failed case logs no fabricated metrics.
        assert tracker.log_metrics.call_count == 1
        run_names = [call.kwargs["run_name"] for call in tracker.init_run.call_args_list]
        assert run_names == ["attempt-m1", "attempt-m2"]


# ── Redacted parent lifecycle events (plan 22 §6 test 9) ───────


class TestCancelledLifecycle:
    def test_one_redacted_case_cancelled_event_per_unstarted_case(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor(
            [_success(), _failure("terminal boom"), _success(), _success()]
        )
        _install_executor(monkeypatch, executor)
        artifact_root = tmp_path / "artifacts"

        results = _run(_platform(tmp_path), 4, failure_policy=FailurePolicy.FAIL_FAST)
        cancelled_rows = [row for row in results if row["state"] == "cancelled"]
        assert len(cancelled_rows) == 2

        lifecycle_root = artifact_root / "control" / "lifecycle"
        repository = LocalRunRepository(lifecycle_root)
        runs_root = lifecycle_root / "runs"
        # Only the never-started attempts get parent-side journals; the
        # scripted executor journals nothing worker-side for executed cases.
        assert {path.name for path in runs_root.iterdir()} == {
            row["run_id"] for row in cancelled_rows
        }
        for row in cancelled_rows:
            events = repository.load_events(row["run_id"])
            assert len(events) == 1
            event = events[0]
            assert event.event_type == "case_cancelled"
            assert event.state is RunState.CANCELLED
            assert event.run_id == row["run_id"]
            assert event.case_id == row["case_fingerprint"]
            assert event.fingerprint == row["case_fingerprint"]
            assert event.failure is not None
            assert event.failure.failure_class is FailureClass.CANCELLED
            assert event.failure.error_type == "CaseCancelled"
            # Redaction: no stage/manifest payload, no free-form details.
            assert event.manifest_ref is None
            assert event.stage is None
            assert event.details == {}
        # The triggering case's exception text never reaches the journal.
        for row in cancelled_rows:
            text = (runs_root / row["run_id"] / "events.jsonl").read_text(encoding="utf-8")
            assert "boom" not in text
            assert "prompt" not in text.lower()
            assert "api_key" not in text


# ── Artifact truthfulness (plan 22 §6 test 10, unit level) ─────


class TestArtifactTruthfulness:
    def test_cancelled_run_leaves_existing_artifacts_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact_root = tmp_path / "artifacts"
        package = artifact_root / "04_platinum" / "runs" / "attempt-m1" / "package.json"
        package.parent.mkdir(parents=True)
        package.write_text('{"verified": true}', encoding="utf-8")

        executor = _ScriptedExecutor([_success(), _failure(), _success()])
        _install_executor(monkeypatch, executor)
        results = _run(_platform(tmp_path), 3, failure_policy=FailurePolicy.FAIL_FAST)

        assert results[-1]["state"] == "cancelled"
        assert package.read_text(encoding="utf-8") == '{"verified": true}'


# ── Backward compatibility (plan 22 §6 test 12) ────────────────


class TestBackwardCompatibility:
    ROW_KEYS: ClassVar[set[str]] = {
        "dataset",
        "method",
        "model",
        "seed",
        "cv_score",
        "gain",
        "baseline_score",
        "num_features_generated",
        "error",
        "run_id",
        "case_fingerprint",
        "stages",
        "failure",
        "state",
        # Directional interval fields (ADR 0018 decisions 6-7, plan 23 PR 4).
        "directional_gain",
        "gain_lower_bound",
        "gain_upper_bound",
        "evaluation_protocol",
    }

    def test_omitted_policy_matches_previous_behavior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor = _ScriptedExecutor([_success(), _failure(), _success()])
        _install_executor(monkeypatch, executor)

        results = _run(_platform(tmp_path), 3)

        assert executor.executed == ["m1", "m2", "m3"]
        assert len(results) == 3
        for row in results:
            assert set(row) == self.ROW_KEYS
        assert [row["state"] for row in results] == ["succeeded", "failed", "succeeded"]

    def test_process_path_receives_policy_and_token_wiring(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The platform drives the bounded adapter with parent-side stop hooks.

        Plan 22 PR 3: the parallel path passes the effective policy (as a
        ``stop_on_result`` verdict) plus the token and the cancelled-row
        factory into the adapter, and rows the adapter reports as cancelled
        initialize no tracker run (plan 22 §3).
        """
        pool = MagicMock()
        captured: dict[str, Any] = {}

        def _pool_run(
            payloads: list[Any], worker: Any, progress: bool = True, **kwargs: Any
        ) -> list[ExperimentResult]:
            captured.update(kwargs)
            return [
                _success(method="m1"),
                _failure(method="m2"),
                _cancelled_row(method="m3"),
            ]

        pool.run.side_effect = _pool_run
        monkeypatch.setattr(platform_module, "ProcessPoolExecutionAdapter", lambda **_: pool)
        executor = _ScriptedExecutor([])
        _install_executor(monkeypatch, executor)
        tracker = MagicMock()
        token = CancellationToken.cancelled("stop_early")

        results = _run(
            _platform(tmp_path),
            3,
            tracker=tracker,
            parallel=True,
            failure_policy=FailurePolicy.FAIL_FAST,
            cancellation_token=token,
        )

        payloads = pool.run.call_args.args[0]
        assert len(payloads) == 3
        assert captured["stop_reason"] == "fail_fast"
        assert captured["token"] is token
        assert callable(captured["cancelled_result"])
        # The parent-side verdict matches fail-fast semantics exactly.
        assert captured["stop_on_result"](_failure()) is True
        assert captured["stop_on_result"](_success()) is False
        # Cancelled rows returned by the adapter initialize no tracker run.
        assert [row["state"] for row in results] == ["succeeded", "failed", "cancelled"]
        assert tracker.init_run.call_count == 2
        assert tracker.finish.call_count == 2


# ── Policy resolution and provenance ───────────────────────────


class TestPolicyResolution:
    def test_override_does_not_mutate_cached_settings_instance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(
            execution={"failure_policy": "continue"},
            dataflow={"artifact_root": tmp_path / "artifacts"},
        )
        platform = ExperimentalPlatform(config=settings)
        executor = _ScriptedExecutor([_success(), _failure(), _success()])
        _install_executor(monkeypatch, executor)

        results = _run(platform, 3, failure_policy=FailurePolicy.FAIL_FAST)

        assert results[-1]["state"] == "cancelled"
        assert platform._config_settings() is settings
        assert settings.execution.failure_policy is FailurePolicy.CONTINUE

    def test_override_does_not_mutate_global_settings_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from feature_forge.config import get_settings

        executor = _ScriptedExecutor([_success(), _failure(), _success()])
        _install_executor(monkeypatch, executor)
        platform = _platform(tmp_path)

        _run(platform, 3, failure_policy=FailurePolicy.FAIL_FAST)

        assert get_settings().execution.failure_policy is FailurePolicy.CONTINUE

    def test_tracker_provenance_records_effective_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracker = MagicMock()

        executor = _ScriptedExecutor([_success(), _success()])
        _install_executor(monkeypatch, executor)
        _run(_platform(tmp_path), 2, tracker=tracker, failure_policy=FailurePolicy.FAIL_FAST)
        config = tracker.init_run.call_args.kwargs["config"]
        assert config["failure_policy"] == "fail_fast"

        executor = _ScriptedExecutor([_success(), _success()])
        _install_executor(monkeypatch, executor)
        tracker = MagicMock()
        _run(_platform(tmp_path), 2, tracker=tracker)
        config = tracker.init_run.call_args.kwargs["config"]
        assert config["failure_policy"] == "continue"
