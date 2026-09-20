"""Focused tests for the failure-policy and cancellation contracts (plan 22 PR 1).

Covers the typed contract slice of ADR 0017: strict ``FailurePolicy``
validation (before any dataset/provider work), environment and YAML
round-trips, the thread-safe parent-process ``CancellationToken``, explicit
``ExperimentResult`` state derivation and its serialized row, the redacted
cancelled ``FailureRecord`` factory, and the ``CaseComputationInput``
serialization guard. Runtime scheduling behavior (sequential fail-fast,
bounded process submission, tracker/lifecycle wiring) is intentionally out of
scope here — it belongs to plan 22 PRs 2-3.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from feature_forge.config import ExecutionPolicyConfig, FailurePolicy, Settings
from feature_forge.contracts.stages import FailureClass, RunState
from feature_forge.experiment.execution import (
    CANCELLED_ERROR_TYPE,
    CancellationToken,
    CaseComputationInput,
    ExperimentCase,
    ExperimentResult,
    cancelled_failure_record,
)

_REPO_ROOT = Path(__file__).parents[2]


class _CodeDefaults(Settings):
    """Code-default Settings; ignores the repo-local settings.yaml.

    Same pattern as ``test_config.py``: a dict() copy of the TypedDict loses
    SettingsConfigDict, so the copy only overrides the allowed key
    ``yaml_file`` to assert code defaults independent of local YAML edits.
    """

    model_config = cast(SettingsConfigDict, dict(Settings.model_config, yaml_file=None))


class TestFailurePolicyConfig:
    def test_policy_vocabulary_matches_plan_22(self) -> None:
        assert FailurePolicy.CONTINUE.value == "continue"
        assert FailurePolicy.FAIL_FAST.value == "fail_fast"

    def test_default_is_continue(self) -> None:
        config = ExecutionPolicyConfig()
        assert config.failure_policy is FailurePolicy.CONTINUE

    def test_invalid_value_rejected(self) -> None:
        with pytest.raises(ValidationError, match="failure_policy"):
            ExecutionPolicyConfig(failure_policy="abort")

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionPolicyConfig(unexpected=True)  # type: ignore[call-arg]

    def test_settings_code_default_is_continue(self) -> None:
        settings = _CodeDefaults()
        assert settings.execution.failure_policy is FailurePolicy.CONTINUE

    def test_nested_constructor_override_round_trips(self) -> None:
        settings = Settings(execution={"failure_policy": "fail_fast"})
        assert settings.execution.failure_policy is FailurePolicy.FAIL_FAST

    def test_env_override_round_trips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FF_EXECUTION__FAILURE_POLICY", "fail_fast")
        settings = _CodeDefaults()
        assert settings.execution.failure_policy is FailurePolicy.FAIL_FAST

    def test_invalid_env_value_fails_at_construction_with_no_filesystem_effects(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Strict validation fires at Settings construction, before any work."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FF_EXECUTION__FAILURE_POLICY", "abort")
        with pytest.raises(ValidationError, match="failure_policy"):
            Settings()
        # No dataset/provider/cache/artifact work left any trace.
        assert list(tmp_path.iterdir()) == []

    def test_committed_yaml_declares_execution_section(self) -> None:
        data = yaml.safe_load((_REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
        assert data["execution"]["failure_policy"] == "continue"


class TestCancellationToken:
    def test_initial_state_is_uncancelled(self) -> None:
        token = CancellationToken()
        assert token.is_cancelled() is False
        assert token.reason is None

    def test_cancel_sets_reason_and_flips_state(self) -> None:
        token = CancellationToken()
        token.cancel("test_stop")
        assert token.is_cancelled() is True
        assert token.reason == "test_stop"

    def test_default_reason_is_operator_request(self) -> None:
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled() is True
        assert token.reason == "operator_request"

    def test_first_reason_wins(self) -> None:
        token = CancellationToken()
        token.cancel("first")
        token.cancel("second")
        assert token.reason == "first"

    def test_cancelled_factory_is_pre_cancelled(self) -> None:
        token = CancellationToken.cancelled("pre_cancelled")
        assert token.is_cancelled() is True
        assert token.reason == "pre_cancelled"

    def test_thread_race_produces_exactly_one_winner(self) -> None:
        token = CancellationToken()
        reasons = [f"reason-{index}" for index in range(8)]
        barrier = threading.Barrier(len(reasons))

        def cancel(reason: str) -> None:
            barrier.wait()
            token.cancel(reason)

        threads = [threading.Thread(target=cancel, args=(reason,)) for reason in reasons]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert token.is_cancelled() is True
        winner = token.reason
        assert winner in reasons
        # Exactly one winner: the first reason is stable across repeated
        # reads and cannot be overwritten by later cancels.
        assert token.reason == winner
        for _ in range(10):
            token.cancel("late_arrival")
        assert token.reason == winner


class TestExperimentResultState:
    @staticmethod
    def _result(**overrides: Any) -> ExperimentResult:
        values: dict[str, Any] = {
            "dataset": "d",
            "method": "m",
            "model": "random_forest",
            "seed": 42,
        }
        values.update(overrides)
        return ExperimentResult(**values)

    def test_state_defaults_to_none_for_back_compat(self) -> None:
        assert self._result().state is None

    def test_none_derives_succeeded_without_error_or_failure(self) -> None:
        assert self._result().resolved_state is RunState.SUCCEEDED

    def test_error_derives_failed(self) -> None:
        assert self._result(error="boom").resolved_state is RunState.FAILED

    def test_failure_record_derives_failed_without_error_string(self) -> None:
        result = self._result(failure=cancelled_failure_record())
        assert result.error is None
        assert result.resolved_state is RunState.FAILED

    def test_explicit_cancelled_survives_without_error(self) -> None:
        assert self._result(state=RunState.CANCELLED).resolved_state is RunState.CANCELLED

    def test_explicit_state_wins_over_derivation(self) -> None:
        result = self._result(error="late failure", state=RunState.CANCELLED)
        assert result.resolved_state is RunState.CANCELLED

    def test_replacement_preserves_state(self) -> None:
        result = replace(self._result(error=None), state=RunState.CANCELLED)
        assert result.resolved_state is RunState.CANCELLED


class TestResultRowSerialization:
    @staticmethod
    def _row(**overrides: Any) -> dict[str, Any]:
        from feature_forge.platform import ExperimentalPlatform

        values: dict[str, Any] = {
            "dataset": "d",
            "method": "m",
            "model": "random_forest",
            "seed": 42,
        }
        values.update(overrides)
        return ExperimentalPlatform._result_to_dict(ExperimentResult(**values))

    def test_row_contains_resolved_state(self) -> None:
        assert self._row(cv_score=0.9)["state"] == "succeeded"
        assert self._row(error="boom")["state"] == "failed"
        assert self._row(failure=cancelled_failure_record())["state"] == "failed"
        assert self._row(state=RunState.CANCELLED)["state"] == "cancelled"

    def test_serialized_state_is_a_plain_string(self) -> None:
        assert type(self._row()["state"]) is str

    def test_existing_keys_unchanged_and_additive_state(self) -> None:
        row = self._row(cv_score=0.9, gain=0.1, run_id="attempt-1")
        assert set(row) == {
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
            # Directional interval fields (ADR 0018 decisions 6-7, plan 23
            # PR 4): additive alongside the legacy ``gain`` key.
            "directional_gain",
            "gain_lower_bound",
            "gain_upper_bound",
            "evaluation_protocol",
        }
        assert row["cv_score"] == 0.9
        assert row["gain"] == 0.1
        assert row["run_id"] == "attempt-1"
        assert row["stages"] == []
        assert row["failure"] is None
        assert row["error"] is None


class TestCancelledFailureRecord:
    def test_factory_shape(self) -> None:
        record = cancelled_failure_record(case_fingerprint="casekey", attempt=2)
        assert record.failure_class is FailureClass.CANCELLED
        assert record.error_type == CANCELLED_ERROR_TYPE == "CaseCancelled"
        assert record.stage == "case"
        assert record.retryable is False
        assert record.attempt == 2
        assert record.cause_types == []

    def test_message_is_a_redacted_stable_template(self) -> None:
        record = cancelled_failure_record(case_fingerprint="fp", reason="policy_fail_fast")
        assert record.message == (
            "Case did not run: cancelled before start (policy_fail_fast) for case 'fp'."
        )

    def test_no_payload_leakage_fields_are_set(self) -> None:
        dump = cancelled_failure_record().model_dump()
        assert dump["cause_types"] == []
        assert "Exception" not in dump["message"]
        assert "prompt" not in dump["message"].lower()
        assert set(dump) == {
            "schema_version",
            "failure_class",
            "error_type",
            "message",
            "stage",
            "retryable",
            "attempt",
            "cause_types",
        }


class TestCaseComputationInputSerializationGuard:
    def test_carries_no_token_or_policy_field(self) -> None:
        names = {f.name for f in dataclass_fields(CaseComputationInput)}
        assert names == {
            "case",
            "settings_data",
            "dataset_overrides",
            "method_overrides",
            "model_overrides",
            "metric_overrides",
        }
        assert not any("token" in name or "cancel" in name or "policy" in name for name in names)

    def test_payload_is_plain_data(self) -> None:
        payload = CaseComputationInput(
            case=ExperimentCase(dataset="d", method="m", model="random_forest", seed=1),
            settings_data=_CodeDefaults().model_dump(),
        )
        data = asdict(payload)
        # Only serializable plain data crosses the process seam.
        assert data["case"]["dataset"] == "d"
        assert data["dataset_overrides"] is None
        assert data["method_overrides"] is None
