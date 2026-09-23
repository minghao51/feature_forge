"""Iteration-failure record contract across iterative methods.

Pins the stable ``IterationRecord`` shape (``gains``/``kept`` always present,
typed ``error`` payload) for caafe, llmfe, and malmas ``feature_failures``.

Uses local test doubles so ``test_malmus.py``/``test_pipeline_core.py`` stay
untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.exceptions import CodeExecutionError, EvaluationError, SandboxTimeoutError
from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.methods.caafe.method import CAAFEMethod
from feature_forge.methods.llmfe.method import LLMFEMethod
from feature_forge.methods.malmas.agents.base import Agent
from feature_forge.methods.malmas.pipeline.core import CorePipeline
from feature_forge.methods.malmas.pipeline.result import PipelineResult
from feature_forge.methods.malmas.types import AgentName
from feature_forge.types import FeatureSpec

GENERATE_FEATURES_CODE = "def generate_features(df):\n    return df"

X_TRAIN = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
Y_TRAIN = pd.Series([0, 1, 0])


class FakeTextLLM(LLMClient):
    """Fake text LLM returning one fixed code response from ``complete``."""

    def __init__(self, content: str = GENERATE_FEATURES_CODE) -> None:
        super().__init__(model="fake", api_key="fake")
        self.content = content

    @property
    def provider_name(self) -> str:
        return "fake"

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        prompt_meta: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del messages, temperature, max_tokens, prompt_meta, kwargs
        return LLMResponse(content=self.content, model="fake")


class _FakeExecutor:
    """Deterministic sandbox double for iteration failure-path tests."""

    def __init__(
        self,
        result: pd.DataFrame | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self._result = result
        self._exc = exc
        self.calls: list[tuple[str, pd.DataFrame]] = []

    def execute(self, code: str, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        self.calls.append((code, df))
        if self._exc is not None:
            raise self._exc
        if self._result is not None:
            return self._result
        return pd.DataFrame(index=df.index)


def _mock_evaluator() -> MagicMock:
    """Mirror the ``test_malmus.py`` evaluator double, with a real config."""
    evaluator = MagicMock()
    evaluator.config = Settings()
    evaluator.evaluate_baseline.return_value = 0.7
    evaluator.evaluate_features_batch.return_value = {}
    return evaluator


def _new_feature_frame(name: str = "new_col") -> pd.DataFrame:
    return pd.DataFrame({name: [0.2, 0.33, 0.43]})


class TestCaafeIterationRecordContract:
    """CAAFE keeps the contract on sandbox, evaluator, and storage failures."""

    def test_sandbox_timeout_records_gains_and_typed_error(self) -> None:
        evaluator = _mock_evaluator()
        method = CAAFEMethod(llm_client=FakeTextLLM(), iterations=1, evaluator=evaluator)
        method.sandbox = cast(Any, _FakeExecutor(exc=SandboxTimeoutError("boom")))

        method.fit(X_TRAIN, Y_TRAIN)

        record = method._artifacts["iterations"][0]
        assert record["gains"] == {}
        assert record["kept"] is False
        assert record["error"] == {"type": "SandboxTimeoutError", "message": "boom"}

    def test_evaluator_failure_records_typed_error(self) -> None:
        evaluator = _mock_evaluator()
        evaluator.evaluate_features_batch.side_effect = EvaluationError("evaluation exploded")
        method = CAAFEMethod(llm_client=FakeTextLLM(), iterations=1, evaluator=evaluator)
        method.sandbox = cast(Any, _FakeExecutor(result=_new_feature_frame()))

        method.fit(X_TRAIN, Y_TRAIN)

        record = method._artifacts["iterations"][0]
        assert record["gains"] == {}
        assert record["kept"] is False
        assert record["error"] == {"type": "EvaluationError", "message": "evaluation exploded"}

    def test_storage_failure_keeps_partial_gains(self) -> None:
        evaluator = _mock_evaluator()
        evaluator.evaluate_features_batch.return_value = {"new_col": 0.05}
        method = CAAFEMethod(llm_client=FakeTextLLM(), iterations=1, evaluator=evaluator)
        method.sandbox = cast(Any, _FakeExecutor(result=_new_feature_frame()))
        method._storage = MagicMock()
        method._storage.store.side_effect = OSError("disk full")

        method.fit(X_TRAIN, Y_TRAIN)

        record = method._artifacts["iterations"][0]
        assert record["gains"] == {"new_col": 0.05}
        assert record["kept"] is False
        assert record["error"] == {"type": "OSError", "message": "disk full"}

    def test_failure_log_includes_error_type(self) -> None:
        evaluator = _mock_evaluator()
        method = CAAFEMethod(llm_client=FakeTextLLM(), iterations=1, evaluator=evaluator)
        method.sandbox = cast(Any, _FakeExecutor(exc=SandboxTimeoutError("boom")))

        with patch("feature_forge.methods.caafe.method.logger") as mock_logger:
            method.fit(X_TRAIN, Y_TRAIN)

        mock_logger.warning.assert_any_call(
            "caafe_iteration_failed",
            iteration=0,
            error_type="SandboxTimeoutError",
            error="boom",
        )

    @pytest.mark.parametrize("path", ["success", "discard", "timeout", "storage"])
    def test_gains_key_always_present(self, path: str) -> None:
        evaluator = _mock_evaluator()
        executor: _FakeExecutor
        storage = MagicMock()
        if path == "success":
            evaluator.evaluate_features_batch.return_value = {"new_col": 0.05}
            executor = _FakeExecutor(result=_new_feature_frame())
        elif path == "discard":
            evaluator.evaluate_features_batch.return_value = {"new_col": -0.02}
            executor = _FakeExecutor(result=_new_feature_frame())
        elif path == "timeout":
            executor = _FakeExecutor(exc=SandboxTimeoutError("timed out"))
        else:
            evaluator.evaluate_features_batch.return_value = {"new_col": 0.05}
            executor = _FakeExecutor(result=_new_feature_frame())
            storage.store.side_effect = OSError("disk full")

        method = CAAFEMethod(llm_client=FakeTextLLM(), iterations=1, evaluator=evaluator)
        method.sandbox = cast(Any, executor)
        if path == "storage":
            method._storage = storage

        method.fit(X_TRAIN, Y_TRAIN)

        record = method._artifacts["iterations"][0]
        assert isinstance(record["gains"], dict)
        assert "kept" in record
        if path in {"timeout", "storage"}:
            assert record["kept"] is False
            assert "error" in record
        else:
            assert "error" not in record


class TestLlmfeIterationRecordContract:
    """LLMFE keeps the same contract as caafe/malmus."""

    def test_sandbox_timeout_records_gains_and_typed_error(self) -> None:
        evaluator = _mock_evaluator()
        method = LLMFEMethod(
            llm_client=FakeTextLLM(), mode="iterative", n_features=1, evaluator=evaluator
        )
        method.sandbox = cast(Any, _FakeExecutor(exc=SandboxTimeoutError("boom")))

        method.fit(X_TRAIN, Y_TRAIN)

        record = method._artifacts["iterations"][0]
        assert record["gains"] == {}
        assert record["kept"] is False
        assert record["error"] == {"type": "SandboxTimeoutError", "message": "boom"}

    def test_failure_log_includes_error_type(self) -> None:
        evaluator = _mock_evaluator()
        method = LLMFEMethod(
            llm_client=FakeTextLLM(), mode="iterative", n_features=1, evaluator=evaluator
        )
        method.sandbox = cast(Any, _FakeExecutor(exc=SandboxTimeoutError("boom")))

        with patch("feature_forge.methods.llmfe.method.logger") as mock_logger:
            method.fit(X_TRAIN, Y_TRAIN)

        mock_logger.warning.assert_any_call(
            "llmfe_iteration_failed",
            iteration=0,
            error_type="SandboxTimeoutError",
            error="boom",
        )

    def test_evaluator_failure_records_typed_error(self) -> None:
        evaluator = _mock_evaluator()
        evaluator.evaluate_features_batch.side_effect = EvaluationError("evaluation exploded")
        method = LLMFEMethod(
            llm_client=FakeTextLLM(), mode="iterative", n_features=1, evaluator=evaluator
        )
        method.sandbox = cast(Any, _FakeExecutor(result=_new_feature_frame()))

        method.fit(X_TRAIN, Y_TRAIN)

        record = method._artifacts["iterations"][0]
        assert record["gains"] == {}
        assert record["kept"] is False
        assert record["error"] == {"type": "EvaluationError", "message": "evaluation exploded"}

    @pytest.mark.parametrize("path", ["success", "discard", "timeout", "evaluator"])
    def test_gains_key_always_present(self, path: str) -> None:
        evaluator = _mock_evaluator()
        executor: _FakeExecutor
        if path == "success":
            evaluator.evaluate_features_batch.return_value = {"new_col": 0.05}
            executor = _FakeExecutor(result=_new_feature_frame())
        elif path == "discard":
            evaluator.evaluate_features_batch.return_value = {"new_col": -0.02}
            executor = _FakeExecutor(result=_new_feature_frame())
        elif path == "timeout":
            executor = _FakeExecutor(exc=SandboxTimeoutError("timed out"))
        else:
            evaluator.evaluate_features_batch.side_effect = EvaluationError("boom")
            executor = _FakeExecutor(result=_new_feature_frame())

        method = LLMFEMethod(
            llm_client=FakeTextLLM(), mode="iterative", n_features=1, evaluator=evaluator
        )
        method.sandbox = cast(Any, executor)

        method.fit(X_TRAIN, Y_TRAIN)

        record = method._artifacts["iterations"][0]
        assert isinstance(record["gains"], dict)
        assert "kept" in record
        if path in {"timeout", "evaluator"}:
            assert record["kept"] is False
            assert "error" in record
        else:
            assert "error" not in record


class FakeLLM(LLMClient):
    """Provider-free LLM double for agent/pipeline construction."""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="fake", api_key="fake")
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def _json_mode_kwargs(self) -> dict[str, Any]:
        return {}

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        self.call_count += 1
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return None

    def _extract_content(self, raw_response: Any) -> str:
        del raw_response
        if not self.responses:
            return "{}"
        return self.responses[(self.call_count - 1) % len(self.responses)]

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        del raw_response
        return 0, 0, 0


class _StubAgent(Agent):
    """Agent double returning a fixed feature-spec list."""

    @property
    def system_prompt(self) -> str:
        return "stub"

    def __init__(self, name: str, specs: list[dict[str, object]]) -> None:
        super().__init__(
            name=AgentName(name), config=Settings(), llm_client=FakeLLM(responses=["{}"])
        )
        self._specs = specs

    async def generate(
        self, X: pd.DataFrame, y: pd.Series, context: dict[str, Any]
    ) -> list[FeatureSpec]:
        del X, y, context
        # The pipeline also accepts raw spec dicts at runtime (see
        # CorePipeline._generate_specs); the declared base type cannot express that.
        return cast("list[FeatureSpec]", self._specs)


class _CountingEvaluator:
    """Minimal evaluator double; gains are always positive."""

    def __init__(self, gain: float = 0.1) -> None:
        self.gain = gain
        self.cv_folds = 5
        self.default_model_name = "random_forest"

    def evaluate_baseline(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        del X_train, y_train
        return 0.5

    def evaluate_feature(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_df: pd.DataFrame,
        baseline_score: float,
    ) -> float:
        del X_train, y_train, feature_df, baseline_score
        return self.gain


class _TypedFailureSandbox:
    """Returns a good feature, but raises ``exc`` for the bad code path."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls: list[str] = []

    def execute(
        self, code: str, df: pd.DataFrame, *, source: str = "unknown", agent_name: str = "unknown"
    ) -> pd.DataFrame:
        del agent_name
        self.calls.append(source)
        if code == "bad_feature":
            raise self.exc
        return pd.DataFrame({"good_feature": [float(i) for i in range(len(df))]}, index=df.index)


class _SchemaDriftSandbox:
    """Train output is int, test output is float (dtype mismatch)."""

    def execute(
        self, code: str, df: pd.DataFrame, *, source: str = "unknown", agent_name: str = "unknown"
    ) -> pd.DataFrame:
        del code, agent_name
        values: list[float] | list[int] = (
            [1.5] * len(df) if source == "malmas_core_test" else [1] * len(df)
        )
        return pd.DataFrame({"good_feature": values}, index=df.index)


def _build_pipeline(sandbox: Any) -> CorePipeline:
    pipeline = CorePipeline(
        config=Settings(task="classification", metric="auc", n_rounds=1),
        llm_client=FakeLLM(responses=["{}"]),
        sandbox=sandbox,
        evaluator=_CountingEvaluator(),  # type: ignore[arg-type]
    )

    async def _gen(
        specs: list[FeatureSpec],
        schema: dict[str, Any] | None = None,
        error_feedback: str | None = None,
    ) -> str:
        del schema, error_feedback
        return specs[0].name

    pipeline.code_generator.generate_code = _gen  # type: ignore[method-assign]
    return pipeline


def _failure_for(result: PipelineResult, feature: str) -> dict[str, Any]:
    matches = [entry for entry in result.feature_failures if entry["feature"] == feature]
    assert len(matches) == 1, result.feature_failures
    return matches[0]


def _good_agent() -> _StubAgent:
    return _StubAgent(
        "good_agent",
        [{"name": "good_feature", "type": "numerical", "transform": "id", "logic": "good"}],
    )


def _bad_agent() -> _StubAgent:
    return _StubAgent(
        "bad_agent",
        [{"name": "bad_feature", "type": "numerical", "transform": "id", "logic": "bad"}],
    )


class TestMalmasFeatureFailureRootCause:
    """malmas ``feature_failures`` entries carry the original exception."""

    @pytest.mark.asyncio
    async def test_train_timeout_carries_typed_error(self) -> None:
        sandbox = _TypedFailureSandbox(SandboxTimeoutError("simulated train timeout"))
        pipeline = _build_pipeline(sandbox)
        X = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        result = await pipeline.run([_good_agent(), _bad_agent()], X, y)

        assert _failure_for(result, "bad_feature") == {
            "feature": "bad_feature",
            "phase": "train",
            "reason_code": "execution_failed",
            "error": {"type": "SandboxTimeoutError", "message": "simulated train timeout"},
        }

    @pytest.mark.asyncio
    async def test_train_code_execution_error_carries_typed_error(self) -> None:
        sandbox = _TypedFailureSandbox(CodeExecutionError("simulated train code error"))
        pipeline = _build_pipeline(sandbox)
        X = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1, 0, 1])

        result = await pipeline.run([_good_agent(), _bad_agent()], X, y)

        assert _failure_for(result, "bad_feature") == {
            "feature": "bad_feature",
            "phase": "train",
            "reason_code": "execution_failed",
            "error": {"type": "CodeExecutionError", "message": "simulated train code error"},
        }

    @pytest.mark.asyncio
    async def test_schema_failure_carries_no_error_key(self) -> None:
        pipeline = _build_pipeline(_SchemaDriftSandbox())
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1])

        result = await pipeline.run([_good_agent()], X, y, X_test=X.copy())

        entry = _failure_for(result, "good_feature")
        assert entry["phase"] == "test"
        assert entry["reason_code"] == "dtype_mismatch"
        assert "error" not in entry
