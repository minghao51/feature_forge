"""Tests for MalmusMethod with structured JSON output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from feature_forge.exceptions import EvaluationError, SandboxTimeoutError
from feature_forge.llm.base import JSONValue, LLMClient, LLMResponse
from feature_forge.methods import MethodRegistry
from feature_forge.methods.base import BaseMethod
from feature_forge.methods.malmus import (
    FeatureDefinition,
    MalmusMethod,
    StructuredFeatureOutput,
)
from feature_forge.methods.malmus.method import _ensure_json_object


class FakeJsonLLM(LLMClient):
    """Fake LLM that returns structured JSON for complete_json()."""

    def __init__(self, json_response: dict[str, Any]) -> None:
        super().__init__(model="fake", api_key="fake")
        self.json_response = json_response

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
        return LLMResponse(content=str(self.json_response), model="fake")

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        prompt_meta: Mapping[str, Any] | None = None,
    ) -> JSONValue:
        return self.json_response

    async def _do_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
        prompt_meta: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.complete(messages, temperature, max_tokens, **kwargs)

    async def _do_complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        prompt_meta: Mapping[str, Any] | None = None,
    ) -> JSONValue:
        return await self.complete_json(messages, schema_description, temperature, max_tokens)


SINGLE_SHOT_JSON = {
    "features": [
        {
            "name": "sum_ab",
            "code": "df['a'] + df['b']",
            "description": "Sum of columns a and b",
            "libraries": ["pandas"],
        },
        {
            "name": "prod_ab",
            "code": "df['a'] * df['b']",
            "description": "Product of columns a and b",
            "libraries": ["pandas"],
        },
    ]
}

ITERATIVE_JSON = {
    "features": [
        {
            "name": "ratio_ab",
            "code": "df['a'] / (df['b'] + 1)",
            "description": "Ratio of a to b (plus one for safety)",
            "libraries": ["pandas"],
        },
    ]
}


class TestFeatureDefinition:
    def test_valid_definition(self) -> None:
        feat = FeatureDefinition(
            name="sum_ab",
            code="df['a'] + df['b']",
            description="Sum of a and b",
            libraries=["pandas"],
        )
        assert feat.name == "sum_ab"
        assert feat.libraries == ["pandas"]

    def test_default_libraries(self) -> None:
        feat = FeatureDefinition(
            name="x",
            code="df['a']",
            description="Pass-through",
        )
        assert feat.libraries == []


class TestStructuredFeatureOutput:
    def test_parse_valid(self) -> None:
        parsed = StructuredFeatureOutput.model_validate(SINGLE_SHOT_JSON)
        assert len(parsed.features) == 2
        assert parsed.features[0].name == "sum_ab"

    def test_parse_missing_features_key_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StructuredFeatureOutput.model_validate({"wrong_key": []})


class TestDefsToCode:
    def test_generates_executable_code(self) -> None:
        defs = [
            FeatureDefinition(name="sum_ab", code="df['a'] + df['b']", description="sum"),
            FeatureDefinition(name="log_a", code="np.log1p(df['a'])", description="log"),
        ]
        code = MalmusMethod._defs_to_code(defs)
        assert "def generate_features(df):" in code
        assert "result['sum_ab'] = df['a'] + df['b']" in code
        assert "result['log_a'] = np.log1p(df['a'])" in code

    def test_empty_defs_produces_empty_function(self) -> None:
        code = MalmusMethod._defs_to_code([])
        assert "def generate_features(df):" in code


class TestMalmusMethodSingleShot:
    def test_init(self) -> None:
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm)
        assert baseline.name == "malmus"
        assert baseline.mode == "single_shot"

    def test_fit_transform(self) -> None:
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm, n_features=2)
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        result = baseline.fit_transform(X, y)
        assert "sum_ab" in result.columns
        assert "prod_ab" in result.columns
        assert list(result["sum_ab"]) == [5, 7, 9]
        assert list(result["prod_ab"]) == [4, 10, 18]

    def test_transform_before_fit_raises(self) -> None:
        llm = FakeJsonLLM({})
        baseline = MalmusMethod(llm_client=llm)
        with pytest.raises(RuntimeError, match="not fitted"):
            baseline.transform(pd.DataFrame())

    def test_artifacts_stored(self) -> None:
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        baseline.fit(X, y)
        artifacts = baseline.get_artifacts()
        assert "raw_json" in artifacts
        assert "feature_definitions" in artifacts
        assert len(artifacts["feature_definitions"]) == 2

    def test_feature_metadata(self) -> None:
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        baseline.fit(X, y)
        meta = baseline.feature_metadata
        assert len(meta) == 2
        assert meta[0]["method"] == "malmus"
        assert meta[0]["name"] == "sum_ab"

    def test_generated_scripts(self) -> None:
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        baseline.fit(X, y)
        scripts = baseline.generated_scripts
        assert len(scripts) == 1
        assert "def generate_features(df):" in scripts[0]

    def test_invalid_json_raises(self) -> None:
        llm = FakeJsonLLM({"not_features": []})
        baseline = MalmusMethod(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        with pytest.raises(Exception, match="invalid structured output"):
            baseline.fit(X, y)


class TestMalmusMethodRegistry:
    def test_malmus_in_builtin_baselines(self) -> None:
        baselines = MethodRegistry.get_builtin_methods()
        assert "malmus" in baselines
        assert baselines["malmus"] is MalmusMethod

    def test_malmus_in_all_baselines(self) -> None:
        baselines = MethodRegistry.get_all_methods()
        assert "malmus" in baselines

    def test_is_subclass_of_baseline(self) -> None:
        assert issubclass(MalmusMethod, BaseMethod)

    def test_existing_baselines_still_registered(self) -> None:
        baselines = MethodRegistry.get_builtin_methods()
        assert "openfe" in baselines
        assert "caafe" in baselines
        assert "llmfe" in baselines
        assert "malmus" in baselines


ITERATIVE_JSON = {
    "features": [
        {
            "name": "ratio_ab",
            "code": "df['a'] / (df['b'] + 1)",
            "description": "Ratio of a to b",
            "libraries": ["pandas"],
        },
    ]
}


@pytest.fixture
def mock_evaluator() -> MagicMock:
    evaluator = MagicMock()
    evaluator.config.evaluation.sandbox_timeout_seconds = 10.0
    evaluator.config.evaluation.sandbox_max_memory_mb = 512
    evaluator.config.metric = "auc"
    evaluator.config.task = "classification"
    return evaluator


class TestMalmusMethodIterative:
    """Cover _fit_iterative path (lines 174-271)."""

    def test_iterative_fit_keeps_feature(self, mock_evaluator: MagicMock) -> None:
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = 0.05
        mock_evaluator.evaluate_features_batch.return_value = {"ratio_ab": 0.05}

        method = MalmusMethod(
            llm_client=llm, mode="iterative", n_features=1, evaluator=mock_evaluator
        )
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        method.fit(X, y)

        assert "ratio_ab" in method._artifacts.get("generated_code", "")
        assert len(method._feature_defs) == 1
        assert method._feature_defs[0].name == "ratio_ab"

    def test_iterative_fit_discards_feature(self, mock_evaluator: MagicMock) -> None:
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = -0.02
        mock_evaluator.evaluate_features_batch.return_value = {"ratio_ab": -0.02}

        method = MalmusMethod(
            llm_client=llm, mode="iterative", n_features=1, evaluator=mock_evaluator
        )
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        method.fit(X, y)

        assert len(method._feature_defs) == 0
        iterations = method._artifacts.get("iterations", [])
        assert len(iterations) == 1
        assert iterations[0]["gains"] == {}

    def test_iterative_baseline_score_in_artifacts(self, mock_evaluator: MagicMock) -> None:
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.85
        mock_evaluator.evaluate_feature.return_value = 0.03
        mock_evaluator.evaluate_features_batch.return_value = {"ratio_ab": 0.03}

        method = MalmusMethod(
            llm_client=llm, mode="iterative", n_features=1, evaluator=mock_evaluator
        )
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        method.fit(X, y)

        assert method._artifacts["baseline_score"] == 0.85

    def test_iterative_generated_scripts(self, mock_evaluator: MagicMock) -> None:
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = 0.05
        mock_evaluator.evaluate_features_batch.return_value = {"ratio_ab": 0.05}

        method = MalmusMethod(
            llm_client=llm, mode="iterative", n_features=1, evaluator=mock_evaluator
        )
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        method.fit(X, y)

        scripts = method.generated_scripts
        assert len(scripts) == 1
        assert "generate_features" in scripts[0]

    def test_iterative_feature_metadata(self, mock_evaluator: MagicMock) -> None:
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = 0.05
        mock_evaluator.evaluate_features_batch.return_value = {"ratio_ab": 0.05}

        method = MalmusMethod(
            llm_client=llm, mode="iterative", n_features=1, evaluator=mock_evaluator
        )
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        method.fit(X, y)

        meta = method.feature_metadata
        assert len(meta) == 1
        assert meta[0]["name"] == "ratio_ab"
        assert meta[0]["method"] == "malmus"
        assert meta[0]["gain"] == 0.05
        assert meta[0]["kept"] is True

    def test_iterative_transform(self, mock_evaluator: MagicMock) -> None:
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = 0.05
        mock_evaluator.evaluate_features_batch.return_value = {"ratio_ab": 0.05}

        method = MalmusMethod(
            llm_client=llm, mode="iterative", n_features=1, evaluator=mock_evaluator
        )
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0])
        method.fit(X_train, y)

        X_test = pd.DataFrame({"a": [10.0, 20.0], "b": [40.0, 50.0]})
        result = method.transform(X_test)
        assert "ratio_ab" in result.columns
        expected = [round(10.0 / 41.0, 2), round(20.0 / 51.0, 2)]
        assert list(round(result["ratio_ab"], 2)) == expected

    def test_ensure_json_object_non_dict_raises(self) -> None:
        with pytest.raises(Exception, match="Expected JSON object"):
            _ensure_json_object("not_a_dict")


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


def _make_iterative_method(
    mock_evaluator: MagicMock,
    executor: _FakeExecutor | None = None,
    json_response: dict[str, Any] | None = None,
) -> MalmusMethod:
    method = MalmusMethod(
        llm_client=FakeJsonLLM(json_response if json_response is not None else ITERATIVE_JSON),
        mode="iterative",
        n_features=1,
        evaluator=mock_evaluator,
    )
    if executor is not None:
        method.sandbox = executor
    return method


class TestMalmusIterationRecordContract:
    """Failure paths keep ``gains``/``kept`` and expose a typed root cause."""

    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.Series([0, 1, 0])

    def test_sandbox_timeout_records_gains_and_typed_error(self, mock_evaluator: MagicMock) -> None:
        mock_evaluator.evaluate_baseline.return_value = 0.7
        timeout = SandboxTimeoutError("Sandbox execution timed out after 5.0s")
        method = _make_iterative_method(mock_evaluator, _FakeExecutor(exc=timeout))

        method.fit(self.X, self.y)

        iterations = method._artifacts["iterations"]
        assert len(iterations) == 1
        record = iterations[0]
        assert record["gains"] == {}
        assert record["kept"] is False
        assert record["error"] == {
            "type": "SandboxTimeoutError",
            "message": "Sandbox execution timed out after 5.0s",
        }
        # Downstream consumers never hit a secondary KeyError.
        assert method.feature_metadata == []
        assert method.provenance_records == []
        assert method.generated_scripts == []

    def test_timeout_log_retains_original_message(self, mock_evaluator: MagicMock) -> None:
        mock_evaluator.evaluate_baseline.return_value = 0.7
        timeout = SandboxTimeoutError("Sandbox execution timed out after 5.0s")
        method = _make_iterative_method(mock_evaluator, _FakeExecutor(exc=timeout))

        with patch("feature_forge.methods.malmus.method.logger") as mock_logger:
            method.fit(self.X, self.y)

        mock_logger.warning.assert_called_once_with(
            "malmus_iteration_failed",
            iteration=0,
            error_type="SandboxTimeoutError",
            error="Sandbox execution timed out after 5.0s",
        )

    def test_parse_failure_records_typed_error(self, mock_evaluator: MagicMock) -> None:
        mock_evaluator.evaluate_baseline.return_value = 0.7
        method = _make_iterative_method(mock_evaluator, json_response={"not_features": []})

        method.fit(self.X, self.y)

        record = method._artifacts["iterations"][0]
        assert record["gains"] == {}
        assert record["kept"] is False
        assert record["error"]["type"] == "EvaluationError"
        assert "invalid structured output" in record["error"]["message"]

    def test_evaluator_failure_records_typed_error(self, mock_evaluator: MagicMock) -> None:
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_features_batch.side_effect = EvaluationError(
            "CV evaluator exploded"
        )
        executor = _FakeExecutor(result=pd.DataFrame({"ratio_ab": [0.2, 0.33, 0.43]}))
        method = _make_iterative_method(mock_evaluator, executor)

        method.fit(self.X, self.y)

        record = method._artifacts["iterations"][0]
        assert record["gains"] == {}
        assert record["kept"] is False
        assert record["error"] == {
            "type": "EvaluationError",
            "message": "CV evaluator exploded",
        }

    def test_storage_failure_keeps_partial_gains(self, mock_evaluator: MagicMock) -> None:
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_features_batch.return_value = {"ratio_ab": 0.05}
        executor = _FakeExecutor(result=pd.DataFrame({"ratio_ab": [0.2, 0.33, 0.43]}))
        method = _make_iterative_method(mock_evaluator, executor)
        method._storage = MagicMock()
        method._storage.store.side_effect = OSError("disk full")

        method.fit(self.X, self.y)

        record = method._artifacts["iterations"][0]
        assert record["gains"] == {"ratio_ab": 0.05}
        assert record["kept"] is False
        assert record["error"] == {"type": "OSError", "message": "disk full"}

    @pytest.mark.parametrize("path", ["success", "discard", "timeout", "parse", "evaluator"])
    def test_gains_key_always_present(self, mock_evaluator: MagicMock, path: str) -> None:
        mock_evaluator.evaluate_baseline.return_value = 0.7
        executor: _FakeExecutor | None = None
        json_response: dict[str, Any] | None = None

        if path == "success":
            mock_evaluator.evaluate_features_batch.return_value = {"ratio_ab": 0.05}
            executor = _FakeExecutor(result=pd.DataFrame({"ratio_ab": [0.2, 0.33, 0.43]}))
        elif path == "discard":
            mock_evaluator.evaluate_features_batch.return_value = {"ratio_ab": -0.02}
            executor = _FakeExecutor(result=pd.DataFrame({"ratio_ab": [0.2, 0.33, 0.43]}))
        elif path == "timeout":
            executor = _FakeExecutor(exc=SandboxTimeoutError("timed out"))
        elif path == "parse":
            json_response = {"not_features": []}
        else:
            mock_evaluator.evaluate_features_batch.side_effect = EvaluationError("boom")
            executor = _FakeExecutor(result=pd.DataFrame({"ratio_ab": [0.2, 0.33, 0.43]}))

        method = _make_iterative_method(mock_evaluator, executor, json_response)
        method.fit(self.X, self.y)

        record = method._artifacts["iterations"][0]
        assert isinstance(record["gains"], dict)
        assert "kept" in record
        if path in {"timeout", "parse", "evaluator"}:
            assert record["kept"] is False
            assert "error" in record
        else:
            assert "error" not in record
