"""Tests for MalmusMethod with structured JSON output."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from feature_forge.llm.base import LLMResponse
from feature_forge.methods import MethodRegistry
from feature_forge.methods.base import BaseMethod
from feature_forge.methods.malmus import (
    FeatureDefinition,
    MalmusMethod,
    StructuredFeatureOutput,
)
from feature_forge.methods.malmus.method import _ensure_json_object


class FakeJsonLLM:
    """Fake LLM that returns structured JSON for complete_json()."""

    def __init__(self, json_response: dict[str, Any]) -> None:
        self.json_response = json_response

    async def complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        return LLMResponse(content=str(self.json_response), model="fake")

    async def complete_json(
        self, messages, schema_description, temperature=0.2, max_tokens=4096
    ) -> dict[str, Any]:
        return self.json_response

    async def _do_complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        return await self.complete(messages, temperature, max_tokens, **kwargs)

    async def _do_complete_json(
        self, messages, schema_description, temperature=0.2, max_tokens=4096
    ) -> dict[str, Any]:
        return await self.complete_json(messages, schema_description, temperature, max_tokens)

    @property
    def provider_name(self) -> str:
        return "fake"


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
    def test_valid_definition(self):
        feat = FeatureDefinition(
            name="sum_ab",
            code="df['a'] + df['b']",
            description="Sum of a and b",
            libraries=["pandas"],
        )
        assert feat.name == "sum_ab"
        assert feat.libraries == ["pandas"]

    def test_default_libraries(self):
        feat = FeatureDefinition(
            name="x",
            code="df['a']",
            description="Pass-through",
        )
        assert feat.libraries == []


class TestStructuredFeatureOutput:
    def test_parse_valid(self):
        parsed = StructuredFeatureOutput.model_validate(SINGLE_SHOT_JSON)
        assert len(parsed.features) == 2
        assert parsed.features[0].name == "sum_ab"

    def test_parse_missing_features_key_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StructuredFeatureOutput.model_validate({"wrong_key": []})


class TestDefsToCode:
    def test_generates_executable_code(self):
        defs = [
            FeatureDefinition(name="sum_ab", code="df['a'] + df['b']", description="sum"),
            FeatureDefinition(name="log_a", code="np.log1p(df['a'])", description="log"),
        ]
        code = MalmusMethod._defs_to_code(defs)
        assert "def generate_features(df):" in code
        assert "result['sum_ab'] = df['a'] + df['b']" in code
        assert "result['log_a'] = np.log1p(df['a'])" in code

    def test_empty_defs_produces_empty_function(self):
        code = MalmusMethod._defs_to_code([])
        assert "def generate_features(df):" in code


class TestMalmusMethodSingleShot:
    def test_init(self):
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm)
        assert baseline.name == "malmus"
        assert baseline.mode == "single_shot"

    def test_fit_transform(self):
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm, n_features=2)
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        result = baseline.fit_transform(X, y)
        assert "sum_ab" in result.columns
        assert "prod_ab" in result.columns
        assert list(result["sum_ab"]) == [5, 7, 9]
        assert list(result["prod_ab"]) == [4, 10, 18]

    def test_transform_before_fit_raises(self):
        llm = FakeJsonLLM({})
        baseline = MalmusMethod(llm_client=llm)
        with pytest.raises(RuntimeError, match="not fitted"):
            baseline.transform(pd.DataFrame())

    def test_artifacts_stored(self):
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        baseline.fit(X, y)
        artifacts = baseline.get_artifacts()
        assert "raw_json" in artifacts
        assert "feature_definitions" in artifacts
        assert len(artifacts["feature_definitions"]) == 2

    def test_feature_metadata(self):
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        baseline.fit(X, y)
        meta = baseline.feature_metadata
        assert len(meta) == 2
        assert meta[0]["method"] == "malmus"
        assert meta[0]["name"] == "sum_ab"

    def test_generated_scripts(self):
        llm = FakeJsonLLM(SINGLE_SHOT_JSON)
        baseline = MalmusMethod(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        baseline.fit(X, y)
        scripts = baseline.generated_scripts
        assert len(scripts) == 1
        assert "def generate_features(df):" in scripts[0]

    def test_invalid_json_raises(self):
        llm = FakeJsonLLM({"not_features": []})
        baseline = MalmusMethod(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        with pytest.raises(Exception, match="invalid structured output"):
            baseline.fit(X, y)


class TestMalmusMethodRegistry:
    def test_malmus_in_builtin_baselines(self):
        baselines = MethodRegistry.get_builtin_methods()
        assert "malmus" in baselines
        assert baselines["malmus"] is MalmusMethod

    def test_malmus_in_all_baselines(self):
        baselines = MethodRegistry.get_all_methods()
        assert "malmus" in baselines

    def test_is_subclass_of_baseline(self):
        assert issubclass(MalmusMethod, BaseMethod)

    def test_existing_baselines_still_registered(self):
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


class TestMalmusMethodIterative:
    """Cover _fit_iterative path (lines 174-271)."""

    @pytest.fixture
    def mock_evaluator(self):
        from unittest.mock import MagicMock

        evaluator = MagicMock()
        evaluator.config.evaluation.sandbox_timeout_seconds = 10.0
        evaluator.config.evaluation.sandbox_max_memory_mb = 512
        evaluator.config.metric = "auc"
        evaluator.config.task = "classification"
        return evaluator

    def test_iterative_fit_keeps_feature(self, mock_evaluator):
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = 0.05
        mock_evaluator.evaluate_features_batch_directional.return_value = {"ratio_ab": 0.05}

        method = MalmusMethod(
            llm_client=llm, mode="iterative", n_features=1, evaluator=mock_evaluator
        )
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        method.fit(X, y)

        assert "ratio_ab" in method._artifacts.get("generated_code", "")
        assert len(method._feature_defs) == 1
        assert method._feature_defs[0].name == "ratio_ab"

    def test_iterative_fit_discards_feature(self, mock_evaluator):
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = -0.02
        mock_evaluator.evaluate_features_batch_directional.return_value = {"ratio_ab": -0.02}

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

    def test_iterative_baseline_score_in_artifacts(self, mock_evaluator):
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.85
        mock_evaluator.evaluate_feature.return_value = 0.03
        mock_evaluator.evaluate_features_batch_directional.return_value = {"ratio_ab": 0.03}

        method = MalmusMethod(
            llm_client=llm, mode="iterative", n_features=1, evaluator=mock_evaluator
        )
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        method.fit(X, y)

        assert method._artifacts["baseline_score"] == 0.85

    def test_iterative_generated_scripts(self, mock_evaluator):
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = 0.05
        mock_evaluator.evaluate_features_batch_directional.return_value = {"ratio_ab": 0.05}

        method = MalmusMethod(
            llm_client=llm, mode="iterative", n_features=1, evaluator=mock_evaluator
        )
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        method.fit(X, y)

        scripts = method.generated_scripts
        assert len(scripts) == 1
        assert "generate_features" in scripts[0]

    def test_iterative_feature_metadata(self, mock_evaluator):
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = 0.05
        mock_evaluator.evaluate_features_batch_directional.return_value = {"ratio_ab": 0.05}

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

    def test_iterative_transform(self, mock_evaluator):
        llm = FakeJsonLLM(ITERATIVE_JSON)
        mock_evaluator.evaluate_baseline.return_value = 0.7
        mock_evaluator.evaluate_feature.return_value = 0.05
        mock_evaluator.evaluate_features_batch_directional.return_value = {"ratio_ab": 0.05}

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

    def test_ensure_json_object_non_dict_raises(self):
        with pytest.raises(Exception, match="Expected JSON object"):
            _ensure_json_object("not_a_dict")
