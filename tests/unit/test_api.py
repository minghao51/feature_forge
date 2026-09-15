"""Tests for the sklearn-compatible public API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from feature_forge.api import FeatureForge
from feature_forge.config import Settings
from feature_forge.llm.base import JSONValue, LLMClient, LLMResponse
from feature_forge.types import FeatureSpec


class StubProvider(LLMClient):
    """Deterministic LLM provider returning canned feature code."""

    def __init__(self) -> None:
        super().__init__(model="stub", api_key="stub")
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "stub"

    async def _do_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
        prompt_meta: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_count += 1
        code = (
            "import pandas as pd\n"
            "def generate_features(df):\n"
            "    result = pd.DataFrame(index=df.index)\n"
            "    result['double_x'] = df['x'] * 2\n"
            "    return result\n"
        )
        return LLMResponse(
            content=code,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

    async def _do_complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        prompt_meta: Mapping[str, Any] | None = None,
    ) -> JSONValue:
        self.call_count += 1
        return {
            "features": [
                {
                    "name": "double_x",
                    "description": "Double x",
                    "rationale": "Simple transform",
                    "code": "result['double_x'] = df['x'] * 2",
                }
            ]
        }


def _make_config() -> Settings:
    return Settings(
        task="regression",
        metric="rmse",
        n_rounds=1,
        max_selected_features=1,
    )


def _make_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    return X, y


class TestFeatureForgeInit:
    def test_is_sklearn_compatible(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert isinstance(fe, BaseEstimator)
        assert isinstance(fe, TransformerMixin)

    def test_default_config(self) -> None:
        fe = FeatureForge(llm_client=StubProvider())
        assert isinstance(fe.config, Settings)

    def test_dict_config(self) -> None:
        fe = FeatureForge(
            config={"task": "regression", "metric": "rmse"},
            llm_client=StubProvider(),
        )
        assert fe.config.task == "regression"

    def test_invalid_mode_fails_closed_before_provider_construction(self) -> None:
        with (
            patch.object(FeatureForge, "_default_llm_client") as provider,
            pytest.raises(ValueError, match="Unknown MALMAS mode"),
        ):
            FeatureForge(config=_make_config(), mode="nonexistent_mode")
        provider.assert_not_called()


class TestFeatureForgeProperties:
    def test_get_feature_names_out_before_fit(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        names = fe.get_feature_names_out()
        assert names == []

    def test_get_feature_names_out_with_input(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        names = fe.get_feature_names_out(["a", "b"])
        assert names == ["a", "b"]

    def test_generated_scripts_before_fit(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert fe.generated_scripts == []

    def test_feature_metadata_before_fit(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert fe.feature_metadata == []


class TestFeatureForgeEdgeCases:
    def test_fit_without_llm_raises(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=None)
        with pytest.raises(RuntimeError, match="requires an LLM client"):
            fe.fit(pd.DataFrame(), pd.Series())

    def test_fit_transform_returns_dataframe(self) -> None:
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        result = fe.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)

    def test_provenance_records_before_fit(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert fe.provenance_records == []

    def test_get_artifacts_before_fit(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert fe.get_artifacts() == {}

    def test_feature_metadata_with_round_artifacts(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.pipeline_result = {
            "round_artifacts": [
                {
                    "specs": [
                        {"name": "f1", "type": "numerical", "agent_name": "unary"},
                        {"name": "f2", "type": "numerical", "agent_name": "cross"},
                    ]
                }
            ]
        }
        meta = fe.feature_metadata
        assert len(meta) == 2

    def test_feature_metadata_aligns_with_successful_code_batches(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.pipeline_result = {
            "round_artifacts": [
                {
                    "generated_codes": ["code-good"],
                    "code_features": [["good"]],
                    "specs": [
                        {"name": "good", "agent_name": "unary"},
                        {"name": "failed", "agent_name": "unary"},
                    ],
                }
            ]
        }

        assert fe.feature_metadata == [
            {
                "names": ["good"],
                "specifications": [{"name": "good", "agent_name": "unary"}],
            }
        ]

    def test_provenance_records_with_artifacts(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.pipeline_result = {
            "round_artifacts": [
                {
                    "round": 0,
                    "agents": ["unary"],
                    "gains": {"f1": 0.05},
                    "generated_code": "def foo(): pass",
                    "specs": [
                        {
                            "name": "f1",
                            "type": "numerical",
                            "agent": "unary",
                        }
                    ],
                }
            ]
        }
        records = fe.provenance_records
        assert len(records) == 1
        assert records[0]["feature_name"] == "f1"

    def test_provenance_records_with_model_specs(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.pipeline_result = {
            "round_artifacts": [
                {
                    "round": 1,
                    "agents": ["unary"],
                    "gains": {"f1": 0.05},
                    "generated_code": "def foo(): pass",
                    "specs": [
                        FeatureSpec(
                            name="f1",
                            type="numerical",
                            transform="x*2",
                            logic="double x",
                            base_columns=["x"],
                            agent_name="unary",
                        )
                    ],
                }
            ]
        }
        records = fe.provenance_records
        assert len(records) == 1
        assert records[0]["feature_name"] == "f1"
        assert records[0]["source_agent"] == "unary"

    def test_get_artifacts_with_round_data(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.selected_features = ["f1"]
        fe.feature_codes = ["code1"]
        fe.pipeline_result = {
            "round_artifacts": [
                {
                    "round": 0,
                    "generated_code": "code1",
                    "specs": [{"name": "f1"}],
                    "baseline_score": 0.8,
                    "gains": {},
                    "agent_gains": {},
                    "agents": [],
                }
            ]
        }
        arts = fe.get_artifacts()
        assert "round_0_generated_code" in arts
        assert arts["selected_features"] == ["f1"]

    def test_fail_on_feature_error_raises(self) -> None:
        from feature_forge.exceptions import CodeExecutionError

        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.config.evaluation.fail_on_feature_error = True
        fe.feature_codes = ["invalid python {"]
        with pytest.raises(CodeExecutionError):
            fe.transform(pd.DataFrame({"x": [1, 2, 3]}))

    def test_transform_applies_only_selected_features(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.selected_features = ["double_x"]
        fe.feature_codes = [
            (
                "import pandas as pd\n"
                "def generate_features(df):\n"
                "    result = pd.DataFrame(index=df.index)\n"
                "    result['double_x'] = df['x'] * 2\n"
                "    result['triple_x'] = df['x'] * 3\n"
                "    return result\n"
            )
        ]
        fe.pipeline_result = {
            "round_artifacts": [
                {
                    "round": 1,
                    "generated_code": fe.feature_codes[0],
                    "selected_features_train": pd.DataFrame(columns=["double_x"]),
                }
            ]
        }

        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        X_out = fe.transform(X)
        assert "double_x" in X_out.columns
        assert "triple_x" not in X_out.columns

    def test_transform_rejects_missing_fitted_input_columns(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.input_features_ = ["x", "required"]

        with pytest.raises(ValueError, match="missing fitted columns"):
            fe.transform(pd.DataFrame({"x": [1.0]}))

    def test_transform_collects_failures_when_not_raising(self) -> None:
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.config.evaluation.fail_on_feature_error = False
        fe.feature_codes = ["invalid python {"]
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        X_out = fe.transform(X)
        assert list(X_out.columns) == ["x"]
        assert len(fe.transform_failures) == 1
