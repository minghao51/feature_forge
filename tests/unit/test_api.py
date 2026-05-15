"""Tests for the sklearn-compatible public API."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from feature_forge.api import FeatureForge
from feature_forge.config import Settings
from feature_forge.llm.base import LLMClient, LLMResponse


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
        **kwargs,
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
    ) -> dict:
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
        min_effective=1,
    )


def _make_data():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    return X, y


class TestFeatureForgeInit:
    def test_is_sklearn_compatible(self):
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert isinstance(fe, BaseEstimator)
        assert isinstance(fe, TransformerMixin)

    def test_default_config(self):
        fe = FeatureForge(llm_client=StubProvider())
        assert isinstance(fe.config, Settings)

    def test_dict_config(self):
        fe = FeatureForge(
            config={"task": "regression", "metric": "rmse"},
            llm_client=StubProvider(),
        )
        assert fe.config.task == "regression"

    def test_invalid_mode_falls_back_to_full(self):
        fe = FeatureForge(
            config=_make_config(),
            llm_client=StubProvider(),
            mode="nonexistent_mode",
        )
        from feature_forge.methods.malmas.pipeline.iterative import IterativePipeline

        pipeline = fe._get_pipeline()
        assert isinstance(pipeline, IterativePipeline)


class TestFeatureForgeProperties:
    def test_get_feature_names_out_before_fit(self):
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        names = fe.get_feature_names_out()
        assert names == []

    def test_get_feature_names_out_with_input(self):
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        names = fe.get_feature_names_out(["a", "b"])
        assert names == ["a", "b"]

    def test_generated_scripts_before_fit(self):
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert fe.generated_scripts == []

    def test_feature_metadata_before_fit(self):
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert fe.feature_metadata == []


class TestFeatureForgeEdgeCases:
    def test_fit_without_llm_raises(self):
        fe = FeatureForge(config=_make_config(), llm_client=None)
        with pytest.raises(RuntimeError, match="requires an LLM client"):
            fe.fit(pd.DataFrame(), pd.Series())

    def test_fit_transform_returns_dataframe(self):
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        result = fe.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)

    def test_provenance_records_before_fit(self):
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert fe.provenance_records == []

    def test_get_artifacts_before_fit(self):
        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        assert fe.get_artifacts() == {}

    def test_feature_metadata_with_round_artifacts(self):
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

    def test_provenance_records_with_artifacts(self):
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

    def test_get_artifacts_with_round_data(self):
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

    def test_fail_on_feature_error_raises(self):
        from feature_forge.exceptions import CodeExecutionError

        fe = FeatureForge(config=_make_config(), llm_client=StubProvider())
        fe.config.evaluation.fail_on_feature_error = True
        fe.feature_codes = ["invalid python {"]
        with pytest.raises(CodeExecutionError):
            fe.transform(pd.DataFrame({"x": [1, 2, 3]}))
