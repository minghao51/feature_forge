"""Tests for the sklearn-compatible public API."""

from __future__ import annotations

import pandas as pd
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
        from feature_forge.pipeline.iterative import IterativePipeline

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
