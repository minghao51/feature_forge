"""Unit tests for the agent system."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from feature_forge.agents import (
    AgentRegistry,
    AggregationConstructAgent,
    CrossCompositionalAgent,
    LocalPatternAgent,
    LocalTransformAgent,
    RouterAgent,
    TemporalFeatureAgent,
    UnaryFeatureAgent,
)
from feature_forge.config import Settings
from feature_forge.llm.base import LLMClient, LLMResponse


class FakeLLM(LLMClient):
    """Fake LLM that returns predetermined responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="fake", api_key="fake")
        self.responses = responses or []
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    async def _do_complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return LLMResponse(content=resp, model=self.model)

    async def _do_complete_json(
        self, messages, schema_description, temperature=0.2, max_tokens=4096
    ):
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return json.loads(resp)


class TestAgentRegistry:
    def test_get_builtin_agents(self):
        agents = AgentRegistry.get_builtin_agents()
        assert "unary" in agents
        assert "cross_compositional" in agents
        assert "aggregation" in agents
        assert "temporal" in agents
        assert "local_transform" in agents
        assert "local_pattern" in agents
        assert len(agents) == 6

    def test_get_all_agents_includes_builtins(self):
        agents = AgentRegistry.get_all_agents()
        assert "unary" in agents


class TestBaseFeatureAgent:
    @pytest.fixture
    def fake_llm(self):
        response = json.dumps(
            [
                {
                    "base_columns": "age",
                    "derived_features": [
                        {
                            "name": "age_squared",
                            "type": "numerical",
                            "transform": "square",
                            "logic": "Age squared captures non-linear effects",
                        }
                    ],
                }
            ]
        )
        return FakeLLM([response])

    @pytest.fixture
    def config(self):
        return Settings()

    @pytest.mark.asyncio
    async def test_generate_parses_features(self, fake_llm, config):
        agent = UnaryFeatureAgent(config=config, llm_client=fake_llm)
        X = pd.DataFrame({"age": [20, 30, 40]})
        y = pd.Series([0, 1, 0])
        specs = await agent.generate(X, y, context={})
        assert len(specs) == 1
        assert specs[0]["name"] == "age_squared"
        assert specs[0]["agent_name"] == "unary"

    @pytest.mark.asyncio
    async def test_generate_with_memory_context(self, fake_llm, config):
        agent = CrossCompositionalAgent(config=config, llm_client=fake_llm)
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        specs = await agent.generate(X, y, context={"memory": "previous success with ratios"})
        assert len(specs) == 1

    def test_parse_response_strips_markdown(self, config):
        agent = UnaryFeatureAgent(config=config, llm_client=FakeLLM())
        content = '```json\n[{"base_columns": "x", "derived_features": [{"name": "f1", "type": "num", "transform": "t", "logic": "l"}]}]\n```'
        specs = agent._parse_response(content)
        assert len(specs) == 1
        assert specs[0]["name"] == "f1"

    def test_parse_response_invalid_json_raises(self, config):
        from feature_forge.exceptions import AgentError

        agent = UnaryFeatureAgent(config=config, llm_client=FakeLLM())
        with pytest.raises(AgentError):
            agent._parse_response("not json")


class TestRouterAgent:
    @pytest.fixture
    def router(self):
        config = Settings()
        return RouterAgent(config=config)

    def test_analyze_dataset(self, router):
        df = pd.DataFrame(
            {
                "num": [1.0, 2.0],
                "cat": ["a", "b"],
                "dt": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            }
        )
        desc = {
            "num": {"type": "numerical"},
            "cat": {"type": "categorical"},
            "dt": {"type": "datetime"},
        }
        chars = router.analyze_dataset(df, desc)
        assert chars["total_columns"] == 3
        assert len(chars["numerical_columns"]) == 1
        assert len(chars["categorical_columns"]) == 1
        assert len(chars["datetime_columns"]) == 1

    def test_data_driven_selection(self, router):
        router.dataset_characteristics = {
            "total_columns": 5,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
        }
        selected = router._data_driven_selection()
        assert "unary" in selected
        assert "cross_compositional" in selected
        assert "aggregation" in selected
        assert "temporal" not in selected

    def test_performance_driven_selection(self, router):
        router.agent_performance["unary"] = [0.1, 0.2]
        router.agent_performance["cross_compositional"] = [-0.05]
        selected = router._performance_driven_selection()
        assert "unary" in selected
        assert "cross_compositional" not in selected

    def test_hybrid_selection(self, router):
        router.dataset_characteristics = {
            "total_columns": 5,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
        }
        router.agent_performance["unary"] = [0.1]
        selected = router._hybrid_selection()
        assert "unary" in selected

    @pytest.mark.asyncio
    async def test_select_agents_warmup(self, router):
        df = pd.DataFrame({"a": [1, 2]})
        selected = await router.select_agents(round_idx=0, df=df)
        assert len(selected) == 6  # All agents during warmup

    @pytest.mark.asyncio
    async def test_select_agents_post_warmup(self, router):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        desc = {"a": {"type": "numerical"}, "b": {"type": "numerical"}}
        selected = await router.select_agents(round_idx=1, df=df, description=desc)
        assert len(selected) >= router.min_agents

    def test_update_performance(self, router):
        router.update_performance("unary", 0.05)
        assert router.agent_performance["unary"] == [0.05]
        router.update_performance("unary", 0.03)
        assert router.agent_performance["unary"] == [0.05, 0.03]

    def test_get_summary(self, router):
        router.update_performance("unary", 0.1)
        summary = router.get_summary()
        assert summary["average_performance"]["unary"] == 0.1
        assert summary["strategy"] == "hybrid"


class TestAllAgentsInstantiate:
    """Smoke test that all agents can be instantiated."""

    def test_unary(self):
        agent = UnaryFeatureAgent(config=Settings(), llm_client=FakeLLM())
        assert agent.name == "unary"

    def test_cross_compositional(self):
        agent = CrossCompositionalAgent(config=Settings(), llm_client=FakeLLM())
        assert agent.name == "cross_compositional"

    def test_aggregation(self):
        agent = AggregationConstructAgent(config=Settings(), llm_client=FakeLLM())
        assert agent.name == "aggregation"

    def test_temporal(self):
        agent = TemporalFeatureAgent(config=Settings(), llm_client=FakeLLM())
        assert agent.name == "temporal"

    def test_local_transform(self):
        agent = LocalTransformAgent(config=Settings(), llm_client=FakeLLM())
        assert agent.name == "local_transform"

    def test_local_pattern(self):
        agent = LocalPatternAgent(config=Settings(), llm_client=FakeLLM())
        assert agent.name == "local_pattern"


class TestInferColumnDescriptions:
    def test_numerical_columns(self):
        from feature_forge.agents.base import BaseFeatureAgent

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        desc = BaseFeatureAgent._infer_column_descriptions(df)
        assert "a" in desc
        assert desc["a"]["type"] == "numerical"
        assert desc["a"]["mean"] == 2.0
        assert desc["a"]["min"] == 1.0
        assert desc["a"]["max"] == 3.0

    def test_categorical_columns(self):
        from feature_forge.agents.base import BaseFeatureAgent

        df = pd.DataFrame({"cat": ["x", "y", "x", "z"]})
        desc = BaseFeatureAgent._infer_column_descriptions(df)
        assert "cat" in desc
        assert desc["cat"]["type"] == "categorical"
        assert desc["cat"]["unique"] == 3

    def test_missing_values(self):
        from feature_forge.agents.base import BaseFeatureAgent

        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        desc = BaseFeatureAgent._infer_column_descriptions(df)
        assert desc["a"]["missing"] == 1

    def test_empty_dataframe(self):
        from feature_forge.agents.base import BaseFeatureAgent

        df = pd.DataFrame()
        desc = BaseFeatureAgent._infer_column_descriptions(df)
        assert desc == {}


class TestBuildUserPromptAutoEnrichment:
    def test_empty_description_auto_enriched(self):
        from feature_forge.agents.base import BaseFeatureAgent

        class DummyAgent(BaseFeatureAgent):
            prompt_filename = "unary.txt"
            agent_name = "dummy"

        agent = DummyAgent(config=Settings(), llm_client=FakeLLM())
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        prompt = agent._build_user_prompt(df, y, context={"description": {}})
        assert "numerical" in prompt
        assert "mean" in prompt
        assert "a" in prompt

    def test_provided_description_not_overwritten(self):
        from feature_forge.agents.base import BaseFeatureAgent

        class DummyAgent(BaseFeatureAgent):
            prompt_filename = "unary.txt"
            agent_name = "dummy"

        agent = DummyAgent(config=Settings(), llm_client=FakeLLM())
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        provided = {"a": {"name": "a", "type": "special", "info": "hand-crafted"}}
        prompt = agent._build_user_prompt(df, y, context={"description": provided})
        assert "hand-crafted" in prompt
        assert "special" in prompt
