"""Unit tests for the agent system."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.methods.malmas.agents import (
    AgentRegistry,
    AggregationConstructAgent,
    CrossCompositionalAgent,
    LocalPatternAgent,
    LocalTransformAgent,
    RouterAgent,
    TemporalFeatureAgent,
    UnaryFeatureAgent,
)


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
        assert specs[0].name == "age_squared"
        assert specs[0].agent_name == "unary"

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
        assert specs[0].name == "f1"

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
            "single_column_dataset": False,
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
            "single_column_dataset": False,
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
        from feature_forge.methods.malmas.agents.base import BaseFeatureAgent

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        desc = BaseFeatureAgent._infer_column_descriptions(df)
        assert "a" in desc
        assert desc["a"]["type"] == "numerical"
        assert desc["a"]["mean"] == 2.0
        assert desc["a"]["min"] == 1.0
        assert desc["a"]["max"] == 3.0

    def test_categorical_columns(self):
        from feature_forge.methods.malmas.agents.base import BaseFeatureAgent

        df = pd.DataFrame({"cat": ["x", "y", "x", "z"]})
        desc = BaseFeatureAgent._infer_column_descriptions(df)
        assert "cat" in desc
        assert desc["cat"]["type"] == "categorical"
        assert desc["cat"]["unique"] == 3

    def test_missing_values(self):
        from feature_forge.methods.malmas.agents.base import BaseFeatureAgent

        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        desc = BaseFeatureAgent._infer_column_descriptions(df)
        assert desc["a"]["missing"] == 1

    def test_empty_dataframe(self):
        from feature_forge.methods.malmas.agents.base import BaseFeatureAgent

        df = pd.DataFrame()
        desc = BaseFeatureAgent._infer_column_descriptions(df)
        assert desc == {}

    def test_cache_key_uses_tuple_not_hash(self):
        from feature_forge.methods.malmas.agents.base import BaseFeatureAgent

        BaseFeatureAgent._column_desc_cache.clear()
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        _ = BaseFeatureAgent._infer_column_descriptions(df)
        cache = BaseFeatureAgent._column_desc_cache
        expected_prefix = (2, 2, ("a", "b"), ("float64", "float64"))
        assert any(k[:4] == expected_prefix for k in cache), (
            f"Expected prefix {expected_prefix} in cache"
        )

    def test_cache_key_changes_with_values(self):
        from feature_forge.methods.malmas.agents.base import BaseFeatureAgent

        BaseFeatureAgent._column_desc_cache.clear()
        df1 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        df2 = pd.DataFrame({"a": [10.0, 20.0], "b": [30.0, 40.0]})
        _ = BaseFeatureAgent._infer_column_descriptions(df1)
        _ = BaseFeatureAgent._infer_column_descriptions(df2)
        assert len(BaseFeatureAgent._column_desc_cache) == 2

    def test_categorical_all_nan_top_is_empty_string(self):
        from feature_forge.methods.malmas.agents.base import BaseFeatureAgent

        df = pd.DataFrame({"cat": [None, None, None]})
        desc = BaseFeatureAgent._infer_column_descriptions(df)
        assert desc["cat"]["top"] == ""


class TestBuildUserPromptAutoEnrichment:
    def test_empty_description_auto_enriched(self):
        from feature_forge.methods.malmas.agents.base import BaseFeatureAgent

        class DummyAgent(BaseFeatureAgent):
            prompt_key = "unary"
            agent_name = "dummy"

        agent = DummyAgent(config=Settings(), llm_client=FakeLLM())
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        prompt = agent._build_user_prompt(df, y, context={"description": {}})
        assert "numerical" in prompt
        assert "mean" in prompt
        assert "a" in prompt

    def test_provided_description_not_overwritten(self):
        from feature_forge.methods.malmas.agents.base import BaseFeatureAgent

        class DummyAgent(BaseFeatureAgent):
            prompt_key = "unary"
            agent_name = "dummy"

        agent = DummyAgent(config=Settings(), llm_client=FakeLLM())
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        provided = {"a": {"name": "a", "type": "special", "info": "hand-crafted"}}
        prompt = agent._build_user_prompt(df, y, context={"description": provided})
        assert "hand-crafted" in prompt
        assert "special" in prompt


class TestRouterAgentEdgeCases:
    """Cover RouterAgent uncovered paths: exclusion rules, LLM selection, strategy dispatch."""

    # ── Exclusion edge cases ──────────────────────────────────

    def test_exclude_by_single_column(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 2,
            "numerical_columns": ["a"],
            "categorical_columns": [],
            "datetime_columns": [],
            "single_column_dataset": True,
            "has_enrich_description": False,
        }
        selected = router._data_driven_selection()
        assert "cross_compositional" not in selected

    def test_exclude_no_numerical_columns(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 3,
            "numerical_columns": [],
            "categorical_columns": ["a", "b"],
            "datetime_columns": ["dt"],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = router._data_driven_selection()
        assert "local_transform" not in selected

    def test_exclude_no_categorical_for_grouping(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 3,
            "numerical_columns": ["a", "b"],
            "categorical_columns": [],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = router._data_driven_selection()
        assert "aggregation" not in selected

    def test_exclude_requires_enrich(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 3,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = router._data_driven_selection()
        assert "local_pattern" not in selected

    def test_requires_enrich_included_when_present(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 3,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": True,
            "single_column_dataset": False,
        }
        selected = router._data_driven_selection()
        assert "local_pattern" in selected

    def test_dataset_char_is_none_returns_all(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = None
        selected = router._data_driven_selection()
        assert len(selected) == len(router.agent_names)

    # ── min_agents padding ────────────────────────────────────

    def test_data_driven_min_agents_padding(self):
        config = Settings()
        config.router.min_agents = 1
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 1,
            "numerical_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = router._data_driven_selection()
        assert len(selected) >= router.min_agents

    def test_perf_driven_min_agents_with_negative_gains(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.agent_performance = {name: [-0.1] for name in router.agent_names}
        selected = router._performance_driven_selection()
        assert len(selected) >= router.min_agents

    # ── Performance-driven empty ──────────────────────────────

    def test_perf_driven_empty_performance(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.agent_performance = {name: [] for name in router.agent_names}
        selected = router._performance_driven_selection()
        assert len(selected) == router.max_agents

    # ── Hybrid padding ────────────────────────────────────────

    def test_hybrid_min_agents_padding(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 1,
            "numerical_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = router._hybrid_selection()
        assert len(selected) >= router.min_agents

    def test_hybrid_capped_at_max_agents(self):
        config = Settings()
        config.router.max_agents = 2
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 10,
            "numerical_columns": ["a", "b", "c", "d"],
            "categorical_columns": ["e", "f"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = router._hybrid_selection()
        assert len(selected) <= 2

    # ── Select agents strategy dispatch ───────────────────────

    @pytest.mark.asyncio
    async def test_select_agents_data_driven_strategy(self):
        config = Settings()
        config.router.strategy = "data_driven"
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 5,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = await router.select_agents(round_idx=1)
        assert isinstance(selected[0], str)

    @pytest.mark.asyncio
    async def test_select_agents_performance_driven_strategy(self):
        config = Settings()
        config.router.strategy = "performance_driven"
        router = RouterAgent(config=config)
        router.agent_performance["unary"] = [0.1]
        router.dataset_characteristics = {
            "total_columns": 5,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = await router.select_agents(round_idx=1)
        assert len(selected) >= 1

    @pytest.mark.asyncio
    async def test_select_agents_llm_strategy_fallback(self):
        config = Settings()
        config.router.strategy = "llm"
        router = RouterAgent(config=config, llm_client=None)
        router.dataset_characteristics = {
            "total_columns": 5,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = await router.select_agents(round_idx=1)
        assert len(selected) >= router.min_agents

    # ── LLM-based selection ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_selection_returns_valid_agents(self):
        class FakeLLM:
            def __init__(self):
                self.model = "fake"

            async def complete_json(
                self, messages, schema_description="", temperature=0.2, max_tokens=4096
            ):
                return json.loads('{"agents":["unary","temporal"]}')

            async def complete(self, messages, **kwargs):
                return LLMResponse(content='{"agents":["unary","temporal"]}', model="fake")

        config = Settings()
        config.router.strategy = "llm"
        router = RouterAgent(config=config, llm_client=FakeLLM())  # type: ignore[arg-type]
        router.dataset_characteristics = {
            "total_columns": 5,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = await router._llm_based_selection(round_idx=0)
        assert "unary" in selected
        assert "temporal" in selected

    @pytest.mark.asyncio
    async def test_llm_selection_invalid_json_falls_back(self):
        class FakeLLM:
            def __init__(self):
                self.model = "fake"

            async def complete_json(
                self, messages, schema_description="", temperature=0.2, max_tokens=4096
            ):
                raise Exception("not json")

            async def complete(self, messages, **kwargs):
                return LLMResponse(content="not json", model="fake")

        config = Settings()
        config.router.strategy = "llm"
        router = RouterAgent(config=config, llm_client=FakeLLM())  # type: ignore[arg-type]
        router.dataset_characteristics = {
            "total_columns": 5,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = await router._llm_based_selection(round_idx=0)
        assert len(selected) >= 1

    @pytest.mark.asyncio
    async def test_llm_selection_no_llm_falls_back(self):
        config = Settings()
        config.router.strategy = "llm"
        router = RouterAgent(config=config, llm_client=None)
        router.dataset_characteristics = {
            "total_columns": 5,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        selected = await router._llm_based_selection(round_idx=0)
        assert len(selected) >= 1

    # ── Build selection context ────────────────────────────────

    def test_build_selection_context_with_chars(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 5,
            "numerical_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        ctx = router._build_selection_context(round_idx=0, description=None, task_description=None)
        assert "Current iteration: Round 1" in ctx
        assert "Total columns: 5" in ctx

    def test_build_selection_context_with_performance(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 3,
            "numerical_columns": ["a"],
            "categorical_columns": ["b"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        router.agent_performance["unary"] = [0.1, 0.2]
        router.agent_performance["temporal"] = []
        ctx = router._build_selection_context(round_idx=2, description=None, task_description=None)
        assert "unary: 0.1500" in ctx or "unary" in ctx
        assert "temporal: No data yet" in ctx

    def test_build_selection_context_with_task_description(self):
        config = Settings()
        router = RouterAgent(config=config)
        router.dataset_characteristics = {
            "total_columns": 3,
            "numerical_columns": ["a"],
            "categorical_columns": ["b"],
            "datetime_columns": [],
            "has_enrich_description": False,
            "single_column_dataset": False,
        }
        ctx = router._build_selection_context(
            round_idx=0, description=None, task_description="Predict churn"
        )
        assert "Task Description: Predict churn" in ctx


class TestLLMConfig:
    def test_max_tokens_default_matches_yaml(self):
        from feature_forge.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.max_tokens == 32768, f"Expected 32768, got {cfg.max_tokens}"
        assert cfg.agent_max_tokens == 8192, f"Expected 8192, got {cfg.agent_max_tokens}"
        assert cfg.codegen_max_tokens == 16384, f"Expected 16384, got {cfg.codegen_max_tokens}"
