"""Unit and integration tests for pipelines and sklearn API."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from feature_forge.agents.base import Agent
from feature_forge.api import MALMASFeatureEngineer
from feature_forge.config import Settings
from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.pipeline.ablations import NoRouterPipeline, SingleAgentPipeline
from feature_forge.pipeline.core import CodeGenerator, CorePipeline
from feature_forge.pipeline.iterative import IterativePipeline


class FakeLLM(LLMClient):
    """Fake LLM for deterministic testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="fake", api_key="fake")
        self.responses = responses or []
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    async def complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return LLMResponse(content=resp, model=self.model)

    async def complete_json(self, messages, schema_description, temperature=0.2, max_tokens=4096):
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return json.loads(resp)


class FakeAgent(Agent):
    """Fake agent that returns predetermined specs."""

    @property
    def system_prompt(self) -> str:
        return "fake"

    def __init__(self, specs: list[dict] | None = None) -> None:
        super().__init__(name="fake_agent", config=Settings(), llm_client=FakeLLM())
        self._specs = specs or []

    async def generate(self, X, y, context):
        return self._specs


class TestCodeGenerator:
    @pytest.mark.asyncio
    async def test_generate_code(self):
        llm = FakeLLM(["def generate_features(df): return df"])
        cg = CodeGenerator(llm)
        code = await cg.generate_code([{"name": "f1"}])
        assert "def generate_features" in code


class TestCorePipeline:
    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "b": [0, 1, 0, 1, 0, 1, 0, 1]}
        )
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        return X, y

    @pytest.mark.asyncio
    async def test_run_with_fake_agent(self, sample_data):
        X, y = sample_data
        specs = [
            {
                "name": "sum_ab",
                "type": "numerical",
                "transform": "add",
                "logic": "sum of a and b",
                "base_columns": ["a", "b"],
            }
        ]
        code = """
import pandas as pd
import numpy as np

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    if 'a' in df.columns and 'b' in df.columns:
        result['sum_ab'] = df['a'] + df['b']
        result['sum_ab'] = result['sum_ab'].fillna(0)
    return result
"""
        llm = FakeLLM([code])
        config = Settings(
            task="classification", metric="auc", n_rounds=1, evaluation={"cv_folds": 2}
        )
        pipeline = CorePipeline(config=config, llm_client=llm)
        agent = FakeAgent(specs)
        result = await pipeline.run([agent], X, y)

        assert "features_train" in result
        assert "sum_ab" in result["features_train"].columns

    @pytest.mark.asyncio
    async def test_run_no_specs_returns_empty(self, sample_data):
        X, y = sample_data
        llm = FakeLLM(["[]"])
        config = Settings(n_rounds=1)
        pipeline = CorePipeline(config=config, llm_client=llm)
        agent = FakeAgent([])
        result = await pipeline.run([agent], X, y)
        assert result["features_train"].empty

    @pytest.mark.asyncio
    async def test_run_returns_generated_code(self, sample_data):
        X, y = sample_data
        specs = [
            {
                "name": "sum_ab",
                "type": "numerical",
                "transform": "add",
                "logic": "sum",
                "base_columns": ["a", "b"],
            }
        ]
        code = """
import pandas as pd
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['sum_ab'] = df['a'] + df['b']
    return result
"""
        llm = FakeLLM([code])
        config = Settings(
            task="classification", metric="auc", n_rounds=1, evaluation={"cv_folds": 2}
        )
        pipeline = CorePipeline(config=config, llm_client=llm)
        agent = FakeAgent(specs)
        result = await pipeline.run([agent], X, y)
        assert "generated_code" in result
        assert "def generate_features" in result["generated_code"]


class TestIterativePipeline:
    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame({"a": list(range(20)), "b": [0, 1] * 10})
        y = pd.Series([0, 1] * 10)
        return X, y

    @pytest.mark.asyncio
    async def test_run_one_round(self, sample_data):
        X, y = sample_data
        json_resp = json.dumps(
            [
                {
                    "base_columns": ["a", "b"],
                    "derived_features": [
                        {"name": "sum_ab", "type": "numerical", "transform": "add", "logic": "sum"}
                    ],
                }
            ]
        )
        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['sum_ab'] = df['a'] + df['b']
    return result
"""
        # 6 agents + 1 code generator = 7 calls; provide json for agents, code for generator
        responses = [json_resp] * 6 + [code]
        llm = FakeLLM(responses)
        config = Settings(
            task="classification",
            metric="auc",
            n_rounds=1,
            evaluation={"cv_folds": 2},
            min_effective=1,
        )
        pipeline = IterativePipeline(config=config, llm_client=llm)
        result = await pipeline.run(X, y)

        assert "X_train_enhanced" in result
        assert "round_summaries" in result
        assert len(result["round_summaries"]) == 1

    @pytest.mark.asyncio
    async def test_round_artifacts_no_selected_features_does_not_crash(self, sample_data):
        X, y = sample_data
        json_resp = json.dumps(
            [
                {
                    "base_columns": ["a"],
                    "derived_features": [
                        {
                            "name": "const_feature",
                            "type": "numerical",
                            "transform": "const",
                            "logic": "const",
                        }
                    ],
                }
            ]
        )
        code = """
import pandas as pd
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['const_feature'] = 1
    return result
"""
        responses = [json_resp] * 6 + [code]
        llm = FakeLLM(responses)
        config = Settings(
            task="classification", metric="auc", n_rounds=1, evaluation={"cv_folds": 2}
        )
        pipeline = IterativePipeline(config=config, llm_client=llm)
        result = await pipeline.run(X, y, X_test=X.copy())
        assert len(result["round_artifacts"]) == 1


class TestAsyncEntryPoints:
    @pytest.mark.asyncio
    async def test_malmas_async_fit(self):
        json_resp = json.dumps(
            [
                {
                    "base_columns": ["a", "b"],
                    "derived_features": [
                        {"name": "sum_ab", "type": "numerical", "transform": "add", "logic": "sum"}
                    ],
                }
            ]
        )
        code = """
import pandas as pd
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['sum_ab'] = df['a'] + df['b']
    return result
"""
        llm = FakeLLM([json_resp] * 6 + [code])
        fe = MALMASFeatureEngineer(
            llm_client=llm,
            config=Settings(
                task="classification", metric="auc", n_rounds=1, evaluation={"cv_folds": 2}
            ),
        )
        X = pd.DataFrame({"a": list(range(20)), "b": [0, 1] * 10})
        y = pd.Series([0, 1] * 10)
        result = await fe.async_fit(X, y)
        assert "selected_features" in result

    @pytest.mark.asyncio
    async def test_malmas_sync_fit_inside_running_loop(self):
        json_resp = json.dumps(
            [
                {
                    "base_columns": ["a"],
                    "derived_features": [
                        {
                            "name": "double_a",
                            "type": "numerical",
                            "transform": "mul",
                            "logic": "double",
                        }
                    ],
                }
            ]
        )
        code = """
import pandas as pd
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['double_a'] = df['a'] * 2
    return result
"""
        llm = FakeLLM([json_resp] * 6 + [code])
        fe = MALMASFeatureEngineer(
            llm_client=llm,
            config=Settings(
                task="classification", metric="auc", n_rounds=1, evaluation={"cv_folds": 2}
            ),
        )
        X = pd.DataFrame({"a": list(range(20)), "b": [0, 1] * 10})
        y = pd.Series([0, 1] * 10)
        fe.fit(X, y)
        assert isinstance(fe.selected_features, list)


class TestAblations:
    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame({"a": list(range(20)), "b": [0, 1] * 10})
        y = pd.Series([0, 1] * 10)
        return X, y

    @pytest.mark.asyncio
    async def test_single_agent_pipeline(self, sample_data):
        X, y = sample_data
        json_resp = json.dumps(
            [
                {
                    "base_columns": ["a"],
                    "derived_features": [
                        {
                            "name": "double_a",
                            "type": "numerical",
                            "transform": "multiply",
                            "logic": "double",
                        }
                    ],
                }
            ]
        )
        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['double_a'] = df['a'] * 2
    return result
"""
        llm = FakeLLM([json_resp, code])
        config = Settings(
            task="classification",
            metric="auc",
            n_rounds=1,
            evaluation={"cv_folds": 2},
            min_effective=1,
        )
        pipeline = SingleAgentPipeline("unary", config=config, llm_client=llm)
        result = await pipeline.run(X, y)
        assert "X_train_enhanced" in result

    @pytest.mark.asyncio
    async def test_no_router_pipeline(self, sample_data):
        X, y = sample_data
        code = """
import pandas as pd

def generate_features(df):
    return pd.DataFrame(index=df.index)
"""
        # 6 agents + 1 code generator
        llm = FakeLLM(["[]"] * 6 + [code])
        config = Settings(
            task="classification", metric="auc", n_rounds=1, evaluation={"cv_folds": 2}
        )
        pipeline = NoRouterPipeline(config=config, llm_client=llm)
        result = await pipeline.run(X, y)
        assert "X_train_enhanced" in result


class TestPerAgentContext:
    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "b": [0, 1, 0, 1, 0, 1, 0, 1]}
        )
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        return X, y

    @pytest.mark.asyncio
    async def test_core_pipeline_passes_per_agent_context(self, sample_data):
        X, y = sample_data

        class ContextCapturingAgent(Agent):
            def __init__(self, name):
                super().__init__(name=name, config=Settings(), llm_client=FakeLLM())
                self.received_context = None

            @property
            def system_prompt(self):
                return "fake"

            async def generate(self, X, y, context):
                self.received_context = context
                return []

        agent1 = ContextCapturingAgent("agent1")
        agent2 = ContextCapturingAgent("agent2")
        config = Settings(
            task="classification", metric="auc", n_rounds=1, evaluation={"cv_folds": 2}
        )
        pipeline = CorePipeline(config=config, llm_client=FakeLLM([]))

        await pipeline.run([agent1, agent2], X, y, context=[{"agent": 1}, {"agent": 2}])
        assert agent1.received_context == {"agent": 1}
        assert agent2.received_context == {"agent": 2}


class TestSklearnAPI:
    def test_malmas_feature_engineer_init(self):
        from feature_forge.api import MALMASFeatureEngineer

        llm = FakeLLM([])
        fe = MALMASFeatureEngineer(llm_client=llm)
        assert fe.mode == "full"

    def test_fit_transform_smoke(self):
        from feature_forge.api import MALMASFeatureEngineer

        llm = FakeLLM([])
        fe = MALMASFeatureEngineer(llm_client=llm, mode="no_router")
        assert fe is not None

    def test_transform_executes_cached_code(self):
        from feature_forge.api import MALMASFeatureEngineer

        json_resp = json.dumps(
            [
                {
                    "base_columns": ["a", "b"],
                    "derived_features": [
                        {"name": "sum_ab", "type": "numerical", "transform": "add", "logic": "sum"}
                    ],
                }
            ]
        )
        code = """
import pandas as pd
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['sum_ab'] = df['a'] + df['b']
    return result
"""
        llm = FakeLLM([json_resp] * 6 + [code])
        config = Settings(
            task="classification",
            metric="auc",
            n_rounds=1,
            evaluation={"cv_folds": 2},
            min_effective=1,
        )
        fe = MALMASFeatureEngineer(config=config, llm_client=llm, mode="no_router")
        X = pd.DataFrame({"a": list(range(20)), "b": [0, 1] * 10})
        y = pd.Series([0, 1] * 10)
        fe.fit(X, y)

        X_new = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})
        X_out = fe.transform(X_new)
        assert "sum_ab" in X_out.columns

    def test_transform_records_failures_in_best_effort_mode(self):
        from feature_forge.api import MALMASFeatureEngineer

        cfg = Settings(evaluation={"fail_on_feature_error": False})
        fe = MALMASFeatureEngineer(llm_client=FakeLLM([]), config=cfg)
        fe.feature_codes = ["def generate_features(df):\n    raise ValueError('boom')"]
        X = pd.DataFrame({"a": [1, 2]})
        X_out = fe.transform(X)
        assert list(X_out.columns) == ["a"]
        assert len(fe.transform_failures) == 1

    def test_get_pipeline_modes(self):
        from feature_forge.api import MALMASFeatureEngineer

        fe = MALMASFeatureEngineer(llm_client=FakeLLM([]), mode="no_memory")
        pipeline = fe._get_pipeline()
        assert pipeline.__class__.__name__ == "NoMemoryPipeline"

        fe = MALMASFeatureEngineer(llm_client=FakeLLM([]), mode="no_router")
        pipeline = fe._get_pipeline()
        assert pipeline.__class__.__name__ == "NoRouterPipeline"

        fe = MALMASFeatureEngineer(llm_client=FakeLLM([]), mode="unary")
        pipeline = fe._get_pipeline()
        assert pipeline.__class__.__name__ == "SingleAgentPipeline"

    def test_transform_gracefully_handles_sandbox_failure(self):
        from feature_forge.api import MALMASFeatureEngineer

        fe = MALMASFeatureEngineer(llm_client=FakeLLM([]))
        fe.feature_codes = ["invalid python {"]
        fe.selected_features = ["bad"]
        X = pd.DataFrame({"a": [1, 2, 3]})
        X_out = fe.transform(X)
        assert list(X_out.columns) == ["a"]

    def test_get_feature_names_out(self):
        from feature_forge.api import MALMASFeatureEngineer

        fe = MALMASFeatureEngineer(llm_client=FakeLLM([]))
        fe.selected_features = ["f1", "f2"]
        names = fe.get_feature_names_out(["a", "b"])
        assert names == ["a", "b", "f1", "f2"]
