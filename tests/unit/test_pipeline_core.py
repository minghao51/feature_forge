"""Unit tests for the core pipeline module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.exceptions import PipelineError
from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.agents.base import Agent
from feature_forge.methods.malmas.pipeline.core import CodeGenerator, CorePipeline
from feature_forge.types import FeatureSpec


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="fake", api_key="fake")
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def _json_mode_kwargs(self) -> dict:
        return {}

    async def _call_api(self, messages, temperature, max_tokens, **kwargs):
        self.call_count += 1
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return None

    def _extract_content(self, raw_response):
        idx = (self.call_count - 1) % len(self.responses)
        return self.responses[idx]

    def _extract_usage(self, raw_response):
        return 0, 0, 0


class TestCodeGenerator:
    @pytest.mark.llm
    @pytest.mark.asyncio
    async def test_generate_code_returns_stripped_response(self):
        code = """
import pandas as pd
def generate_features(df):
    df['feat_a'] = df['num_a'] ** 2
    return df
"""
        llm = FakeLLM(responses=[f"```python\n{code.strip()}\n```"])
        gen = CodeGenerator(llm)
        specs = [
            FeatureSpec(name="feat_a", type="numerical", transform="square", base_columns=["num_a"])
        ]
        result = await gen.generate_code(specs)
        assert "generate_features" in result
        assert "feat_a" in result
        assert llm.call_count == 1

    @pytest.mark.llm
    @pytest.mark.asyncio
    async def test_generate_code_sends_specs_in_prompt(self):
        llm = FakeLLM(
            responses=[
                "def generate_features(df):\n    out = df.copy()\n    out['x'] = 1\n    return out"
            ]
        )
        gen = CodeGenerator(llm)
        specs = [FeatureSpec(name="x", type="numerical")]
        await gen.generate_code(specs)
        call_args = llm.calls[0]
        user_msg = call_args["messages"][1]["content"]
        assert "x" in user_msg

    def test_instantiation(self):
        llm = FakeLLM(responses=[""])
        gen = CodeGenerator(llm)
        assert gen.llm_client is llm

    @pytest.mark.llm
    @pytest.mark.asyncio
    async def test_generate_code_raises_after_two_invalid_attempts(self):
        llm = FakeLLM(responses=["not python code", "still invalid syntax"])
        gen = CodeGenerator(llm)
        specs = [FeatureSpec(name="x", type="numerical")]
        with pytest.raises(PipelineError, match="failed validation after 2 attempts"):
            await gen.generate_code(specs)


class TestCorePipeline:
    @pytest.fixture
    def config(self):
        return Settings()

    @pytest.fixture
    def fake_llm(self):
        return FakeLLM(responses=["df['feat'] = df['num_a'] * 2"])

    def test_instantiation(self, config, fake_llm):
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        assert pipeline.config is config
        assert pipeline.llm_client is fake_llm
        assert isinstance(pipeline.code_generator, CodeGenerator)

    def test_instantiation_with_custom_code_generator(self, config, fake_llm):
        gen = CodeGenerator(fake_llm)
        pipeline = CorePipeline(config=config, llm_client=fake_llm, code_generator=gen)
        assert pipeline.code_generator is gen

    @pytest.mark.asyncio
    async def test_run_empty_agents_returns_empty_features(self, config, fake_llm):
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        result = await pipeline.run(agents=[], X_train=X, y_train=y)
        assert result["specs"] == []
        assert result["features_train"].empty
        assert result["features_test"].empty
        assert result["generated_code"] == ""
        assert result["baseline_score"] == 0.0
        assert result["gains"] == {}
        assert result["agent_gains"] == {}

    @pytest.mark.asyncio
    async def test_run_empty_agents_with_test_set(self, config, fake_llm):
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        X_train = pd.DataFrame({"a": [1, 2, 3]})
        X_test = pd.DataFrame({"a": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        result = await pipeline.run(agents=[], X_train=X_train, y_train=y, X_test=X_test)
        assert result["features_train"].empty
        assert result["features_test"].empty
        assert list(result["features_test"].index) == list(X_test.index)

    def test_prefilter_candidate_columns_removes_constant(self, config, fake_llm):
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        df = pd.DataFrame(
            {
                "const": [1, 1, 1, 1],
                "varying": [1.0, 2.0, 3.0, 4.0],
                "const_nan": [float("nan")] * 4,
            }
        )
        result = pipeline._prefilter_candidate_columns(df)
        assert "const" not in result
        assert "const_nan" not in result
        assert "varying" in result

    def test_prefilter_candidate_columns_empty_df(self, config, fake_llm):
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        result = pipeline._prefilter_candidate_columns(pd.DataFrame())
        assert result == []

    def test_prefilter_candidate_columns_respects_max_candidates(self, fake_llm):
        config = Settings(evaluation={"max_candidate_features": 2})
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
                "c": [100, 200, 300, 400, 500],
                "d": [1000, 2000, 3000, 4000, 5000],
            }
        )
        result = pipeline._prefilter_candidate_columns(df)
        assert len(result) == 2

    def test_prefilter_candidate_columns_all_constant(self, config, fake_llm):
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "x"]})
        result = pipeline._prefilter_candidate_columns(df)
        assert result == []

    def test_eval_single_feature_returns_float_on_success(self):
        evaluator = MagicMock()
        evaluator.evaluate_feature.return_value = 0.05
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        feat_df = pd.DataFrame({"f1": [10, 20, 30]})
        result = CorePipeline._eval_single_feature(evaluator, X, y, feat_df, "f1", 0.5)
        assert isinstance(result, float)
        assert result == 0.05

    def test_eval_single_feature_returns_exception_on_failure(self):
        evaluator = MagicMock()
        evaluator.evaluate_feature.side_effect = ValueError("bad feature")
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        feat_df = pd.DataFrame({"f1": [10, 20, 30]})
        result = CorePipeline._eval_single_feature(evaluator, X, y, feat_df, "f1", 0.5)
        assert isinstance(result, Exception)
        assert "bad feature" in str(result)


class TestColumnDedup:
    def test_dedup_keeps_first_occurrence(self):
        df = pd.DataFrame([[1, 3, 5], [2, 4, 6]], columns=["a", "b", "a"])
        deduped = df.loc[:, ~df.columns.duplicated()]
        assert list(deduped.columns) == ["a", "b"]
        assert deduped["a"].tolist() == [1, 2]


class _StubAgent(Agent):
    @property
    def system_prompt(self) -> str:
        return "stub"

    def __init__(self, name: str, specs: list[dict[str, object]]) -> None:
        super().__init__(name=name, config=Settings(), llm_client=FakeLLM(responses=["{}"]))
        self._specs = specs

    async def generate(self, X, y, context):
        del X, y, context
        return self._specs


class _FaultyTestSandbox:
    def execute(self, code, df, *, source="unknown", agent_name="unknown"):
        if source == "malmas_core_test" and agent_name == "bad_agent":
            raise RuntimeError("simulated test-time failure")
        result = pd.DataFrame(index=df.index)
        if "good_feature" in code:
            result["good_feature"] = 1.0
        if "bad_feature" in code:
            result["bad_feature"] = 2.0
        return result


class _DeterministicCodeGenerator:
    async def generate_code(self, specs, schema=None, error_feedback=None):
        del schema, error_feedback
        names = ",".join(s.name for s in specs)
        return f"def generate_features(df):\n    # {names}\n    return df\n"


class TestCorePipelineXTestFaultTolerance:
    @pytest.mark.asyncio
    async def test_x_test_execution_continues_on_per_agent_failure(self):
        config = Settings(
            task="classification", metric="auc", n_rounds=1, evaluation={"cv_folds": 2}
        )
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            sandbox=_FaultyTestSandbox(),  # type: ignore[arg-type]
            code_generator=_DeterministicCodeGenerator(),  # type: ignore[arg-type]
        )

        # Monkey patch deterministic code bodies keyed by agent specs names.
        async def _gen(specs, schema=None, error_feedback=None):
            del schema, error_feedback
            first = specs[0].name
            if first == "good_feature":
                return "good_feature"
            return "bad_feature"

        pipeline.code_generator.generate_code = _gen  # type: ignore[method-assign]

        good_agent = _StubAgent(
            "good_agent",
            [{"name": "good_feature", "type": "numerical", "transform": "id", "logic": "good"}],
        )
        bad_agent = _StubAgent(
            "bad_agent",
            [{"name": "bad_feature", "type": "numerical", "transform": "id", "logic": "bad"}],
        )

        X_train = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1, 0, 1])
        X_test = X_train.copy()

        result = await pipeline.run([good_agent, bad_agent], X_train, y, X_test=X_test)
        assert "good_feature" in result["features_test"].columns
