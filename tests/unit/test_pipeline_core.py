"""Unit tests for the core pipeline module."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

from feature_forge.config import EvaluationConfig, Settings
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import PipelineError
from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.agents.base import Agent
from feature_forge.methods.malmas.pipeline import core as core_module
from feature_forge.methods.malmas.pipeline.codegen import CodeGenerator
from feature_forge.methods.malmas.pipeline.core import CorePipeline
from feature_forge.methods.malmas.types import AgentName
from feature_forge.types import FeatureSpec


class FakeLLM(LLMClient):
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
        self.call_count += 1
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return None

    def _extract_content(self, raw_response: Any) -> str:
        idx = (self.call_count - 1) % len(self.responses)
        return self.responses[idx]

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        return 0, 0, 0


class TestCodeGenerator:
    @pytest.mark.llm
    @pytest.mark.asyncio
    async def test_generate_code_returns_stripped_response(self) -> None:
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
    async def test_generate_code_sends_specs_in_prompt(self) -> None:
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

    def test_instantiation(self) -> None:
        llm = FakeLLM(responses=[""])
        gen = CodeGenerator(llm)
        assert gen.llm_client is llm

    @pytest.mark.llm
    @pytest.mark.asyncio
    async def test_generate_code_raises_after_two_invalid_attempts(self) -> None:
        llm = FakeLLM(responses=["not python code", "still invalid syntax"])
        gen = CodeGenerator(llm)
        specs = [FeatureSpec(name="x", type="numerical")]
        with pytest.raises(PipelineError, match="failed validation after 2 attempts"):
            await gen.generate_code(specs)


class TestCorePipeline:
    @pytest.fixture
    def config(self) -> Settings:
        return Settings()

    @pytest.fixture
    def fake_llm(self) -> FakeLLM:
        return FakeLLM(responses=["df['feat'] = df['num_a'] * 2"])

    def test_instantiation(self, config: Settings, fake_llm: FakeLLM) -> None:
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        assert pipeline.config is config
        assert pipeline.llm_client is fake_llm
        assert isinstance(pipeline.code_generator, CodeGenerator)

    def test_instantiation_with_custom_code_generator(
        self, config: Settings, fake_llm: FakeLLM
    ) -> None:
        gen = CodeGenerator(fake_llm)
        pipeline = CorePipeline(config=config, llm_client=fake_llm, code_generator=gen)
        assert pipeline.code_generator is gen

    def test_eval_kit_param(self, config: Settings, fake_llm: FakeLLM) -> None:
        from feature_forge.evaluation.kit import EvaluationKit

        kit = EvaluationKit.from_settings(config)
        pipeline = CorePipeline(config=config, llm_client=fake_llm, eval_kit=kit)
        assert pipeline.evaluator is kit.evaluator
        assert pipeline.sandbox is kit.sandbox

    def test_backward_compat_evaluator_param(self, config: Settings, fake_llm: FakeLLM) -> None:
        from feature_forge.evaluation.cv import CVEvaluator

        evaluator = CVEvaluator(config)
        pipeline = CorePipeline(config=config, llm_client=fake_llm, evaluator=evaluator)
        assert pipeline.evaluator is evaluator
        assert isinstance(pipeline.sandbox, SandboxedExecutor)

    @pytest.mark.asyncio
    async def test_run_empty_agents_returns_empty_features(
        self, config: Settings, fake_llm: FakeLLM
    ) -> None:
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        result = await pipeline.run(agents=[], X_train=X, y_train=y)
        assert result.specs == []
        assert result.features_train.empty
        assert result.features_test.empty
        assert result.generated_code == ""
        assert result.baseline_score == 0.0
        assert result.gains == {}
        assert result.agent_gains == {}

    @pytest.mark.asyncio
    async def test_run_empty_agents_with_test_set(
        self, config: Settings, fake_llm: FakeLLM
    ) -> None:
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        X_train = pd.DataFrame({"a": [1, 2, 3]})
        X_test = pd.DataFrame({"a": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        result = await pipeline.run(agents=[], X_train=X_train, y_train=y, X_test=X_test)
        assert result.features_train.empty
        assert result.features_test.empty
        assert list(result.features_test.index) == list(X_test.index)

    def test_prefilter_candidate_columns_removes_constant(
        self, config: Settings, fake_llm: FakeLLM
    ) -> None:
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

    def test_prefilter_candidate_columns_empty_df(
        self, config: Settings, fake_llm: FakeLLM
    ) -> None:
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        result = pipeline._prefilter_candidate_columns(pd.DataFrame())
        assert result == []

    def test_prefilter_candidate_columns_respects_max_candidates(self, fake_llm: FakeLLM) -> None:
        config = Settings(evaluation=EvaluationConfig(max_candidate_features=2))
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

    def test_prefilter_candidate_columns_all_constant(
        self, config: Settings, fake_llm: FakeLLM
    ) -> None:
        pipeline = CorePipeline(config=config, llm_client=fake_llm)
        df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "x"]})
        result = pipeline._prefilter_candidate_columns(df)
        assert result == []

    def test_eval_single_feature_returns_float_on_success(self) -> None:
        evaluator = MagicMock()
        evaluator.evaluate_feature.return_value = 0.05
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        feat_df = pd.DataFrame({"f1": [10, 20, 30]})
        result = CorePipeline._eval_single_feature(evaluator, X, y, feat_df, "f1", 0.5)
        assert isinstance(result, float)
        assert result == 0.05

    def test_eval_single_feature_returns_exception_on_failure(self) -> None:
        evaluator = MagicMock()
        evaluator.evaluate_feature.side_effect = ValueError("bad feature")
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        feat_df = pd.DataFrame({"f1": [10, 20, 30]})
        result = CorePipeline._eval_single_feature(evaluator, X, y, feat_df, "f1", 0.5)
        assert isinstance(result, Exception)
        assert "bad feature" in str(result)


class TestColumnDedup:
    def test_dedup_keeps_first_occurrence(self) -> None:
        df = pd.DataFrame([[1, 3, 5], [2, 4, 6]], columns=["a", "b", "a"])
        deduped = df.loc[:, ~df.columns.duplicated()]
        assert list(deduped.columns) == ["a", "b"]
        assert deduped["a"].tolist() == [1, 2]


class _StubAgent(Agent):
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


class _FaultyTestSandbox:
    def execute(
        self, code: str, df: pd.DataFrame, *, source: str = "unknown", agent_name: str = "unknown"
    ) -> pd.DataFrame:
        if source == "malmas_core_test" and agent_name == "bad_agent":
            raise RuntimeError("simulated test-time failure")
        result = pd.DataFrame(index=df.index)
        if "good_feature" in code:
            result["good_feature"] = 1.0
        if "bad_feature" in code:
            result["bad_feature"] = 2.0
        return result


class _DeterministicCodeGenerator:
    async def generate_code(
        self,
        specs: list[FeatureSpec],
        schema: dict[str, Any] | None = None,
        error_feedback: str | None = None,
    ) -> str:
        del schema, error_feedback
        names = ",".join(s.name for s in specs)
        return f"def generate_features(df):\n    # {names}\n    return df\n"


class TestCorePipelineXTestFaultTolerance:
    @pytest.mark.asyncio
    async def test_x_test_execution_continues_on_per_agent_failure(self) -> None:
        config = Settings(
            task="classification", metric="auc", n_rounds=1, evaluation=EvaluationConfig(cv_folds=2)
        )
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            # duck-typed test doubles, not SandboxedExecutor/CodeGenerator subclasses
            sandbox=_FaultyTestSandbox(),  # type: ignore[arg-type]
            code_generator=_DeterministicCodeGenerator(),  # type: ignore[arg-type]
        )

        # Monkey patch deterministic code bodies keyed by agent specs names.
        async def _gen(
            specs: list[FeatureSpec],
            schema: dict[str, Any] | None = None,
            error_feedback: str | None = None,
        ) -> str:
            del schema, error_feedback
            first = specs[0].name
            if first == "good_feature":
                return "good_feature"
            return "bad_feature"

        # plain closure with a narrower signature than the bound method
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
        assert "good_feature" in result.features_test.columns


class _SchemaSandbox:
    def __init__(self, *, test_dtype_mismatch: bool = False) -> None:
        self.calls: list[str] = []
        self.test_dtype_mismatch = test_dtype_mismatch

    def execute(
        self, code: str, df: pd.DataFrame, *, source: str = "unknown", agent_name: str = "unknown"
    ) -> pd.DataFrame:
        del code, agent_name
        self.calls.append(source)
        values: list[float] | list[int]
        if source == "malmas_core_test" and self.test_dtype_mismatch:
            values = [1.5] * len(df)
        else:
            values = [1] * len(df)
        return pd.DataFrame({"f1": values, "f2": [2] * len(df)}, index=df.index)


class TestSuccessfulCodeReplay:
    @pytest.mark.asyncio
    async def test_duplicate_successful_code_executes_once_per_partition(self) -> None:
        config = Settings(
            n_rounds=1,
            max_selected_features=2,
            evaluation=EvaluationConfig(cv_folds=2),
        )
        sandbox = _SchemaSandbox()
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            sandbox=sandbox,  # type: ignore[arg-type]
            evaluator=_CountingEvaluator(),  # type: ignore[arg-type]
        )

        async def _same_code(*args: object, **kwargs: object) -> str:
            del args, kwargs
            return "same-code"

        pipeline.code_generator.generate_code = _same_code  # type: ignore[method-assign]
        agents: list[Agent] = [
            _StubAgent(
                "a1",
                [{"name": "f1", "type": "numerical", "transform": "id", "logic": "f1"}],
            ),
            _StubAgent(
                "a2",
                [{"name": "f2", "type": "numerical", "transform": "id", "logic": "f2"}],
            ),
        ]
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1])

        result = await pipeline.run(agents, X, y, X_test=X.copy())

        assert sandbox.calls.count("malmas_core_train") == 1
        assert sandbox.calls.count("malmas_core_test") == 1
        assert result.generated_codes == ["same-code"]
        assert result.code_features == [["f1", "f2"]]
        assert result.feature_failures == []

    @pytest.mark.asyncio
    async def test_test_schema_mismatch_is_reported_per_feature(self) -> None:
        config = Settings(n_rounds=1, evaluation=EvaluationConfig(cv_folds=2))
        sandbox = _SchemaSandbox(test_dtype_mismatch=True)
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            sandbox=sandbox,  # type: ignore[arg-type]
            evaluator=_CountingEvaluator(),  # type: ignore[arg-type]
        )

        async def _code(*args: object, **kwargs: object) -> str:
            del args, kwargs
            return "one-code"

        pipeline.code_generator.generate_code = _code  # type: ignore[method-assign]
        agent = _StubAgent(
            "a1",
            [{"name": "f1", "type": "numerical", "transform": "id", "logic": "f1"}],
        )
        X = pd.DataFrame({"a": [0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1])

        result = await pipeline.run([agent], X, y, X_test=X.copy())

        assert {tuple(item.values()) for item in result.feature_failures} >= {
            ("f1", "test", "dtype_mismatch")
        }
        assert result.selected_features_test.empty


class _CountingEvaluator:
    def __init__(self, gain: float = 0.1) -> None:
        self.calls = 0
        self.gain = gain
        self.cv_folds = 5
        self.default_model_name = "random_forest"

    def evaluate_baseline(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        del X_train, y_train
        self.calls += 1
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


class TestCorePipelineBaselineCache:
    def test_baseline_cache_key_changes_when_train_schema_changes(self) -> None:
        config = Settings(evaluation=EvaluationConfig(feature_eval_backend="threading"))
        evaluator = _CountingEvaluator()
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            # duck-typed test double; CVEvaluator is a concrete class, not a Protocol
            evaluator=evaluator,  # type: ignore[arg-type]
        )

        X_train = pd.DataFrame({"a": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])
        features = pd.DataFrame({"f1": [0.2, 0.3, 0.4, 0.5]})

        pipeline._evaluate_and_select(
            features_train=features,
            features_test=pd.DataFrame(),
            X_train=X_train,
            y_train=y_train,
            X_test=None,
            all_specs=[],
            agents=[],
            code="",
        )
        assert evaluator.calls == 1

        # In-place schema mutation should invalidate baseline cache key.
        X_train["b"] = [10, 20, 30, 40]

        pipeline._evaluate_and_select(
            features_train=features,
            features_test=pd.DataFrame(),
            X_train=X_train,
            y_train=y_train,
            X_test=None,
            all_specs=[],
            agents=[],
            code="",
        )
        assert evaluator.calls == 2

    def test_baseline_cache_key_tracks_values_fold_and_model_identity(self) -> None:
        config = Settings(evaluation=EvaluationConfig(feature_eval_backend="threading"))
        evaluator = _CountingEvaluator()
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            evaluator=evaluator,  # type: ignore[arg-type]
        )
        X_train = pd.DataFrame({"a": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])

        original = pipeline._baseline_cache_key(X_train, y_train)
        X_train.loc[0, "a"] = 99
        assert pipeline._baseline_cache_key(X_train, y_train) != original

        x_mutated = pipeline._baseline_cache_key(X_train, y_train)
        y_train.iloc[0] = 1
        assert pipeline._baseline_cache_key(X_train, y_train) != x_mutated

        data_mutated = pipeline._baseline_cache_key(X_train, y_train)
        evaluator.cv_folds = 3
        assert pipeline._baseline_cache_key(X_train, y_train) != data_mutated

        folds_mutated = pipeline._baseline_cache_key(X_train, y_train)
        evaluator.default_model_name = "mlp"
        assert pipeline._baseline_cache_key(X_train, y_train) != folds_mutated


class TestMetricDirectionSelection:
    def test_minimize_metric_keeps_negative_raw_gain(self) -> None:
        config = Settings(task="regression", metric="rmse", max_selected_features=1)
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            evaluator=_CountingEvaluator(gain=-0.1),  # type: ignore[arg-type]
        )
        X = pd.DataFrame({"a": [1, 2, 3, 4]})
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        features = pd.DataFrame({"better": [1.0, 1.5, 3.0, 3.5]})

        result = pipeline._evaluate_and_select(features, pd.DataFrame(), X, y, None, [], [], "")

        assert list(result.selected_features_train.columns) == ["better"]


class TestFeatureEvalBackendSelection:
    def test_parallel_uses_threading_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = Settings(
            evaluation=EvaluationConfig(feature_eval_backend="threading", max_cv_workers=2),
        )
        evaluator = _CountingEvaluator()
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            # duck-typed test double; CVEvaluator is a concrete class, not a Protocol
            evaluator=evaluator,  # type: ignore[arg-type]
        )
        X_train = pd.DataFrame({"a": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])
        features = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1.1, 1.2, 1.3, 1.4]})

        captured: dict[str, object] = {}

        def _runner(tasks: list[Callable[[], object]]) -> list[object]:
            return [task() for task in tasks]

        def _fake_parallel(
            *, n_jobs: int, backend: str
        ) -> Callable[[list[Callable[[], object]]], list[object]]:
            captured["n_jobs"] = n_jobs
            captured["backend"] = backend
            return _runner

        monkeypatch.setattr(core_module, "Parallel", _fake_parallel)
        monkeypatch.setattr(core_module, "delayed", lambda fn: lambda *a, **k: lambda: fn(*a, **k))

        result = pipeline._evaluate_and_select(
            features_train=features,
            features_test=pd.DataFrame(),
            X_train=X_train,
            y_train=y_train,
            X_test=None,
            all_specs=[],
            agents=[],
            code="",
        )
        assert captured["backend"] == "threading"
        assert captured["n_jobs"] == 2
        assert "f1" in result.gains

    def test_parallel_uses_loky_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Without Intel acceleration, the user's loky choice is honored (it only
        # falls back to threading for very large matrices).
        monkeypatch.setattr(core_module, "is_intel_active", lambda: False)
        config = Settings(
            evaluation=EvaluationConfig(feature_eval_backend="loky", max_cv_workers=3),
        )
        evaluator = _CountingEvaluator()
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            # duck-typed test double; CVEvaluator is a concrete class, not a Protocol
            evaluator=evaluator,  # type: ignore[arg-type]
        )
        X_train = pd.DataFrame({"a": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])
        features = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1.1, 1.2, 1.3, 1.4]})

        captured: dict[str, object] = {}

        def _runner(tasks: list[Callable[[], object]]) -> list[object]:
            return [task() for task in tasks]

        def _fake_parallel(
            *, n_jobs: int, backend: str
        ) -> Callable[[list[Callable[[], object]]], list[object]]:
            captured["n_jobs"] = n_jobs
            captured["backend"] = backend
            return _runner

        monkeypatch.setattr(core_module, "Parallel", _fake_parallel)
        monkeypatch.setattr(core_module, "delayed", lambda fn: lambda *a, **k: lambda: fn(*a, **k))

        result = pipeline._evaluate_and_select(
            features_train=features,
            features_test=pd.DataFrame(),
            X_train=X_train,
            y_train=y_train,
            X_test=None,
            all_specs=[],
            agents=[],
            code="",
        )
        assert captured["backend"] == "loky"
        assert captured["n_jobs"] == 3
        assert "f2" in result.gains

    def test_intel_acceleration_forces_threading_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When Intel acceleration is engaged, any fork-based backend must be
        # downgraded to threading: forking after OpenMP (libiomp5) is initialized
        # can deadlock in the child process. Boosting estimators stay at n_jobs=1.
        monkeypatch.setattr(core_module, "is_intel_active", lambda: True)
        config = Settings(
            evaluation=EvaluationConfig(feature_eval_backend="loky", max_cv_workers=2),
        )
        evaluator = _CountingEvaluator()
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            # duck-typed test double; CVEvaluator is a concrete class, not a Protocol
            evaluator=evaluator,  # type: ignore[arg-type]
        )
        X_train = pd.DataFrame({"a": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])
        features = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1.1, 1.2, 1.3, 1.4]})

        captured: dict[str, object] = {}

        def _runner(tasks: list[Callable[[], object]]) -> list[object]:
            return [task() for task in tasks]

        def _fake_parallel(
            *, n_jobs: int, backend: str
        ) -> Callable[[list[Callable[[], object]]], list[object]]:
            captured["n_jobs"] = n_jobs
            captured["backend"] = backend
            return _runner

        monkeypatch.setattr(core_module, "Parallel", _fake_parallel)
        monkeypatch.setattr(core_module, "delayed", lambda fn: lambda *a, **k: lambda: fn(*a, **k))

        pipeline._evaluate_and_select(
            features_train=features,
            features_test=pd.DataFrame(),
            X_train=X_train,
            y_train=y_train,
            X_test=None,
            all_specs=[],
            agents=[],
            code="",
        )
        assert captured["backend"] == "threading"
        assert captured["n_jobs"] == 2

    def test_baseline_cache_key_changes_when_target_values_change(self) -> None:
        config = Settings(evaluation=EvaluationConfig(feature_eval_backend="threading"))
        evaluator = _CountingEvaluator()
        pipeline = CorePipeline(
            config=config,
            llm_client=FakeLLM(responses=["{}"]),
            # duck-typed test double; CVEvaluator is a concrete class, not a Protocol
            evaluator=evaluator,  # type: ignore[arg-type]
        )

        X_train = pd.DataFrame({"a": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])
        features = pd.DataFrame({"f1": [0.2, 0.3, 0.4, 0.5]})

        pipeline._evaluate_and_select(
            features_train=features,
            features_test=pd.DataFrame(),
            X_train=X_train,
            y_train=y_train,
            X_test=None,
            all_specs=[],
            agents=[],
            code="",
        )
        assert evaluator.calls == 1

        y_train = pd.Series([1, 0, 1, 0])
        pipeline._evaluate_and_select(
            features_train=features,
            features_test=pd.DataFrame(),
            X_train=X_train,
            y_train=y_train,
            X_test=None,
            all_specs=[],
            agents=[],
            code="",
        )
        assert evaluator.calls == 2
