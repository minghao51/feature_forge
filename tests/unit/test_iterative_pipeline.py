"""Unit tests for BaseIterativePipeline orchestration behavior."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.agents.base import Agent
from feature_forge.methods.malmas.memory.base import AgentMemory
from feature_forge.methods.malmas.pipeline.iterative import BaseIterativePipeline, IterativePipeline
from feature_forge.methods.malmas.pipeline.result import PipelineResult


class _FakeLLM(LLMClient):
    def __init__(self) -> None:
        super().__init__(model="fake", api_key="fake")

    @property
    def provider_name(self) -> str:
        return "fake"

    async def _call_api(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int, **kwargs: Any
    ) -> Any:
        del messages, temperature, max_tokens, kwargs
        return {}

    def _extract_content(self, raw_response: Any) -> str:
        del raw_response
        return ""

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        del raw_response
        return (0, 0, 0)


class _StubPipeline(BaseIterativePipeline):
    async def _select_agents(
        self,
        round_idx: int,
        X_train: pd.DataFrame,
        description: dict[str, Any],
        task_description: str,
    ) -> list[str]:
        del round_idx, X_train, description, task_description
        return []


class _UnknownAgentPipeline(_StubPipeline):
    async def _select_agents(
        self,
        round_idx: int,
        X_train: pd.DataFrame,
        description: dict[str, Any],
        task_description: str,
    ) -> list[str]:
        del round_idx, X_train, description, task_description
        return ["not-a-real-agent"]


class TestBaseIterativePipelineRun:
    @pytest.mark.asyncio
    async def test_run_collects_round_artifacts_and_feature_codes(self) -> None:
        config = Settings(task="classification", metric="auc", n_rounds=2)
        pipeline = _StubPipeline(config=config, llm_client=_FakeLLM())

        async def _fake_core_run(
            agents: list[Agent],
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame | None,
            context: dict[str, Any] | list[dict[str, Any]] | None = None,
        ) -> PipelineResult:
            del agents, y_train, context
            round_num = len(pipeline.round_artifacts) + 1
            col = f"feat_{round_num}"
            top_train = pd.DataFrame({col: [1.0] * len(X_train)}, index=X_train.index)
            top_test = (
                pd.DataFrame({col: [2.0] * len(X_test)}, index=X_test.index)
                if X_test is not None
                else pd.DataFrame()
            )
            return PipelineResult(
                features_train=top_train.copy(),
                features_test=top_test.copy(),
                all_features_train=top_train.copy(),
                all_features_test=top_test.copy(),
                selected_features_train=top_train,
                selected_features_test=top_test,
                top_features_train=top_train,
                top_features_test=top_test,
                baseline_score=0.5,
                specs=[],
                agent_gains={},
                gains={col: 0.1},
                generated_code=f"# code_{round_num}",
            )

        # Test seam: replace the core pipeline method at runtime (still called
        # with keyword arguments by BaseIterativePipeline.run).
        pipeline.core.run = _fake_core_run  # type: ignore[method-assign, assignment]

        X_train = pd.DataFrame({"x": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])
        X_test = pd.DataFrame({"x": [5, 6]})

        result = await pipeline.run(X_train, y_train, X_test=X_test)

        assert len(result["round_artifacts"]) == 2
        assert result["feature_codes"] == ["# code_1", "# code_2"]
        assert set(result["selected_features"]) == {"feat_1", "feat_2"}
        assert "feat_1" in result["X_train_enhanced"].columns
        assert "feat_2" in result["X_train_enhanced"].columns
        assert result["X_test_enhanced"] is not None
        assert "feat_1" in result["X_test_enhanced"].columns
        assert "feat_2" in result["X_test_enhanced"].columns

    @pytest.mark.asyncio
    async def test_run_resets_state_between_invocations(self) -> None:
        config = Settings(task="classification", metric="auc", n_rounds=1)
        pipeline = _StubPipeline(config=config, llm_client=_FakeLLM())

        async def _fake_core_run(
            agents: list[Agent],
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame | None,
            context: dict[str, Any] | list[dict[str, Any]] | None = None,
        ) -> PipelineResult:
            del agents, y_train, X_test, context
            top_train = pd.DataFrame({"feat": [1.0] * len(X_train)}, index=X_train.index)
            return PipelineResult(
                features_train=top_train.copy(),
                features_test=pd.DataFrame(),
                all_features_train=top_train.copy(),
                all_features_test=pd.DataFrame(),
                selected_features_train=top_train,
                selected_features_test=pd.DataFrame(),
                top_features_train=top_train,
                top_features_test=pd.DataFrame(),
                baseline_score=0.5,
                specs=[],
                agent_gains={},
                gains={"feat": 0.1},
                generated_code="# code",
            )

        # Test seam: replace the core pipeline method at runtime (still called
        # with keyword arguments by BaseIterativePipeline.run).
        pipeline.core.run = _fake_core_run  # type: ignore[method-assign, assignment]
        X_train = pd.DataFrame({"x": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])

        first = await pipeline.run(X_train, y_train)
        second = await pipeline.run(X_train, y_train)

        assert len(first["round_artifacts"]) == 1
        assert len(second["round_artifacts"]) == 1
        assert second["feature_codes"] == ["# code"]

    @pytest.mark.asyncio
    async def test_unknown_selected_agent_fails_before_core_execution(self) -> None:
        pipeline = _UnknownAgentPipeline(Settings(n_rounds=1), _FakeLLM())
        X_train = pd.DataFrame({"x": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])

        with pytest.raises(ValueError, match="Unknown agent"):
            await pipeline.run(X_train, y_train)


class TestIterativeWarmStart:
    def test_normal_fit_ignores_persisted_memory_and_warm_start_loads_it(
        self, tmp_path: Any
    ) -> None:
        memory_path = tmp_path / "unary_memory.json"
        saved = AgentMemory("unary", str(memory_path))
        saved.record_feedback("old", "auc", 0.1, True, 0, ["x"], "numerical")
        saved.save()

        cold = IterativePipeline(
            Settings(n_rounds=1),
            _FakeLLM(),
            memory_dir=str(tmp_path),
            warm_start=False,
        )
        warm = IterativePipeline(
            Settings(n_rounds=1),
            _FakeLLM(),
            memory_dir=str(tmp_path),
            warm_start=True,
        )

        assert cold._get_memory("unary").feedback == []
        assert [item["feature_name"] for item in warm._get_memory("unary").feedback] == ["old"]

    def test_normal_reset_clears_router_learning(self) -> None:
        pipeline = IterativePipeline(Settings(n_rounds=1), _FakeLLM(), warm_start=False)
        pipeline.router.update_performance("unary", 0.5)
        pipeline.router.dataset_characteristics = {"total_columns": 1}

        pipeline._reset_learned_state()

        assert pipeline.router.agent_performance["unary"] == []
        assert pipeline.router.dataset_characteristics is None
