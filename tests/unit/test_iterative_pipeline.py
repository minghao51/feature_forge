"""Unit tests for BaseIterativePipeline orchestration behavior."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from feature_forge.config import Settings
from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.pipeline.iterative import BaseIterativePipeline
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


class TestBaseIterativePipelineRun:
    @pytest.mark.asyncio
    async def test_run_collects_round_artifacts_and_feature_codes(self):
        config = Settings(task="classification", metric="auc", n_rounds=2)
        pipeline = _StubPipeline(config=config, llm_client=_FakeLLM())

        async def _fake_core_run(
            *,
            agents: list[Any],
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

        pipeline.core.run = _fake_core_run  # type: ignore[method-assign]

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
    async def test_run_resets_state_between_invocations(self):
        config = Settings(task="classification", metric="auc", n_rounds=1)
        pipeline = _StubPipeline(config=config, llm_client=_FakeLLM())

        async def _fake_core_run(
            *,
            agents: list[Any],
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

        pipeline.core.run = _fake_core_run  # type: ignore[method-assign]
        X_train = pd.DataFrame({"x": [1, 2, 3, 4]})
        y_train = pd.Series([0, 1, 0, 1])

        first = await pipeline.run(X_train, y_train)
        second = await pipeline.run(X_train, y_train)

        assert len(first["round_artifacts"]) == 1
        assert len(second["round_artifacts"]) == 1
        assert second["feature_codes"] == ["# code"]
