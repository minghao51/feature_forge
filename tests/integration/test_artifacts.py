"""Integration tests for unified artifact system across methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from feature_forge.api import FeatureForge
from feature_forge.config import EvaluationConfig, Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.llm.base import LLMClient, LLMResponse


def _make_llm(code: str) -> LLMClient:
    class FakeLLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(model="fake", api_key="fake")

        @property
        def provider_name(self) -> str:
            return "fake"

        async def complete(
            self,
            messages: list[dict[str, str]],
            temperature: float = 0.2,
            max_tokens: int = 4096,
            prompt_meta: Mapping[str, Any] | None = None,
            **kw: Any,
        ) -> LLMResponse:
            return LLMResponse(content=code, model="fake")

        async def _do_complete(
            self,
            messages: list[dict[str, str]],
            temperature: float = 0.2,
            max_tokens: int = 4096,
            json_mode: bool = False,
            prompt_meta: Mapping[str, Any] | None = None,
            **kw: Any,
        ) -> LLMResponse:
            return await self.complete(messages, temperature, max_tokens, **kw)

        async def _do_complete_json(
            self,
            messages: list[dict[str, str]],
            schema_description: str,
            temperature: float = 0.2,
            max_tokens: int = 4096,
            prompt_meta: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            return {}

    return FakeLLM()


def _make_df() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


def _make_evaluator() -> CVEvaluator:
    cfg = Settings(evaluation=EvaluationConfig(cv_folds=2))
    return CVEvaluator(config=cfg)


class TestLLMFESingleShotArtifacts:
    def test_artifacts_populated(self) -> None:
        from feature_forge.methods.llmfe import LLMFEMethod

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['feat1'] = df['a'] * 2
    result['feat2'] = df['b'] + df['a']
    return result
"""
        llm = _make_llm(code)
        baseline = LLMFEMethod(llm_client=llm, mode="single_shot")
        X, y = _make_df()
        baseline.fit(X, y)

        artifacts = baseline.get_artifacts()
        assert "prompt" in artifacts
        assert "raw_response" in artifacts
        assert "generated_code" in artifacts
        assert isinstance(artifacts["generated_code"], str)

    def test_transform_after_fit(self) -> None:
        from feature_forge.methods.llmfe import LLMFEMethod

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['sum_ab'] = df['a'] + df['b']
    return result
"""
        llm = _make_llm(code)
        baseline = LLMFEMethod(llm_client=llm)
        X, y = _make_df()
        baseline.fit(X, y)
        result = baseline.transform(X)
        assert "sum_ab" in result.columns


class TestLLMFEIterativeArtifacts:
    def test_iterations_list_populated(self) -> None:
        from feature_forge.methods.llmfe import LLMFEMethod

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['feat'] = df['a'] + 1
    return result
"""
        llm = _make_llm(code)
        baseline = LLMFEMethod(
            llm_client=llm,
            mode="iterative",
            n_features=2,
            evaluator=_make_evaluator(),
        )
        X, y = _make_df()
        baseline.fit(X, y)

        artifacts = baseline.get_artifacts()
        assert "iterations" in artifacts
        assert isinstance(artifacts["iterations"], list)
        assert len(artifacts["iterations"]) == 2
        for it in artifacts["iterations"]:
            assert "iteration" in it
            assert "generated_code" in it


class TestCAAFEUnifiedArtifacts:
    def test_unified_artifacts(self) -> None:
        from feature_forge.methods.caafe import CAAFEMethod

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['cfeat'] = df['a'] * 3
    return result
"""
        llm = _make_llm(code)
        baseline = CAAFEMethod(
            llm_client=llm,
            variant="unified",
            iterations=2,
            evaluator=_make_evaluator(),
        )
        X, y = _make_df()
        baseline.fit(X, y)

        artifacts = baseline.get_artifacts()
        assert artifacts.get("variant") == "unified"
        assert "iterations" in artifacts
        assert "dataset_description" in artifacts
        assert len(artifacts["iterations"]) == 2

    def test_unified_transform(self) -> None:
        from feature_forge.methods.caafe import CAAFEMethod

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['cf'] = df['a'] + df['b']
    return result
"""
        llm = _make_llm(code)
        baseline = CAAFEMethod(
            llm_client=llm,
            variant="unified",
            iterations=1,
            evaluator=_make_evaluator(),
        )
        X, y = _make_df()
        baseline.fit(X, y)
        result = baseline.transform(X)
        assert isinstance(result, pd.DataFrame)


class TestMALMASArtifacts:
    def _make_fe(self) -> FeatureForge:
        return FeatureForge(llm_client=_make_llm("pass"))

    def test_get_artifacts_empty_before_fit(self) -> None:
        fe = self._make_fe()
        assert fe.get_artifacts() == {}

    def test_generated_scripts_empty_before_fit(self) -> None:
        fe = self._make_fe()
        assert fe.generated_scripts == []

    def test_feature_metadata_empty_before_fit(self) -> None:
        fe = self._make_fe()
        assert fe.feature_metadata == []


class TestDiskModeArtifacts:
    def test_llmfe_disk_mode_returns_lazy_ref(self) -> None:
        from feature_forge.artifacts.base import ArtifactConfig
        from feature_forge.artifacts.storage import LazyDataFrameRef
        from feature_forge.methods.llmfe import LLMFEMethod

        code = """
import pandas as pd
def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['f1'] = df['a'] * 2
    return result
"""
        llm = _make_llm(code)
        cfg = ArtifactConfig(storage_mode="disk", storage_format="parquet")
        baseline = LLMFEMethod(
            llm_client=llm,
            mode="iterative",
            n_features=1,
            evaluator=_make_evaluator(),
            artifact_config=cfg,
        )
        X, y = _make_df()
        baseline.fit(X, y)

        # Iterative mode stores DataFrames inside iterations list
        artifacts = baseline.get_artifacts()
        assert "iterations" in artifacts
        it = artifacts["iterations"][0]
        # In disk mode, DataFrames inside iterations are stored as LazyDataFrameRef
        assert isinstance(it["all_new_features"], LazyDataFrameRef)
        # Verify lazy loading works
        loaded = it["all_new_features"].load()
        assert isinstance(loaded, pd.DataFrame)
        assert "f1" in loaded.columns
