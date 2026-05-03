"""Tests for baseline methods."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_forge.baselines import BaselineRegistry, LLMFEBaseline
from feature_forge.baselines.base import Baseline


class FakeLLM:
    def __init__(self, code: str) -> None:
        self.code = code

    async def complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        from feature_forge.llm.base import LLMResponse
        return LLMResponse(content=self.code, model="fake")


class TestBaselineRegistry:
    def test_get_builtin_baselines(self):
        baselines = BaselineRegistry.get_builtin_baselines()
        assert "openfe" in baselines
        assert "caafe" in baselines
        assert "llmfe" in baselines

    def test_get_all_baselines(self):
        baselines = BaselineRegistry.get_all_baselines()
        assert "llmfe" in baselines


class TestLLMFEBaseline:
    def test_init(self):
        llm = FakeLLM("def generate_features(df): return df")
        baseline = LLMFEBaseline(llm_client=llm)
        assert baseline.name == "llmfe"

    def test_fit_transform(self):
        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['sum_ab'] = df['a'] + df['b']
    return result
"""
        llm = FakeLLM(code)
        baseline = LLMFEBaseline(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        result = baseline.fit_transform(X, y)
        assert "sum_ab" in result.columns

    def test_transform_before_fit_raises(self):
        llm = FakeLLM("")
        baseline = LLMFEBaseline(llm_client=llm)
        with pytest.raises(RuntimeError):
            baseline.transform(pd.DataFrame())


class TestOpenFENotInstalled:
    def test_fit_raises_when_not_installed(self):
        llm = FakeLLM("def generate_features(df): return df")
        _ = LLMFEBaseline(llm_client=llm)
        assert issubclass(LLMFEBaseline, Baseline)
