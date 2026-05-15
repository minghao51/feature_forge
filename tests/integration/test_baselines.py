"""Tests for baseline methods."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_forge.methods import LLMFEMethod, MethodRegistry
from feature_forge.methods.base import BaseMethod


class FakeLLM:
    def __init__(self, code: str) -> None:
        self.code = code

    async def complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        from feature_forge.llm.base import LLMResponse

        return LLMResponse(content=self.code, model="fake")

    async def _do_complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        return await self.complete(messages, temperature, max_tokens, **kwargs)


class TestMethodRegistry:
    def test_get_builtin_methods(self):
        baselines = MethodRegistry.get_builtin_methods()
        assert "malmas" in baselines
        assert "openfe" in baselines
        assert "caafe" in baselines
        assert "llmfe" in baselines

    def test_get_all_methods(self):
        baselines = MethodRegistry.get_all_methods()
        assert "llmfe" in baselines


class TestLLMFEMethod:
    def test_init(self):
        llm = FakeLLM("def generate_features(df): return df")
        baseline = LLMFEMethod(llm_client=llm)
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
        baseline = LLMFEMethod(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        result = baseline.fit_transform(X, y)
        assert "sum_ab" in result.columns

    def test_transform_before_fit_raises(self):
        llm = FakeLLM("")
        baseline = LLMFEMethod(llm_client=llm)
        with pytest.raises(RuntimeError):
            baseline.transform(pd.DataFrame())


class TestOpenFENotInstalled:
    def test_fit_raises_when_not_installed(self):
        llm = FakeLLM("def generate_features(df): return df")
        _ = LLMFEMethod(llm_client=llm)
        assert issubclass(LLMFEMethod, BaseMethod)
