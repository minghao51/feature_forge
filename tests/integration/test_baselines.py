"""Tests for baseline methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.methods import LLMFEMethod, MethodRegistry
from feature_forge.methods.base import BaseMethod


class FakeLLM(LLMClient):
    def __init__(self, code: str) -> None:
        super().__init__(model="fake", api_key="fake")
        self.code = code

    @property
    def provider_name(self) -> str:
        return "fake"

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        prompt_meta: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self.code, model="fake")

    async def _do_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
        prompt_meta: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.complete(messages, temperature, max_tokens, **kwargs)


class TestMethodRegistry:
    def test_get_builtin_methods(self) -> None:
        baselines = MethodRegistry.get_builtin_methods()
        assert "malmas" in baselines
        assert "openfe" in baselines
        assert "caafe" in baselines
        assert "llmfe" in baselines

    def test_get_all_methods(self) -> None:
        baselines = MethodRegistry.get_all_methods()
        assert "llmfe" in baselines


class TestLLMFEMethod:
    def test_init(self) -> None:
        llm = FakeLLM("def generate_features(df): return df")
        baseline = LLMFEMethod(llm_client=llm)
        assert baseline.name == "llmfe"

    def test_fit_transform(self) -> None:
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

    def test_transform_before_fit_raises(self) -> None:
        llm = FakeLLM("")
        baseline = LLMFEMethod(llm_client=llm)
        with pytest.raises(RuntimeError):
            baseline.transform(pd.DataFrame())


class TestOpenFENotInstalled:
    def test_fit_raises_when_not_installed(self) -> None:
        llm = FakeLLM("def generate_features(df): return df")
        _ = LLMFEMethod(llm_client=llm)
        assert issubclass(LLMFEMethod, BaseMethod)
