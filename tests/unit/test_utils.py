"""Tests for shared utility helpers."""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import CodeExecutionError
from feature_forge.utils import run_coro_sync, strip_markdown_fences


async def _echo(value: int) -> int:
    await asyncio.sleep(0.01)
    return value


def test_run_coro_sync_outside_event_loop() -> None:
    assert run_coro_sync(_echo(7)) == 7


def test_run_coro_sync_repeated_calls() -> None:
    assert run_coro_sync(_echo(1)) == 1
    assert run_coro_sync(_echo(2)) == 2


class TestStripMarkdownFences:
    def test_removes_python_fence(self):
        raw = '```python\ndef foo(): pass\n```'
        assert strip_markdown_fences(raw) == 'def foo(): pass'

    def test_removes_bare_fence(self):
        raw = '```\ndef bar(): return 1\n```'
        assert strip_markdown_fences(raw) == 'def bar(): return 1'

    def test_no_fences_passthrough(self):
        code = 'def baz(): return 2'
        assert strip_markdown_fences(code) == code

    def test_fence_with_leading_whitespace(self):
        raw = '```python\nx = 1\n```'
        assert strip_markdown_fences(raw) == 'x = 1'

    def test_only_opening_fence(self):
        raw = '```python\nx = 1'
        assert strip_markdown_fences(raw) == 'x = 1'


class TestSandboxMarkdownFenceHandling:
    def test_code_with_python_fence_succeeds_after_strip(self):
        executor = SandboxedExecutor(timeout_seconds=5.0)
        raw_llm_output = """```python
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['feat_x2'] = df['x'] * 2
    return result
```"""
        code = strip_markdown_fences(raw_llm_output)
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = executor.execute(code, df)
        assert "feat_x2" in result.columns
        assert result["feat_x2"].tolist() == [2, 4, 6]

    def test_code_with_python_fence_fails_without_strip(self):
        executor = SandboxedExecutor(timeout_seconds=5.0)
        raw_llm_output = """```python
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['feat_x2'] = df['x'] * 2
    return result
```"""
        with pytest.raises(CodeExecutionError, match="Invalid syntax"):
            executor.execute(raw_llm_output, pd.DataFrame({"x": [1, 2, 3]}))

    def test_code_with_bare_fence_succeeds_after_strip(self):
        executor = SandboxedExecutor(timeout_seconds=5.0)
        raw_llm_output = """```
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['feat_x3'] = df['x'] * 3
    return result
```"""
        code = strip_markdown_fences(raw_llm_output)
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = executor.execute(code, df)
        assert result["feat_x3"].tolist() == [3, 6, 9]

    def test_plain_code_works_without_strip(self):
        executor = SandboxedExecutor(timeout_seconds=5.0)
        code = """import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['feat_x4'] = df['x'] * 4
    return result
"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = executor.execute(code, df)
        assert result["feat_x4"].tolist() == [4, 8, 12]
