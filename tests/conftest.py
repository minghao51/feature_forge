"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from feature_forge.llm.base import LLMClient

_AUTO_MARK_DIRS = {"unit": "unit", "integration": "integration"}
_MARKER_DECORATORS = {"property", "metamorphic", "contract", "differential"}


class FakeLLM(LLMClient):
    """Fake LLM that returns predetermined responses for deterministic testing."""

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
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs}
        )
        return None

    def _extract_content(self, raw_response: Any) -> str:
        idx = (self.call_count - 1) % len(self.responses)
        return self.responses[idx]

    def _extract_usage(self, raw_response: Any) -> tuple[int, int, int]:
        return 0, 0, 0

    async def _do_complete_json(
        self,
        messages: list[dict[str, str]],
        schema_description: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        enhanced = self._inject_json_schema(messages, schema_description)
        response = await self._do_complete(enhanced, temperature, max_tokens, json_mode=True)
        return json.loads(response.content)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark tests by directory and propagate AI-code markers."""
    for item in items:
        for dir_name, marker_name in _AUTO_MARK_DIRS.items():
            if f"/{dir_name}/" in item.nodeid or f"\\{dir_name}\\" in item.nodeid:
                item.add_marker(getattr(pytest.mark, marker_name))
                break


@pytest.fixture
def fake_llm():
    """Return a FakeLLM instance with no responses configured."""
    return FakeLLM()


@pytest.fixture
def sample_config():
    """Return a minimal valid configuration dict."""
    return {
        "task": "classification",
        "metric": "auc",
        "n_rounds": 2,
        "random_state": 42,
    }


@pytest.fixture
def sample_dataframe():
    """Return a small synthetic DataFrame for tests."""
    return pd.DataFrame(
        {
            "num_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat_c": ["x", "y", "x", "y", "x"],
        }
    )


@pytest.fixture
def sample_series():
    """Return a binary classification target."""
    return pd.Series([0, 1, 0, 1, 0], name="target")
