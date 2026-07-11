"""Langfuse integration for LLM tracing and observability.

Provides a decorator for automatic tracing of LLM generation calls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast


def _get_langfuse_observe() -> Callable[..., Any]:
    """Lazy import of langfuse.observe."""
    try:
        from langfuse import observe
    except ImportError as exc:
        raise ImportError("langfuse not installed. Run: uv pip install langfuse") from exc
    return cast(Callable[..., Any], observe)


def trace_generation(name: str | None = None) -> Callable[..., Any]:
    """Decorator to trace LLM generation calls.

    Usage:
        @trace_generation(name="feature-plan")
        async def generate_plan(self, prompt):
            ...
    """
    observe = _get_langfuse_observe()
    return cast(
        Callable[..., Any],
        observe(
            name=name or "generation",
            as_type="generation",
            capture_input=True,
            capture_output=True,
        ),
    )
