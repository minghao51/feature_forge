"""Langfuse integration for LLM tracing and observability.

Provides a decorator for automatic tracing of LLM generation calls.

Langfuse is an optional dependency (``uv sync --extra observability``). When it
is not installed, ``trace_generation`` degrades to an identity decorator so an
explicitly enabled tracer never crashes a run — it simply no-ops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

_WARNED_ABOUT_LANGFUSE = False


def _identity_decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Passthrough decorator used when langfuse is unavailable."""
    return fn


def _get_langfuse_observe() -> Callable[..., Any] | None:
    """Lazy import of langfuse.observe; returns ``None`` if unavailable."""
    global _WARNED_ABOUT_LANGFUSE
    try:
        from langfuse import observe
    except ImportError:
        if not _WARNED_ABOUT_LANGFUSE:
            _WARNED_ABOUT_LANGFUSE = True
        return None
    return cast(Callable[..., Any], observe)


def trace_generation(name: str | None = None) -> Callable[..., Any]:
    """Decorator to trace LLM generation calls.

    When langfuse is not installed, returns an identity decorator so callers
    that opted into tracing still run without error.

    Usage:
        @trace_generation(name="feature-plan")
        async def generate_plan(self, prompt):
            ...
    """
    observe = _get_langfuse_observe()
    if observe is None:
        return _identity_decorator
    return cast(
        Callable[..., Any],
        observe(
            name=name or "generation",
            as_type="generation",
            capture_input=True,
            capture_output=True,
        ),
    )
