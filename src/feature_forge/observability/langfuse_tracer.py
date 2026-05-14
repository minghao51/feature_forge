"""Langfuse integration for LLM tracing and observability.

Provides decorators and utilities for automatic tracing of:
- Agent execution
- LLM generation calls
- Tool / sandbox execution
- Multi-agent pipelines
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from feature_forge.config import Settings

if TYPE_CHECKING:
    from langfuse import Langfuse

# Global Langfuse client (lazy initialization)
_langfuse: Langfuse | None = None


def get_langfuse(settings: Settings | None = None) -> Langfuse:
    """Get or create the global Langfuse client."""
    global _langfuse
    if _langfuse is None:
        import os

        try:
            from langfuse import Langfuse
        except ImportError as exc:
            raise ImportError("langfuse not installed. Run: uv pip install langfuse") from exc

        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
        _langfuse = Langfuse(host=host) if host else Langfuse()
    return _langfuse


def _get_langfuse_observe() -> Callable[..., Any]:
    """Lazy import of langfuse.observe."""
    try:
        from langfuse import observe
    except ImportError as exc:
        raise ImportError("langfuse not installed. Run: uv pip install langfuse") from exc
    return cast(Callable[..., Any], observe)


def trace_agent(name: str | None = None) -> Callable[..., Any]:
    """Decorator to trace agent execution.

    Usage:
        @trace_agent(name="unary-agent")
        async def generate_features(self, X, y, context):
            ...
    """
    observe = _get_langfuse_observe()
    return cast(
        Callable[..., Any],
        observe(
            name=name or "agent",
            as_type="agent",
            capture_input=True,
            capture_output=True,
        ),
    )


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


def trace_tool(name: str | None = None) -> Callable[..., Any]:
    """Decorator to trace tool / sandbox execution.

    Usage:
        @trace_tool(name="sandbox")
        def execute_code(self, code):
            ...
    """
    observe = _get_langfuse_observe()
    return cast(
        Callable[..., Any],
        observe(
            name=name or "tool",
            as_type="tool",
            capture_input=True,
            capture_output=True,
        ),
    )


def trace_pipeline(name: str = "pipeline") -> Callable[..., Any]:
    """Decorator to trace the full pipeline execution.

    This creates the root trace that all agent and generation spans
    attach to.
    """
    observe = _get_langfuse_observe()
    return cast(
        Callable[..., Any],
        observe(
            name=name,
            as_type="span",
            capture_input=True,
            capture_output=True,
        ),
    )
