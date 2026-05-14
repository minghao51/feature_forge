"""Structured logging configuration with OpenTelemetry integration.

Outputs pretty colorful logs in development (TTY) and JSON in production.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import structlog

_LOG_LEVELS = {"debug": 0, "info": 10, "warning": 20, "error": 30, "critical": 40}


def add_open_telemetry_spans(
    _logger: structlog.types.WrappedLogger,
    _method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Inject current OpenTelemetry span context into log events.

    This allows correlating logs with distributed traces.
    """
    try:
        from opentelemetry.trace import get_current_span
    except ImportError:
        return event_dict

    span = get_current_span()
    if not span.is_recording():
        event_dict["span"] = None
        return event_dict

    ctx = span.get_span_context()
    event_dict["span"] = {
        "span_id": format(ctx.span_id, "016x"),
        "trace_id": format(ctx.trace_id, "032x"),
    }
    return event_dict


def _drop_below_level(min_level: str) -> Callable[..., Any]:
    """Return a processor that drops log events below *min_level*."""
    threshold = _LOG_LEVELS.get(min_level.lower(), 0)

    def processor(
        _logger: structlog.types.WrappedLogger,
        _method_name: str,
        event_dict: structlog.types.EventDict,
    ) -> structlog.types.EventDict:
        level = event_dict.get("level", "info")
        if _LOG_LEVELS.get(level, 10) < threshold:
            raise structlog.DropEvent
        return event_dict

    return processor


def configure_logging(level: str | None = None) -> None:
    """Configure structlog processors.

    Uses pretty console output in TTY (development) and JSON otherwise
    (production / CI).

    Args:
        level: Minimum log level to output. Defaults to ``INFO`` in TTY,
            ``WARNING`` otherwise (notebooks, CI). Can also be set via
            the ``FF_LOG_LEVEL`` environment variable.
    """
    is_tty = sys.stderr.isatty()
    if level is None:
        level = os.environ.get("FF_LOG_LEVEL", "info" if is_tty else "warning")

    shared_processors: list[Callable[..., Any]] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _drop_below_level(level),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_open_telemetry_spans,
    ]

    processors: list[Any]
    if is_tty:
        processors = [*shared_processors, structlog.dev.ConsoleRenderer(colors=True)]
    else:
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
