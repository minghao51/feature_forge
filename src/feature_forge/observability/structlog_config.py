"""Structured logging configuration with OpenTelemetry integration.

Outputs pretty colorful logs in development (TTY) and JSON in production.
"""

from __future__ import annotations

import sys

import structlog
from opentelemetry import trace


def add_open_telemetry_spans(
    _logger: structlog.types.WrappedLogger,
    _method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Inject current OpenTelemetry span context into log events.

    This allows correlating logs with distributed traces.
    """
    span = trace.get_current_span()
    if not span.is_recording():
        event_dict["span"] = None
        return event_dict

    ctx = span.get_span_context()
    event_dict["span"] = {
        "span_id": format(ctx.span_id, "016x"),
        "trace_id": format(ctx.trace_id, "032x"),
    }
    return event_dict


def configure_logging() -> None:
    """Configure structlog processors.

    Uses pretty console output in TTY (development) and JSON otherwise
    (production / CI).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_open_telemetry_spans,
    ]

    if sys.stderr.isatty():
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
    return structlog.get_logger(name)
