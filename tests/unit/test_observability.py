"""Tests for observability modules."""

from __future__ import annotations

from feature_forge.observability.structlog_config import (
    add_open_telemetry_spans,
    configure_logging,
    get_logger,
)


class TestStructlogConfig:
    def test_configure_logging_does_not_raise(self):
        configure_logging()

    def test_get_logger_returns_logger(self):
        configure_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_add_otel_spans_no_span(self):
        event_dict = {"event": "test"}
        result = add_open_telemetry_spans(None, "info", event_dict)
        assert result["span"] is None


class TestLangfuseTracer:
    def test_trace_agent_decorator(self):
        from feature_forge.observability.langfuse_tracer import trace_agent

        @trace_agent(name="test-agent")
        def dummy_agent():
            return "ok"

        # Decorator should wrap the function without error
        assert dummy_agent() == "ok"

    def test_trace_generation_decorator(self):
        from feature_forge.observability.langfuse_tracer import trace_generation

        @trace_generation(name="test-gen")
        def dummy_gen():
            return "generated"

        assert dummy_gen() == "generated"
