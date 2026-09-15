"""Tests for observability modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from feature_forge.contracts import Layer, RunState
from feature_forge.observability.hamilton_adapter import (
    HamiltonObservedDriver,
    HamiltonTelemetryRecorder,
    TelemetrySink,
    create_hamilton_lifecycle_adapter,
    create_run_event_sink,
)
from feature_forge.observability.structlog_config import (
    add_open_telemetry_spans,
    configure_logging,
    get_logger,
)


class TestStructlogConfig:
    def test_configure_logging_does_not_raise(self) -> None:
        configure_logging()

    def test_get_logger_returns_logger(self) -> None:
        configure_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_add_otel_spans_no_span(self) -> None:
        event_dict = {"event": "test"}
        result = add_open_telemetry_spans(None, "info", event_dict)
        assert result["span"] is None


class TestLangfuseTracer:
    def test_trace_generation_decorator(self) -> None:
        from feature_forge.observability.langfuse_tracer import trace_generation

        @trace_generation(name="test-gen")
        def dummy_gen() -> str:
            return "generated"

        assert dummy_gen() == "generated"

    def test_add_otel_spans_with_active_span(self) -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        trace.set_tracer_provider(TracerProvider())
        event_dict = {"event": "test"}
        with trace.get_tracer(__name__).start_as_current_span("test-span"):
            result = add_open_telemetry_spans(None, "info", event_dict)
            assert "span" in result
            assert "span_id" in result["span"]
            assert "trace_id" in result["span"]


class _ResultStore:
    """Minimal stand-in exposing only the ``path`` the recorder reads."""

    def __init__(self, path: Path) -> None:
        self.path = path


def _recorder(events: list[dict[str, Any]], **kwargs: Any) -> HamiltonTelemetryRecorder:
    return HamiltonTelemetryRecorder(
        run_id="run-1",
        case_id="case-1",
        sink=cast("TelemetrySink", events.append),
        **kwargs,
    )


def _caching_event(event_type: Any, *, node: str = "bronze_snapshot", value: Any = None) -> Any:
    from hamilton.caching.adapter import CachingEvent  # type: ignore[import-untyped]

    return CachingEvent(
        run_id="h-run-1",
        actor="adapter",
        event_type=event_type,
        node_name=node,
        task_id=None,
        value=value,
        timestamp=1.0,
    )


class TestHamiltonTelemetryRecorder:
    def test_node_events_emit_layer_duration_and_error_type_only(self) -> None:
        events: list[dict[str, Any]] = []
        recorder = _recorder(events)
        tags = {"layer": "bronze", "cost": "low", "owner": "data-team"}

        recorder.before(hamilton_run_id="h1", node_name="bronze_snapshot", task_id=None, tags=tags)
        recorder.after(
            hamilton_run_id="h1",
            node_name="bronze_snapshot",
            task_id=None,
            tags=tags,
            success=True,
            error=None,
        )

        assert [event["state"] for event in events] == ["node_running", "node_succeeded"]
        assert all(event["layer"] == "bronze" for event in events)
        assert all(event["run_id"] == "run-1" and event["case_id"] == "case-1" for event in events)
        assert all(event["hamilton_run_id"] == "h1" for event in events)
        assert events[1]["duration_ms"] is not None
        assert events[1]["duration_ms"] >= 0
        assert events[1]["error_type"] is None

    def test_failure_events_carry_error_type_but_never_exception_text(self) -> None:
        events: list[dict[str, Any]] = []
        recorder = _recorder(events)

        recorder.before(
            hamilton_run_id="h1",
            node_name="gold_manifest",
            task_id=None,
            tags={"layer": "gold"},
        )
        recorder.after(
            hamilton_run_id="h1",
            node_name="gold_manifest",
            task_id=None,
            tags={"layer": "gold"},
            success=False,
            error=ValueError("db password=hunter2 leaked"),
        )

        failure = events[-1]
        assert failure["state"] == "node_failed"
        assert failure["error_type"] == "ValueError"
        assert "hunter2" not in json.dumps(events)

    def test_tags_are_allowlisted_and_secrets_never_leak(self) -> None:
        events: list[dict[str, Any]] = []
        recorder = _recorder(events)

        recorder.before(
            hamilton_run_id="h1",
            node_name="bronze_snapshot",
            task_id=None,
            tags={
                "layer": "gold",
                "owner": "team-x",
                "sensitivity": "public",
                "prompt": "SECRET-PROMPT",
                "api_key": "sk-SECRET-KEY",
                "node_input": "SECRET-INPUT",
            },
        )

        emitted = events[0]
        assert set(emitted["tags"]) <= {"layer", "owner", "sensitivity"}
        assert emitted["layer"] == "gold"
        serialized = json.dumps(events)
        for secret in ("SECRET-PROMPT", "sk-SECRET-KEY", "SECRET-INPUT"):
            assert secret not in serialized

    def test_cache_events_classify_hit_miss_error_with_layer_and_duration(self) -> None:
        from hamilton.caching.adapter import CachingEventType

        events: list[dict[str, Any]] = []
        recorder = _recorder(events)
        recorder.before(
            hamilton_run_id="h-run-1", node_name="bronze_snapshot", task_id=None, tags={}
        )
        recorder.after(
            hamilton_run_id="h-run-1",
            node_name="bronze_snapshot",
            task_id=None,
            tags={},
            success=True,
            error=None,
        )

        recorder.cache_events(
            [
                _caching_event(CachingEventType.GET_RESULT, value="v1"),
                _caching_event(CachingEventType.EXECUTE_NODE, node="canonical_features"),
                _caching_event(CachingEventType.MISSING_RESULT, node="gold_manifest"),
                _caching_event(CachingEventType.FAILED_RETRIEVAL, node="platinum_checks"),
                _caching_event(CachingEventType.FAILED_EXECUTION, node="mystery_node"),
                _caching_event(CachingEventType.SET_RESULT, value="v1"),
            ],
            result_store=None,
        )

        cache_events = [event for event in events if event["state"] == "node_cache"]
        assert [event["cache_outcome"] for event in cache_events] == [
            "hit",
            "miss",
            "error",
            "error",
            "error",
        ]
        assert [event["layer"] for event in cache_events] == [
            "bronze",
            "silver",
            "gold",
            "platinum",
            None,
        ]
        assert cache_events[0]["error_type"] is None
        assert all(event["error_type"] == "HamiltonCacheError" for event in cache_events[2:])
        assert all(event["serialized_bytes"] == 0 for event in cache_events)
        assert cache_events[0]["duration_ms"] is not None

    def test_cache_events_report_serialized_bytes_and_set_result_pairing(
        self, tmp_path: Path
    ) -> None:
        from hamilton.caching.adapter import CachingEventType

        (tmp_path / "v1").write_bytes(b"1234567")
        (tmp_path / "v3").write_bytes(b"abc")
        events: list[dict[str, Any]] = []
        recorder = _recorder(events)

        # Realistic miss sequence: retrieval fails, node executes, result stored.
        recorder.cache_events(
            [
                _caching_event(CachingEventType.GET_RESULT, value="v1"),
                _caching_event(CachingEventType.MISSING_RESULT, value="missing-file"),
                _caching_event(CachingEventType.EXECUTE_NODE, value=None),
                _caching_event(CachingEventType.SET_RESULT, value="v3"),
            ],
            result_store=_ResultStore(tmp_path),
        )

        cache_events = [event for event in events if event["state"] == "node_cache"]
        assert [event["cache_outcome"] for event in cache_events] == ["hit", "error", "miss"]
        # Hit reads its own data_version file; the error/miss pair fall back to
        # the bytes recorded by the node's set_result event.
        assert [event["serialized_bytes"] for event in cache_events] == [7, 3, 3]

    def test_event_stream_is_bounded_by_max_events(self) -> None:
        events: list[dict[str, Any]] = []
        recorder = _recorder(events, max_events=3)
        tags = {"layer": "bronze"}

        for index in range(5):
            recorder.before(
                hamilton_run_id=f"h{index}", node_name="bronze_snapshot", task_id=None, tags=tags
            )

        assert len(events) == 3
        assert len(recorder._durations) <= 3

    def test_finish_run_clears_transient_state_and_reopens_event_budget(self) -> None:
        from hamilton.caching.adapter import CachingEventType

        events: list[dict[str, Any]] = []
        recorder = _recorder(events, max_events=4)
        for index in range(2):
            recorder.before(
                hamilton_run_id=f"h{index}", node_name="bronze_snapshot", task_id=None, tags={}
            )
            recorder.after(
                hamilton_run_id=f"h{index}",
                node_name="bronze_snapshot",
                task_id=None,
                tags={},
                success=True,
                error=None,
            )
        assert len(events) == 4  # budget consumed: node_running + node_succeeded per node

        recorder.cache_events(
            [_caching_event(CachingEventType.GET_RESULT, value="v")], result_store=None
        )
        assert not [event for event in events if event["state"] == "node_cache"]

        recorder.finish_run()

        assert recorder._started == {}
        assert recorder._durations == {}

        # An ``after`` without a ``before`` finds no timing state: duration is None.
        recorder.after(
            hamilton_run_id="h1",
            node_name="bronze_snapshot",
            task_id=None,
            tags={},
            success=True,
            error=None,
        )
        assert events[-1]["state"] == "node_succeeded"
        assert events[-1]["duration_ms"] is None

        # The event budget reopens after finish_run.
        recorder.before(hamilton_run_id="h2", node_name="bronze_snapshot", task_id=None, tags={})
        assert events[-1]["state"] == "node_running"

    def test_max_events_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_events must be >= 1"):
            _recorder([], max_events=0)


def test_lifecycle_repository_rejects_path_traversal_run_id(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from feature_forge.experiment.lifecycle import LocalRunRepository

    repository = LocalRunRepository(tmp_path / "journal")
    with pytest.raises(ValidationError, match="identifier"):
        repository.record(
            run_id="../../escape",
            case_id="case",
            event_type="case_running",
            state=RunState.RUNNING,
        )
    assert not (tmp_path / "escape").exists()


class TestHamiltonAdapterWiring:
    def test_lifecycle_adapter_forwards_redacted_events_to_recorder(self) -> None:
        events: list[dict[str, Any]] = []
        adapter = create_hamilton_lifecycle_adapter(_recorder(events))

        adapter.run_before_node_execution(
            node_name="bronze_snapshot",
            node_tags={"layer": "bronze", "api_key": "sk-HOOK-SECRET"},
            task_id=None,
            run_id="h1",
            unused_kwarg="ignored",
        )
        adapter.run_after_node_execution(
            node_name="bronze_snapshot",
            node_tags={"layer": "bronze"},
            error=None,
            success=True,
            task_id=None,
            run_id="h1",
        )

        assert [event["state"] for event in events] == ["node_running", "node_succeeded"]
        assert events[0]["tags"] == {"layer": "bronze"}
        assert "sk-HOOK-SECRET" not in json.dumps(events)

    def test_lifecycle_repository_rejects_traversal_attempt_id(self, tmp_path: Path) -> None:
        from pydantic import ValidationError

        from feature_forge.experiment.lifecycle import LocalRunRepository

        repository = LocalRunRepository(tmp_path / "journal")
        with pytest.raises(ValidationError, match="identifier"):
            repository.record(
                run_id="../../escape",
                case_id="case",
                event_type="case_running",
            )
        assert not (tmp_path / "escape").exists()

    def test_run_event_sink_maps_states_layers_and_details(self) -> None:
        class _Repository:
            def __init__(self) -> None:
                self.records: list[dict[str, Any]] = []

            def record(self, **values: Any) -> None:
                self.records.append(values)

        repository = _Repository()
        recorder = HamiltonTelemetryRecorder(
            run_id="run-1",
            case_id="case-1",
            sink=create_run_event_sink(repository),
        )

        recorder.before(
            hamilton_run_id="h1",
            node_name="bronze_snapshot",
            task_id=None,
            tags={"layer": "bronze"},
        )
        recorder.after(
            hamilton_run_id="h1",
            node_name="bronze_snapshot",
            task_id=None,
            tags={"layer": "bronze"},
            success=True,
            error=None,
        )

        succeeded = repository.records[-1]
        assert succeeded["event_type"] == "hamilton_node_succeeded"
        assert succeeded["state"] is RunState.SUCCEEDED
        assert succeeded["stage"] == "bronze_snapshot"
        assert succeeded["layer"] is Layer.BRONZE
        assert succeeded["run_id"] == "run-1"
        assert succeeded["case_id"] == "case-1"
        assert "state" not in succeeded["details"]
        assert "node" not in succeeded["details"]
        assert "duration_ms" in succeeded["details"]

        # Unknown nodes carry no layer mapping and stay at the journal untouched.
        recorder.before(hamilton_run_id="h1", node_name="unknown_node", task_id=None, tags={})
        unknown = repository.records[-1]
        assert unknown["state"] is RunState.RUNNING
        assert unknown["layer"] is None

    def test_observed_driver_flushes_cache_events_and_clears_state(self, tmp_path: Path) -> None:
        from hamilton.caching.adapter import CachingEventType

        (tmp_path / "v1").write_bytes(b"payload")

        class _Cache:
            run_ids: ClassVar[list[str]] = ["h-run-1"]
            last_run_id = "h-run-1"
            result_store = _ResultStore(tmp_path)

            @staticmethod
            def logs(run_id: str, level: str) -> dict[str, list[Any]]:
                assert run_id == "h-run-1"
                assert level == "debug"
                return {
                    "bronze_snapshot": [_caching_event(CachingEventType.GET_RESULT, value="v1")]
                }

        class _Driver:
            def __init__(self) -> None:
                self.cache = _Cache()
                self.calls = 0

            def execute(self, *args: Any, **kwargs: Any) -> str:
                self.calls += 1
                return "ok"

        events: list[dict[str, Any]] = []
        driver = HamiltonObservedDriver(_Driver(), _recorder(events))

        assert driver.execute(["final"]) == "ok"
        assert driver._driver.calls == 1  # attribute access proxies to the driver

        cache_events = [event for event in events if event["state"] == "node_cache"]
        assert len(cache_events) == 1
        assert cache_events[0]["cache_outcome"] == "hit"
        assert cache_events[0]["serialized_bytes"] == len(b"payload")
        assert driver._recorder._started == {}  # transient state cleared after execute

    def test_observed_driver_flushes_and_clears_even_when_execution_fails(self) -> None:
        from hamilton.caching.adapter import CachingEventType

        class _Cache:
            run_ids: ClassVar[list[str]] = ["h-run-1"]
            last_run_id = "h-run-1"
            result_store = None

            @staticmethod
            def logs(run_id: str, level: str) -> dict[str, list[Any]]:
                return {"n": [_caching_event(CachingEventType.EXECUTE_NODE)]}

        class _FailingDriver:
            def __init__(self) -> None:
                self.cache = _Cache()

            def execute(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("boom")

        events: list[dict[str, Any]] = []
        driver = HamiltonObservedDriver(_FailingDriver(), _recorder(events))

        with pytest.raises(RuntimeError, match="boom"):
            driver.execute()

        cache_events = [event for event in events if event["state"] == "node_cache"]
        assert [event["cache_outcome"] for event in cache_events] == ["miss"]
        assert driver._recorder._started == {}  # finally-branch cleared state

    def test_observed_driver_without_cache_still_clears_state(self) -> None:
        class _Driver:
            def __init__(self) -> None:
                self.value = 41

            def execute(self, *args: Any, **kwargs: Any) -> int:
                return 42

        events: list[dict[str, Any]] = []
        driver = HamiltonObservedDriver(_Driver(), _recorder(events))

        assert driver.execute() == 42
        assert driver.value == 41
        assert events == []
        assert driver._recorder._started == {}
