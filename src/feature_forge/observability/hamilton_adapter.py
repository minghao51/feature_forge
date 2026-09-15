"""Redacted Hamilton node and cache telemetry."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from feature_forge.contracts.stages import Layer, RunState
from feature_forge.dataflows.inventory import node_layer
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)
_ALLOWED_TAGS = {"layer", "cost", "persistence", "sensitivity", "owner"}
_CACHE_OUTCOMES = {
    "get_result": "hit",
    "execute_node": "miss",
    "missing_result": "error",
    "failed_retrieval": "error",
    "failed_execution": "error",
}


class TelemetrySink(Protocol):
    """Receive one already-redacted telemetry event."""

    def __call__(self, event: dict[str, Any]) -> None: ...


class HamiltonTelemetryRecorder:
    """Thread-safe bounded recorder that never accepts node inputs or results."""

    def __init__(
        self,
        *,
        run_id: str,
        case_id: str,
        sink: TelemetrySink | None = None,
        max_events: int = 10_000,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        self.run_id = run_id
        self.case_id = case_id
        self.sink = sink or self._log
        self.max_events = max_events
        self._started: dict[tuple[str, str | None, str], float] = {}
        self._durations: dict[tuple[str, str | None, str], float] = {}
        self._emitted = 0
        self._lock = threading.Lock()

    def before(
        self,
        *,
        hamilton_run_id: str,
        node_name: str,
        task_id: str | None,
        tags: dict[str, Any],
    ) -> None:
        """Record node start metadata without arguments."""
        key = (hamilton_run_id, task_id, node_name)
        with self._lock:
            self._started[key] = time.monotonic()
        self._emit("node_running", hamilton_run_id, node_name, task_id, tags)

    def after(
        self,
        *,
        hamilton_run_id: str,
        node_name: str,
        task_id: str | None,
        tags: dict[str, Any],
        success: bool,
        error: Exception | None,
    ) -> None:
        """Record node completion metadata without result or exception text."""
        key = (hamilton_run_id, task_id, node_name)
        with self._lock:
            started = self._started.pop(key, None)
            duration_ms = None if started is None else (time.monotonic() - started) * 1000
            if duration_ms is not None:
                self._durations[key] = duration_ms
                while len(self._durations) > self.max_events:
                    self._durations.pop(next(iter(self._durations)))
        self._emit(
            "node_succeeded" if success else "node_failed",
            hamilton_run_id,
            node_name,
            task_id,
            tags,
            duration_ms=duration_ms,
            error_type=type(error).__name__ if error else None,
        )

    def cache_events(self, events: list[Any], *, result_store: Any | None = None) -> None:
        """Emit hit/miss/error metadata from Hamilton's official cache events."""
        serialized_by_node: dict[tuple[str, str | None, str], int] = {}
        for event in events:
            event_type = getattr(getattr(event, "event_type", None), "value", "")
            if event_type not in {"get_result", "set_result"}:
                continue
            size = _serialized_size(result_store, getattr(event, "value", None))
            serialized_by_node[
                (
                    str(getattr(event, "run_id", "")),
                    getattr(event, "task_id", None),
                    str(getattr(event, "node_name", "")),
                )
            ] = size

        for event in events:
            event_type = getattr(getattr(event, "event_type", None), "value", "")
            outcome = _CACHE_OUTCOMES.get(event_type)
            if outcome is None:
                continue
            hamilton_run_id = str(getattr(event, "run_id", ""))
            node_name = str(getattr(event, "node_name", ""))
            task_id = getattr(event, "task_id", None)
            key = (hamilton_run_id, task_id, node_name)
            size = _serialized_size(result_store, getattr(event, "value", None))
            if size == 0:
                size = serialized_by_node.get(key, 0)
            with self._lock:
                duration_ms = self._durations.get(key)
            layer = node_layer(node_name)
            self._emit(
                "node_cache",
                hamilton_run_id,
                node_name,
                task_id,
                {"layer": layer} if layer is not None else {},
                cache_outcome=outcome,
                duration_ms=duration_ms,
                serialized_bytes=size,
                error_type=("HamiltonCacheError" if outcome == "error" else None),
            )

    def finish_run(self) -> None:
        """Clear transient timing state after one Hamilton execution."""
        with self._lock:
            self._started.clear()
            self._durations.clear()
            self._emitted = 0

    def _emit(
        self,
        state: str,
        hamilton_run_id: str,
        node_name: str,
        task_id: str | None,
        tags: dict[str, Any],
        **values: Any,
    ) -> None:
        with self._lock:
            if self._emitted >= self.max_events:
                return
            self._emitted += 1
        allowed = {
            key: str(value)[:200]
            for key, value in tags.items()
            if key in _ALLOWED_TAGS and value is not None
        }
        self.sink(
            {
                "schema_version": "1",
                "event_id": uuid.uuid4().hex,
                "run_id": self.run_id,
                "case_id": self.case_id,
                "hamilton_run_id": hamilton_run_id,
                "task_id": task_id,
                "node": node_name,
                "layer": allowed.get("layer"),
                "state": state,
                "tags": allowed,
                **values,
            }
        )

    @staticmethod
    def _log(event: dict[str, Any]) -> None:
        logger.info("hamilton_lifecycle", **event)


def create_hamilton_lifecycle_adapter(recorder: HamiltonTelemetryRecorder) -> Any:
    """Create a Hamilton node hook backed by ``recorder``."""
    try:
        from hamilton.lifecycle import NodeExecutionHook  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - core install health
        raise RuntimeError("Hamilton is a core dependency; reinstall with `uv sync`") from exc

    class Adapter(NodeExecutionHook):  # type: ignore[misc]
        def run_before_node_execution(
            self,
            *,
            node_name: str,
            node_tags: dict[str, Any],
            task_id: str | None,
            run_id: str,
            **_: Any,
        ) -> None:
            recorder.before(
                hamilton_run_id=run_id,
                node_name=node_name,
                task_id=task_id,
                tags=node_tags,
            )

        def run_after_node_execution(
            self,
            *,
            node_name: str,
            node_tags: dict[str, Any],
            error: Exception | None,
            success: bool,
            task_id: str | None,
            run_id: str,
            **_: Any,
        ) -> None:
            recorder.after(
                hamilton_run_id=run_id,
                node_name=node_name,
                task_id=task_id,
                tags=node_tags,
                success=success,
                error=error,
            )

    return Adapter()


def create_run_event_sink(repository: Any) -> TelemetrySink:
    """Bridge redacted Hamilton telemetry into the lifecycle journal."""

    def sink(event: dict[str, Any]) -> None:
        state = {
            "node_running": RunState.RUNNING,
            "node_succeeded": RunState.SUCCEEDED,
            "node_failed": RunState.FAILED,
        }.get(str(event["state"]))
        layer_value = event.get("layer")
        try:
            layer = Layer(str(layer_value)) if layer_value is not None else None
        except ValueError:
            layer = None
        details = {
            key: value
            for key, value in event.items()
            if key not in {"run_id", "case_id", "state", "node", "layer"}
        }
        repository.record(
            run_id=str(event["run_id"]),
            case_id=str(event["case_id"]),
            event_type=f"hamilton_{event['state']}",
            state=state,
            stage=str(event["node"]),
            layer=layer,
            details=details,
        )

    return sink


class HamiltonObservedDriver:
    """Driver proxy that flushes cache events and clears transient state."""

    def __init__(self, driver: Any, recorder: HamiltonTelemetryRecorder) -> None:
        self._driver = driver
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._driver, name)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._driver.execute(*args, **kwargs)
        finally:
            self._flush_cache_events()

    def materialize(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._driver.materialize(*args, **kwargs)
        finally:
            self._flush_cache_events()

    def _flush_cache_events(self) -> None:
        try:
            cache = getattr(self._driver, "cache", None)
            if cache is None or not getattr(cache, "run_ids", None):
                return
            grouped = cache.logs(cache.last_run_id, level="debug")
            events = sorted(
                (event for values in grouped.values() for event in values),
                key=lambda event: float(event.timestamp),
            )
            self._recorder.cache_events(events, result_store=cache.result_store)
        finally:
            self._recorder.finish_run()


def _serialized_size(result_store: Any | None, data_version: Any) -> int:
    if result_store is None or not isinstance(data_version, str):
        return 0
    path = getattr(result_store, "path", None)
    if path is None:
        return 0
    candidate = Path(path) / data_version
    try:
        return candidate.stat().st_size if candidate.is_file() else 0
    except OSError:
        return 0
