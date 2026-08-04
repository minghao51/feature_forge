"""Redacted Hamilton lifecycle telemetry with optional cache-event enrichment."""

from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Any, Protocol

from feature_forge.contracts.stages import Layer, RunState
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|"
    r"openai_api_key|anthropic_api_key|gemini_api_key|deepseek_api_key|provider[_-]?key)"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+\-/=]+")
_PROVIDER_KEY = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b")
_ALLOWED_TAGS = {"layer", "cost", "persistence", "sensitivity", "owner"}


class TelemetrySink(Protocol):
    def __call__(self, event: dict[str, Any]) -> None: ...


class HamiltonTelemetryRecorder:
    """Thread-safe domain recorder that never records node inputs or results."""

    def __init__(self, *, run_id: str, case_id: str, sink: TelemetrySink | None = None) -> None:
        self.run_id = run_id
        self.case_id = case_id
        self.sink = sink or self._log
        self._started: dict[tuple[str, str | None, str], float] = {}
        self._lock = threading.Lock()

    def before(
        self, *, hamilton_run_id: str, node_name: str, task_id: str | None, tags: dict[str, Any]
    ) -> None:
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
        key = (hamilton_run_id, task_id, node_name)
        with self._lock:
            started = self._started.pop(key, None)
        duration_ms = None if started is None else (time.monotonic() - started) * 1000
        self._emit(
            "node_succeeded" if success else "node_failed",
            hamilton_run_id,
            node_name,
            task_id,
            tags,
            duration_ms=duration_ms,
            error_type=type(error).__name__ if error else None,
            error_message=_redact(str(error)) if error else None,
        )

    def cache_events(self, events: list[Any]) -> None:
        """Emit cache outcomes from Hamilton's official cache event records."""
        for event in events:
            event_type = getattr(getattr(event, "event_type", None), "value", "")
            if event_type not in {"get_result", "execute_node", "missing_result"}:
                continue
            outcome = {"get_result": "hit", "execute_node": "miss", "missing_result": "missing"}[
                event_type
            ]
            self._emit(
                "node_cache",
                str(getattr(event, "run_id", "")),
                str(getattr(event, "node_name", "")),
                getattr(event, "task_id", None),
                {},
                cache_outcome=outcome,
            )

    def _emit(
        self,
        state: str,
        hamilton_run_id: str,
        node_name: str,
        task_id: str | None,
        tags: dict[str, Any],
        **values: Any,
    ) -> None:
        allowed = {key: _redact(str(value)) for key, value in tags.items() if key in _ALLOWED_TAGS}
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
    """Create the Hamilton adapter only when explicitly requested."""
    try:
        from hamilton.lifecycle import NodeExecutionHook
    except ImportError as exc:  # pragma: no cover - optional installation
        raise RuntimeError(
            "Hamilton observability is optional; run `uv sync --extra pipeline`"
        ) from exc

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
                hamilton_run_id=run_id, node_name=node_name, task_id=task_id, tags=node_tags
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
    """Bridge redacted node telemetry into the authoritative PR5 journal."""

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
    """Driver proxy that flushes official cache events after every execution."""

    def __init__(self, driver: Any, recorder: HamiltonTelemetryRecorder) -> None:
        self._driver = driver
        self._recorder = recorder
        self._seen_cache_events: set[tuple[str, str, str | None, float]] = set()

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
        cache = getattr(self._driver, "cache", None)
        if cache is None or not getattr(cache, "run_ids", None):
            return
        grouped = cache.logs(cache.last_run_id, level="debug")
        events: list[Any] = []
        for values in grouped.values():
            for event in values:
                key = (
                    str(event.run_id),
                    str(event.node_name),
                    event.task_id,
                    float(event.timestamp),
                )
                if key not in self._seen_cache_events:
                    self._seen_cache_events.add(key)
                    events.append(event)
        self._recorder.cache_events(events)


def _redact(message: str) -> str:
    redacted = _SECRET_ASSIGNMENT.sub(lambda match: match.group(1) + "=<redacted>", message)
    redacted = _BEARER.sub("Bearer <redacted>", redacted)
    return _PROVIDER_KEY.sub("<redacted>", redacted)[:500]
