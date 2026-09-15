"""Atomic local lifecycle journal for worker attempt state."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from feature_forge.contracts.orchestration import RunEvent
from feature_forge.contracts.stages import RunState
from feature_forge.storage.atomic import atomic_write_json


class LocalRunRepository:
    """Append-only per-attempt events plus an atomic current-state snapshot."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def record(
        self,
        *,
        run_id: str,
        case_id: str,
        event_type: str,
        state: RunState | None = None,
        **values: Any,
    ) -> RunEvent:
        event = RunEvent(
            event_id=uuid.uuid4().hex,
            run_id=run_id,
            case_id=case_id,
            event_type=event_type,
            occurred_at=datetime.now(UTC),
            state=state,
            **values,
        )
        run_dir = self.root / "runs" / event.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n"
        with self._lock:
            with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            atomic_write_json(run_dir / "state.json", event.model_dump(mode="json"))
        return event

    def load_events(self, run_id: str) -> list[RunEvent]:
        path = self.root / "runs" / run_id / "events.jsonl"
        if not path.is_file():
            return []
        return [
            RunEvent.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
