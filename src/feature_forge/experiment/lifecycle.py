"""Atomic local lifecycle journal for scheduler state and catalog ingestion."""

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
    """Append-only event journal plus atomic current-state snapshots."""

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
        """Append one durable event and update the current snapshot."""
        event = RunEvent(
            event_id=uuid.uuid4().hex,
            run_id=run_id,
            case_id=case_id,
            event_type=event_type,
            occurred_at=datetime.now(UTC),
            state=state,
            **values,
        )
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n"
        with self._lock:
            event_path = run_dir / "events.jsonl"
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            atomic_write_json(run_dir / "state.json", event.model_dump(mode="json"))
        return event

    def load_events(self, run_id: str) -> list[RunEvent]:
        """Load and validate the append-only event sequence."""
        path = self.root / "runs" / run_id / "events.jsonl"
        if not path.is_file():
            return []
        return [
            RunEvent.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
