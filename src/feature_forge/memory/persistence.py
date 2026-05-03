"""Memory persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MemoryPersistence:
    """JSON-based persistence for agent memory."""

    def __init__(self, memory_path: str) -> None:
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict[str, Any]) -> None:
        """Save memory data to JSON file."""
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> dict[str, Any] | None:
        """Load memory data from JSON file."""
        if not self.memory_path.exists():
            return None
        with open(self.memory_path, encoding="utf-8") as f:
            return json.load(f)
