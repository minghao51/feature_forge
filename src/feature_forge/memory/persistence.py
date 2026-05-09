"""Memory persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class MemoryPersistence:
    """JSON-based persistence for agent memory."""

    def __init__(self, memory_path: str) -> None:
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict[str, Any]) -> None:
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug("memory_save", path=str(self.memory_path), num_keys=len(data))

    def load(self) -> dict[str, Any] | None:
        if not self.memory_path.exists():
            logger.debug("memory_load", path=str(self.memory_path), exists=False)
            return None
        with open(self.memory_path, encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("memory_load", path=str(self.memory_path), exists=True, num_keys=len(data))
        return data
