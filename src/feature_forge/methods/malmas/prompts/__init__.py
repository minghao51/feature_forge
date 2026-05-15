"""Prompt registry with YAML-backed loading and Pydantic validation.

Each prompt is stored as a separate ``.yaml`` file under ``config/prompts/``.
The ``PromptRegistry`` loads them lazily and caches the result.

Usage::

    from feature_forge.prompts import get_registry

    prompt = get_registry().get("unary")
    system_text = prompt.system
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

_PROMPTS_DIR = (
    Path(str(importlib.resources.files("feature_forge"))).parent.parent / "config" / "prompts"
)


class Prompt(BaseModel):
    """Single prompt with validated structure."""

    system: str
    description: str = ""

    @field_validator("system")
    @classmethod
    def _system_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt system must be a non-empty string")
        return v


class PromptRegistry:
    """Load and cache individual prompt YAML files from config/prompts/."""

    def __init__(self, prompts_dir: Path = _PROMPTS_DIR) -> None:
        self._dir = prompts_dir
        self._cache: dict[str, Prompt] = {}

    def get(self, name: str) -> Prompt:
        if name not in self._cache:
            path = self._dir / f"{name}.yaml"
            if not path.exists():
                raise KeyError(f"Prompt '{name}' not found at {path}")
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._cache[name] = Prompt(**data)
        return self._cache[name]

    def clear_cache(self) -> None:
        self._cache.clear()


_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    """Return the module-level PromptRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
