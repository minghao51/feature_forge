from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, field_validator


def prompts_dir(package: str) -> Path:
    return Path(str(importlib.resources.files(package)))


class Prompt(BaseModel):
    system: str
    description: str = ""

    @field_validator("system")
    @classmethod
    def _system_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt system must be non-empty")
        return value


class PromptRegistry:
    # Process-level singletons keyed by package name, so each method package
    # (caafe/llmfe/malmus/malmas) gets one registry without hand-rolling a
    # module-global ``_registry`` + ``get_registry()`` pair.
    _instances: ClassVar[dict[str, PromptRegistry]] = {}

    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir
        self._cache: dict[str, Prompt] = {}

    @classmethod
    def for_package(cls, package: str) -> PromptRegistry:
        """Return the cached registry for a method's prompt package."""
        existing = cls._instances.get(package)
        if existing is None:
            existing = cls(prompts_dir(package))
            cls._instances[package] = existing
        return existing

    def get(self, name: str) -> Prompt:
        if name not in self._cache:
            path = self._dir / f"{name}.yaml"
            if not path.exists():
                raise KeyError(f"Prompt '{name}' not found at {path}")
            with open(path, encoding="utf-8") as file:
                data = yaml.safe_load(file)
            self._cache[name] = Prompt(**data)
        return self._cache[name]

    def clear_cache(self) -> None:
        self._cache.clear()
