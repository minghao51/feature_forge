import importlib.resources
from pathlib import Path

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
    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir
        self._cache: dict[str, Prompt] = {}

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
