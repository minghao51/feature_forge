"""Shared prompt infrastructure: versioned YAML registry + Jinja2 rendering.

Prompts are colocated with their method (ADR 0005) as versioned YAML
files named ``<name>.v<N>.yaml`` with ``system`` and optional
``user``/``description`` templates (ADR 0008). A bare ``name`` resolves
to the highest version present; ``version=N`` pins an exact file for
reproducible reruns (ADR 0009). Variables are declared on
:class:`PromptParams` subclasses — typed Pydantic domain models — and
rendered through the registry with Jinja2 StrictUndefined, so a missing
variable fails before any LLM call. Every prompt carries a
:class:`PromptProvenance` (name, version, content sha256) that call
sites pass to the LLM client for cache payloads and logs.
"""

import hashlib
import importlib.resources
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


def prompts_dir(package: str) -> Path:
    return Path(str(importlib.resources.files(package)))


class PromptParams(BaseModel):
    """Base for typed prompt injection variables.

    Subclasses declare (and Pydantic validates) the variables a template
    may use; all fields are exposed to the template context.
    """

    def model_context(self) -> dict[str, Any]:
        return self.model_dump()


class PromptProvenance(BaseModel):
    """Identity of the exact prompt template used for an LLM call."""

    prompt_name: str
    prompt_version: int
    prompt_sha256: str


class Prompt(BaseModel):
    system: str
    user: str = ""
    description: str = ""
    version: int = 0
    sha256: str = ""

    @field_validator("system")
    @classmethod
    def _system_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt system must be non-empty")
        return value


class PromptRegistry:
    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir
        self._cache: dict[tuple[str, int], Prompt] = {}
        self._versions: dict[str, list[int]] = {}
        self._env = self._build_env()

    @staticmethod
    def _build_env() -> Any:
        from jinja2 import Environment, StrictUndefined

        return Environment(
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    @staticmethod
    def _content_sha256(prompt: Prompt) -> str:
        payload = json.dumps(
            {"system": prompt.system, "user": prompt.user},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _available_versions(self, name: str) -> list[int]:
        if name not in self._versions:
            self._versions[name] = sorted(
                int(p.stem.rsplit(".v", 1)[1])
                for p in self._dir.glob(f"{name}.v*.yaml")
                if p.stem.rsplit(".v", 1)[1].isdigit()
            )
        return self._versions[name]

    def _resolve_version(self, name: str, version: int | None) -> int:
        available = self._available_versions(name)
        if not available:
            raise KeyError(f"Prompt '{name}' not found at {self._dir}/{name}.v<N>.yaml")
        if version is None:
            return available[-1]
        if version not in available:
            raise KeyError(f"Prompt '{name}' version {version} not found; available: {available}")
        return version

    def get(self, name: str, version: int | None = None) -> Prompt:
        resolved = self._resolve_version(name, version)
        key = (name, resolved)
        if key not in self._cache:
            path = self._dir / f"{name}.v{resolved}.yaml"
            with open(path, encoding="utf-8") as file:
                data = yaml.safe_load(file)
            prompt = Prompt(**data)
            prompt.version = resolved
            prompt.sha256 = self._content_sha256(prompt)
            self._cache[key] = prompt
        return self._cache[key]

    def provenance(self, name: str, version: int | None = None) -> PromptProvenance:
        prompt = self.get(name, version)
        return PromptProvenance(
            prompt_name=name,
            prompt_version=prompt.version,
            prompt_sha256=prompt.sha256,
        )

    def render(
        self,
        name: str,
        params: PromptParams,
        version: int | None = None,
    ) -> str:
        """Render the ``system`` template of prompt ``name`` with ``params``."""
        rendered: str = self._env.from_string(self.get(name, version).system).render(
            **params.model_context()
        )
        return rendered

    def render_messages(
        self,
        name: str,
        params: PromptParams,
        version: int | None = None,
    ) -> list[dict[str, str]]:
        """Render a chat message list: system prompt plus optional user template."""
        prompt = self.get(name, version)
        context = params.model_context()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._env.from_string(prompt.system).render(**context)}
        ]
        if prompt.user.strip():
            messages.append(
                {"role": "user", "content": self._env.from_string(prompt.user).render(**context)}
            )
        return messages

    def clear_cache(self) -> None:
        self._cache.clear()
        self._versions.clear()
