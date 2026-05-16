"""Shared entry-point discovery for metric and model registries."""

from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Callable
from typing import Any


def discover_entry_points(
    group: str,
    builtins: dict[str, Any] | None = None,
) -> dict[str, Callable[..., Any]]:
    discovered: dict[str, Callable[..., Any]] = {}
    for ep in importlib.metadata.entry_points(group=group):
        try:
            loaded = ep.load()
        except Exception as exc:
            warnings.warn(
                f"Failed to load entry point '{ep.name}': {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        if builtins and ep.name in builtins and builtins[ep.name] is not loaded:
            warnings.warn(
                f"Entry point '{ep.name}' overrides built-in.",
                RuntimeWarning,
                stacklevel=3,
            )
        if ep.name in discovered:
            warnings.warn(
                f"Duplicate entry point name '{ep.name}'. Last registered wins.",
                RuntimeWarning,
                stacklevel=3,
            )
        discovered[ep.name] = loaded
    return discovered
