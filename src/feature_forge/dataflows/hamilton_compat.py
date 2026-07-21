"""Small optional-import shim for Hamilton decorators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

T = TypeVar("T", bound=Callable[..., Any])

try:
    from hamilton.function_modifiers import tag as _hamilton_tag
except ImportError:  # pragma: no cover - exercised only without the pipeline extra

    def tag(**_: str) -> Callable[[T], T]:
        """Return a no-op decorator when the optional extra is absent."""

        def decorate(function: T) -> T:
            return function

        return decorate

else:

    def tag(**kwargs: str) -> Callable[[T], T]:
        """Apply Hamilton metadata tags."""
        return cast(Callable[[T], T], _hamilton_tag(**kwargs))
