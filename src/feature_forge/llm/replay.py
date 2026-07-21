"""Process-local guard that makes offline replay fail closed."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from feature_forge.exceptions import ReplayProviderError

_REPLAY_ACTIVE: ContextVar[bool] = ContextVar("feature_forge_replay_active", default=False)


def ensure_provider_allowed(action: str) -> None:
    if _REPLAY_ACTIVE.get():
        raise ReplayProviderError(f"LLM provider {action} is forbidden during offline replay")


@contextmanager
def replay_provider_guard() -> Iterator[None]:
    token = _REPLAY_ACTIVE.set(True)
    try:
        yield
    finally:
        _REPLAY_ACTIVE.reset(token)
