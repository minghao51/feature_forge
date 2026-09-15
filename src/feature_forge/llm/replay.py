"""Guards that make artifact replay provider-free."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from feature_forge.exceptions import ReplayProviderError

_REPLAY = ContextVar("feature_forge_replay", default=False)


def replay_forbidden() -> bool:
    return _REPLAY.get()


def ensure_provider_allowed(action: str = "provider construction") -> None:
    if replay_forbidden():
        raise ReplayProviderError(f"{action} is forbidden during replay")


@contextmanager
def replay_provider_guard() -> Iterator[None]:
    token = _REPLAY.set(True)
    try:
        yield
    finally:
        _REPLAY.reset(token)
