"""Isolation helpers for offline generated documentation."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from unittest.mock import patch

_PRESERVED_ENVIRONMENT = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "TMPDIR",
    "TEMP",
    "TMP",
)


def scrubbed_subprocess_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return only process-launch settings, never application configuration or secrets."""
    values = source or os.environ
    result = {name: values[name] for name in _PRESERVED_ENVIRONMENT if name in values}
    result.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0"})
    return result


@contextmanager
def scrubbed_environment() -> Iterator[None]:
    """Temporarily remove application configuration and credentials from the process."""
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(scrubbed_subprocess_environment(previous))
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _network_forbidden(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise RuntimeError("Generated documentation forbids network access")


@contextmanager
def deny_network() -> Iterator[None]:
    """Fail any Python socket connection attempt during documentation generation."""
    with (
        patch.object(socket.socket, "connect", _network_forbidden),
        patch.object(socket.socket, "connect_ex", _network_forbidden),
        patch.object(socket, "create_connection", _network_forbidden),
    ):
        yield
