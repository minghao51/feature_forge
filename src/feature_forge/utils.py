"""Shared utilities used across feature_forge."""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")


class _AsyncBridge:
    """Thread-safe async-to-sync bridge backed by a daemon event loop.

    Lazily creates a background thread running an asyncio event loop
    that survives the lifetime of the process. Thread-safe via a lock.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _start_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()
        loop.close()

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        with self._lock:
            if self._loop is not None:
                return self._loop
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._start_loop, name="feature-forge-async-runner", daemon=True
            )
            self._thread.start()
            self._ready.wait(timeout=5.0)
            if self._loop is None:
                raise RuntimeError("Failed to initialize shared async runner loop")
            return self._loop

    def run_coro_sync(self, coro: Coroutine[object, object, T]) -> T:
        """Run an async coroutine from synchronous code.

        Handles both the case where no event loop is running (creates one
        via asyncio.run) and the case where one already exists (uses the
        shared background loop via run_coroutine_threadsafe).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        loop = self._get_loop()
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def shutdown(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None


_bridge = _AsyncBridge()
atexit.register(_bridge.shutdown)


def run_coro_sync(coro: Coroutine[object, object, T]) -> T:
    """Run coroutine from sync code, including running-event-loop contexts."""
    return _bridge.run_coro_sync(coro)


def strip_markdown_fences(code: str) -> str:
    """Remove leading/trailing markdown code fences from a string.

    Handles both `````python`` and bare ``````` fences.
    """
    if code.startswith("```"):
        code = code.removeprefix("```python").removeprefix("```")
        code = code.removesuffix("```").strip()
    return code


def _create_lazy_getattr(lazy_map: dict[str, str], module_name: str) -> Callable[[str], Any]:
    """Return a ``__getattr__`` for module-level lazy imports.

    Each key maps an attribute name to a ``"pkg.module"`` path whose
    top-level namesake is imported on first access.
    """
    import importlib

    def __getattr__(name: str) -> Any:
        if name in lazy_map:
            mod = importlib.import_module(lazy_map[name])
            return getattr(mod, name)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__
