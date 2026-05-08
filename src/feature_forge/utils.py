"""Shared utilities used across feature_forge."""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import TypeVar

T = TypeVar("T")

_LOOP_THREAD: threading.Thread | None = None
_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_READY = threading.Event()
_LOOP_LOCK = threading.Lock()


def _loop_thread_main() -> None:
    global _LOOP
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _LOOP = loop
    _LOOP_READY.set()
    loop.run_forever()
    loop.close()


def _get_shared_event_loop() -> asyncio.AbstractEventLoop:
    global _LOOP_THREAD
    if _LOOP is not None:
        return _LOOP
    with _LOOP_LOCK:
        if _LOOP is not None:
            return _LOOP
        _LOOP_READY.clear()
        _LOOP_THREAD = threading.Thread(
            target=_loop_thread_main, name="feature-forge-async-runner", daemon=True
        )
        _LOOP_THREAD.start()
        _LOOP_READY.wait(timeout=5.0)
        if _LOOP is None:
            raise RuntimeError("Failed to initialize shared async runner loop")
        return _LOOP


def _shutdown_shared_event_loop() -> None:
    global _LOOP
    if _LOOP is not None:
        _LOOP.call_soon_threadsafe(_LOOP.stop)
        _LOOP = None


atexit.register(_shutdown_shared_event_loop)


def strip_markdown_fences(code: str) -> str:
    """Remove leading/trailing markdown code fences from a string.

    Handles both `````python`` and bare ``````` fences.
    """
    if code.startswith("```"):
        code = code.removeprefix("```python").removeprefix("```")
        code = code.removesuffix("```").strip()
    return code


def run_coro_sync(coro: Coroutine[object, object, T]) -> T:
    """Run coroutine from sync code, including running-event-loop contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = _get_shared_event_loop()
    future: Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()
