"""Tests for shared utility helpers."""

from __future__ import annotations

import asyncio

from feature_forge.utils import run_coro_sync


async def _echo(value: int) -> int:
    await asyncio.sleep(0.01)
    return value


def test_run_coro_sync_outside_event_loop() -> None:
    assert run_coro_sync(_echo(7)) == 7


def test_run_coro_sync_repeated_calls() -> None:
    assert run_coro_sync(_echo(1)) == 1
    assert run_coro_sync(_echo(2)) == 2
