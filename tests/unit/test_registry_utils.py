"""Tests for shared registry entry-point discovery helper."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from feature_forge.evaluation.registry_utils import discover_entry_points


def _fn_a() -> float:
    return 1.0


def _fn_b() -> float:
    return 2.0


def test_discover_entry_points_returns_loaded_callables() -> None:
    ep = SimpleNamespace(name="metric_a", load=lambda: _fn_a)

    with patch("importlib.metadata.entry_points", return_value=[ep]):
        discovered = discover_entry_points("feature_forge.metrics")

    assert discovered == {"metric_a": _fn_a}


def test_discover_entry_points_warns_on_load_failure() -> None:
    def _bad_loader() -> Callable[..., Any]:
        raise RuntimeError("boom")

    ep = SimpleNamespace(name="broken", load=_bad_loader)

    with patch("importlib.metadata.entry_points", return_value=[ep]):
        with pytest.warns(RuntimeWarning, match="Failed to load entry point"):
            discovered = discover_entry_points("feature_forge.metrics")

    assert discovered == {}


def test_discover_entry_points_warns_on_builtin_override() -> None:
    ep = SimpleNamespace(name="metric_a", load=lambda: _fn_b)

    with patch("importlib.metadata.entry_points", return_value=[ep]):
        with pytest.warns(RuntimeWarning, match="overrides built-in"):
            discovered = discover_entry_points(
                "feature_forge.metrics",
                builtins={"metric_a": _fn_a},
            )

    assert discovered["metric_a"] is _fn_b


def test_discover_entry_points_warns_on_duplicate_names() -> None:
    ep1 = SimpleNamespace(name="dup", load=lambda: _fn_a)
    ep2 = SimpleNamespace(name="dup", load=lambda: _fn_b)

    with patch("importlib.metadata.entry_points", return_value=[ep1, ep2]):
        with pytest.warns(RuntimeWarning, match="Duplicate entry point name"):
            discovered = discover_entry_points("feature_forge.metrics")

    assert discovered["dup"] is _fn_b
