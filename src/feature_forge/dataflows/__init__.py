"""Optional Hamilton dataflows with lazy public exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ExecutionProfile",
    "ProfilePolicy",
    "build_driver",
    "execute_gold",
    "execute_platinum",
    "load_gold_package",
    "load_platinum_package",
    "load_silver_package",
    "replay_gold_package",
]


def __getattr__(name: str) -> Any:
    if name in {"ExecutionProfile", "ProfilePolicy"}:
        from feature_forge.dataflows import profile

        return getattr(profile, name)
    if name == "build_driver":
        from feature_forge.dataflows.driver import build_driver

        return build_driver
    if name in {"execute_gold", "load_gold_package", "replay_gold_package"}:
        from feature_forge.dataflows import gold

        return getattr(gold, name)
    if name in {"execute_platinum", "load_platinum_package"}:
        from feature_forge.dataflows import platinum

        return getattr(platinum, name)
    if name == "load_silver_package":
        from feature_forge.dataflows.silver import load_silver_package

        return load_silver_package
    raise AttributeError(name)
