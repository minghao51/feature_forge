"""Feature Forge: Modular experimentation platform for LLM-based multi-agent automated feature engineering."""

import os as _os
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import Any

_os.environ.setdefault("FF_LOG_LEVEL", "warning")

# Public attributes are resolved lazily via __getattr__ below. Eagerly
# importing EvaluationKit / ExperimentalPlatform / PipelineResult here would
# force every ``multiprocessing.spawn`` worker (sandbox, scheduler) to import
# the full feature_forge graph (numpy, pandas, sklearn, duckdb, ...) just to
# unpickle a target function, which on memory-constrained CI runners pushes
# the worker past OOM at numpy import time. Keeping __init__.py import-free
# keeps the spawn worker's RSS small enough to start reliably.

try:
    __version__ = _version("feature-forge")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["EvaluationKit", "ExperimentalPlatform", "PipelineResult", "__version__"]


def __getattr__(name: str) -> Any:
    # PEP 562: resolve public names lazily so importing feature_forge itself
    # does not pull in the heavy dependency graph.
    if name in ("EvaluationKit", "ExperimentalPlatform", "PipelineResult"):
        # First access of any public name configures logging once in this
        # process. Spawn workers never touch these names, so they skip both
        # the heavy imports and the logging side effect.
        from feature_forge.observability.structlog_config import configure_logging

        configure_logging()
        if name == "EvaluationKit":
            from feature_forge.evaluation.kit import EvaluationKit

            return EvaluationKit
        if name == "ExperimentalPlatform":
            from feature_forge.platform import ExperimentalPlatform

            return ExperimentalPlatform
        if name == "PipelineResult":
            from feature_forge.methods.malmas.pipeline.result import PipelineResult

            return PipelineResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
