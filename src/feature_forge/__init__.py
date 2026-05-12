"""Feature Forge: Modular experimentation platform for LLM-based multi-agent automated feature engineering."""

import os as _os
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

_os.environ.setdefault("FF_LOG_LEVEL", "warning")

from feature_forge.observability.structlog_config import configure_logging

configure_logging()

try:
    __version__ = _version("feature-forge")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
