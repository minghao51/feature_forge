"""Feature Forge: Modular experimentation platform for LLM-based multi-agent automated feature engineering."""

import os as _os

_os.environ.setdefault("FF_LOG_LEVEL", "warning")

from feature_forge._version import __version__
from feature_forge.observability.structlog_config import configure_logging

configure_logging()

__all__ = ["__version__"]
