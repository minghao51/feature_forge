"""Feature Forge: Modular experimentation platform for LLM-based multi-agent automated feature engineering."""

import os as _os
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from feature_forge.observability.structlog_config import configure_logging
from feature_forge.runtime.intel_bootstrap import bootstrap_intel_runtime

# (1) OpenMP / Intel safety must be set before any numpy/sklearn/xgboost/lightgbm
#     import in this process. See docs/decisions/0010-intel-openmp-bootstrap.md.
_os.environ.setdefault("FF_LOG_LEVEL", "warning")
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

configure_logging()

# (2) Patch Intel scikit-learn FIRST, before any sklearn/xgboost/lightgbm import.
bootstrap_intel_runtime()

# These submodule imports transitively import sklearn/xgboost/lightgbm, so they
# must run AFTER bootstrap_intel_runtime() (hence the E402 suppression).
from feature_forge.evaluation.kit import EvaluationKit  # noqa: E402
from feature_forge.methods.malmas.pipeline.result import PipelineResult  # noqa: E402
from feature_forge.platform import ExperimentalPlatform  # noqa: E402

try:
    __version__ = _version("feature-forge")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["EvaluationKit", "ExperimentalPlatform", "PipelineResult", "__version__"]
