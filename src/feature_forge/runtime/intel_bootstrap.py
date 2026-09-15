"""Intel / OpenMP runtime bootstrap for tabular model acceleration.

Why this module exists
----------------------
Intel Extension for Scikit-learn (``sklearnex``) patches scikit-learn to run on
the Intel oneAPI / DAAL backend, which loads the **Intel OpenMP runtime
(libiomp5)**. The pip wheels of XGBoost and LightGBM are built against the
**GNU OpenMP runtime (libgomp)**. When two OpenMP runtimes are loaded into the
same Python process and both try to drive threads at the same time (e.g. a
booster fit with ``n_jobs > 1`` inside a ``sklearnex``-patched CV loop, or
sklearnex + libgomp estimators evaluated under a ``threading`` joblib backend),
they deadlock: the process spins at 100% CPU and never returns.

The mitigation has three parts, all applied here **before any
numpy / sklearn / xgboost / lightgbm import**:

1. **Patch scikit-learn first.** ``patch_sklearn()`` must run before any
   ``sklearn`` (or xgboost/lightgbm) module is imported, so the patched
   estimators are the ones actually used. This is the import-order rule the
   project follows.
2. **Set ``KMP_DUPLICATE_LIB_OK=TRUE``** before libiomp5 is loaded, so a
   secondary OpenMP load does not raise ``OMP: Error #15`` and abort. This is a
   safety net, not a cure for the deadlock.
3. **Keep a single OpenMP runtime actually threading.** Boosting estimators are
   pinned to ``n_jobs=1`` (see ``evaluation/model_factory.py``) and experiment
   evaluation uses process isolation (``backend="loky"``) when Intel
   acceleration is enabled (see ``methods/malmas/pipeline/core.py``), so the
   libiomp5 and libgomp runtimes never share one thread pool. The only way to
   fully eliminate the conflict is to install the Intel-optimized XGBoost /
   LightGBM builds (which also link libiomp5) via the ``intel`` conda channel.

See ``docs/decisions/0010-intel-openmp-bootstrap.md``.
"""

from __future__ import annotations

import os
from typing import Any

# Set before importing anything that could pull in an OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)

# Whether Intel acceleration was actually engaged (sklearnex patched) in this
# process. The deadlock risk only exists once libiomp5 is loaded, so the
# backend-isolation guard in the pipeline keys off this, not merely the config
# flag. This keeps default behavior unchanged when the `intel` extra is absent.
_INTEL_ACTIVE = False


def is_intel_active() -> bool:
    """Return True if Intel acceleration was successfully engaged this process."""
    return _INTEL_ACTIVE


def bootstrap_intel_runtime(*, enabled: bool | None = None) -> bool:
    """Initialize Intel scikit-learn acceleration and OpenMP safety.

    Must be called as the very first import-time action in the package (see
    ``feature_forge/__init__.py``) so ``patch_sklearn()`` executes before any
    ``sklearn`` / ``xgboost`` / ``lightgbm`` module is imported elsewhere.

    Args:
        enabled: Override the ``intel_acceleration`` setting. When ``None``, the
            value is read from :func:`feature_forge.config.get_settings`.

    Returns:
        ``True`` if Intel acceleration was successfully enabled, else ``False``.

    Raises:
        ImportError: Only if the setting is enabled *and* the patch call itself
            fails for a reason other than a missing package.
    """
    # Re-assert the safety net even if called outside the package __init__.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if enabled is None:
        from feature_forge.config import get_settings

        try:
            enabled = get_settings().intel_acceleration
        except Exception as exc:
            # settings file (bad YAML, unreadable .env) must never crash the
            # whole package import; acceleration is an optimization, so fail
            # closed to the stock libgomp stack.
            logger.warning(
                "intel_bootstrap_settings_error",
                error=str(exc)[:200],
                enabled=False,
                hint="fix config/settings.yaml or .env; Intel acceleration stays off",
            )
            return False

    if not enabled:
        logger.debug("intel_bootstrap_skipped", reason="disabled")
        return False

    try:
        from sklearnex import patch_sklearn
    except ImportError:
        logger.warning(
            "intel_bootstrap_unavailable",
            reason=(
                "scikit-learn-intelex not installed; install with "
                "`pip install 'feature-forge[intel]'` or `uv sync --extra intel` "
                "to enable Intel acceleration"
            ),
        )
        return False

    # Patch BEFORE any sklearn/xgboost/lightgbm import downstream.
    # verbose=False: keep stdout clean (sklearnex prints a banner by default);
    # the structured log line below is the single activation record.
    patch_sklearn(verbose=False)
    global _INTEL_ACTIVE
    _INTEL_ACTIVE = True
    logger.info("intel_bootstrap_enabled", sklearn_patched=True)
    return True


def runtime_info() -> dict[str, Any]:
    """Runtime provenance for run metadata (ADR 0010 follow-up).

    Distinguishes *actual* activation from the config flag: the flag may be
    true while the ``intel`` extra is absent (bootstrap no-op), and activation
    state is process-global (import-time). Include in tracker/run metadata
    so experiment results record which numeric backend produced them.
    """
    info: dict[str, Any] = {"intel_active": _INTEL_ACTIVE}
    if _INTEL_ACTIVE:
        try:
            from importlib.metadata import version

            info["sklearnex_version"] = version("scikit-learn-intelex")
        except Exception:
            info["sklearnex_version"] = None
    return info
