"""Stage-specific Hamilton driver factories.

Every stage builder constructs a driver from exactly one node module so the
``HamiltonLayerExecutor`` can verify a stage boundary before the next stage
executes. Cache behavior honors ``Settings.dataflow.cache``; the default cache
location is ``<artifact_root>/control/cache/hamilton``.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from feature_forge.config import Settings
from feature_forge.dataflows.profile import ExecutionProfile, ProfilePolicy, get_profile_policy
from feature_forge.storage.local import LocalArtifactStore

BRONZE_FINAL_VARS = [
    "dataset_computation_request",
    "raw_dataset",
    "bronze_snapshot",
    "source_metadata",
    "bronze_materialization",
]
SILVER_FINAL_VARS = ["silver_materialization"]
GOLD_FINAL_VARS = ["gold_materialization"]
PLATINUM_FINAL_VARS = ["platinum_materialization"]

DEFAULT_ARTIFACT_ROOT = Path("experiments/artifacts")


def resolve_cache_path(
    *,
    settings: Settings | None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve the Hamilton cache directory.

    Precedence is run override, then ``Settings.dataflow.cache.path``, then
    ``<resolved artifact_root>/control/cache/hamilton``.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    if settings is not None and settings.dataflow.cache.path is not None:
        return settings.dataflow.cache.path.expanduser().resolve()
    artifact_root = (
        settings.dataflow.artifact_root if settings is not None else DEFAULT_ARTIFACT_ROOT
    )
    return Path(artifact_root).resolve() / "control" / "cache" / "hamilton"


def _cache_enabled(policy: ProfilePolicy, settings: Settings | None) -> bool:
    if not policy.cache_enabled:
        return False
    return settings.dataflow.cache.enabled if settings is not None else True


def _log_to_file(settings: Settings | None) -> bool:
    return settings.dataflow.cache.log_to_file if settings is not None else False


def _import_builder() -> Any:
    try:
        from hamilton import driver  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on install health
        raise RuntimeError(
            "Hamilton is a core Feature Forge dependency; reinstall with `uv sync`"
        ) from exc
    return driver


def _settings_config(settings: Settings | None) -> dict[str, Any]:
    if settings is None:
        return {}
    return {
        "task": settings.task,
        "metric": settings.metric,
        "cv_folds": settings.evaluation.cv_folds,
    }


def _build_stage_driver(
    modules: tuple[ModuleType, ...],
    *,
    profile: ExecutionProfile,
    settings: Settings | None,
    artifact_store: LocalArtifactStore | None,
    adapters: tuple[Any, ...],
    cache_dir: str | Path | None,
    telemetry_recorder: Any | None,
) -> Any:
    """Build one Hamilton driver over the given node modules with shared policy."""
    policy = get_profile_policy(profile)
    if policy.requires_artifact_store and artifact_store is None:
        raise ValueError(f"Profile '{profile.value}' requires an artifact store")
    driver = _import_builder()
    builder = (
        driver.Builder()
        .with_modules(*modules)
        .with_config(
            {
                "execution_profile": profile,
                **_settings_config(settings),
            }
        )
    )
    resolved_adapters = list(adapters)
    if telemetry_recorder is not None:
        from feature_forge.observability.hamilton_adapter import (
            create_hamilton_lifecycle_adapter,
        )

        resolved_adapters.append(create_hamilton_lifecycle_adapter(telemetry_recorder))
    if resolved_adapters:
        builder = builder.with_adapters(*resolved_adapters)
    if _cache_enabled(policy, settings):
        from feature_forge.storage.hamilton_cache import hamilton_cache_stores

        cache_path = resolve_cache_path(settings=settings, cache_dir=cache_dir)
        metadata_store, result_store = hamilton_cache_stores(cache_path)
        builder = builder.with_cache(
            path=cache_path,
            metadata_store=metadata_store,
            result_store=result_store,
            default_behavior="default",
            log_to_file=_log_to_file(settings),
        )
    built = builder.build()
    if telemetry_recorder is not None:
        from feature_forge.observability.hamilton_adapter import HamiltonObservedDriver

        return HamiltonObservedDriver(built, telemetry_recorder)
    return built


def _reject_lifecycle() -> None:
    raise RuntimeError(
        "Lifecycle-observed drivers arrive with HamiltonLayerExecutor; pass lifecycle "
        "adapters explicitly via `adapters=` instead"
    )


def build_bronze_driver(
    *,
    profile: ExecutionProfile,
    settings: Settings | None = None,
    artifact_store: LocalArtifactStore | None = None,
    adapters: tuple[Any, ...] = (),
    cache_dir: str | Path | None = None,
    modules: tuple[ModuleType, ...] = (),
    telemetry_recorder: Any | None = None,
) -> Any:
    """Build the Bronze source driver; it loads only the Bronze node module."""
    from feature_forge.dataflows import bronze

    return _build_stage_driver(
        (bronze, *modules),
        profile=profile,
        settings=settings,
        artifact_store=artifact_store,
        adapters=adapters,
        cache_dir=cache_dir,
        telemetry_recorder=telemetry_recorder,
    )


def build_silver_driver(
    *,
    profile: ExecutionProfile,
    settings: Settings | None = None,
    artifact_store: LocalArtifactStore | None = None,
    adapters: tuple[Any, ...] = (),
    cache_dir: str | Path | None = None,
    modules: tuple[ModuleType, ...] = (),
    telemetry_recorder: Any | None = None,
) -> Any:
    """Build the Silver canonicalization driver; it loads only the Silver module.

    The Silver driver consumes the Bronze driver's outputs as execution inputs
    (``raw_dataset``, ``bronze_snapshot``, ``source_metadata``,
    ``bronze_materialization``) and never reloads the dataset source.
    """
    from feature_forge.dataflows import silver

    return _build_stage_driver(
        (silver, *modules),
        profile=profile,
        settings=settings,
        artifact_store=artifact_store,
        adapters=adapters,
        cache_dir=cache_dir,
        telemetry_recorder=telemetry_recorder,
    )


def build_gold_driver(
    *,
    profile: ExecutionProfile,
    settings: Settings | None = None,
    artifact_store: LocalArtifactStore | None = None,
    adapters: tuple[Any, ...] = (),
    cache_dir: str | Path | None = None,
    modules: tuple[ModuleType, ...] = (),
    telemetry_recorder: Any | None = None,
) -> Any:
    """Build the provider-free Gold DAG; it loads only the Gold node module."""
    from feature_forge.dataflows import gold

    return _build_stage_driver(
        (gold, *modules),
        profile=profile,
        settings=settings,
        artifact_store=artifact_store,
        adapters=adapters,
        cache_dir=cache_dir,
        telemetry_recorder=telemetry_recorder,
    )


def build_platinum_driver(
    *,
    profile: ExecutionProfile,
    settings: Settings | None = None,
    artifact_store: LocalArtifactStore | None = None,
    adapters: tuple[Any, ...] = (),
    cache_dir: str | Path | None = None,
    modules: tuple[ModuleType, ...] = (),
    telemetry_recorder: Any | None = None,
) -> Any:
    """Build the deterministic Platinum DAG; it loads only the Platinum module."""
    from feature_forge.dataflows import platinum

    return _build_stage_driver(
        (platinum, *modules),
        profile=profile,
        settings=settings,
        artifact_store=artifact_store,
        adapters=adapters,
        cache_dir=cache_dir,
        telemetry_recorder=telemetry_recorder,
    )


def build_driver(
    *,
    profile: ExecutionProfile,
    settings: Settings | None = None,
    artifact_store: LocalArtifactStore | None = None,
    adapters: tuple[Any, ...] = (),
    modules: tuple[ModuleType, ...] = (),
    cache_dir: str | Path | None = None,
    lifecycle_repository: Any | None = None,
    run_id: str | None = None,
    case_id: str | None = None,
) -> Any:
    """Build the transitional combined Bronze+Silver driver.

    Kept for compatibility with callers that predate the stage split; new code
    should use :func:`build_bronze_driver` and :func:`build_silver_driver` so
    the Bronze boundary can be verified before Silver executes.
    """
    from feature_forge.dataflows import bronze, silver

    if lifecycle_repository is not None:
        _reject_lifecycle()
    return _build_stage_driver(
        (bronze, silver, *modules),
        profile=profile,
        settings=settings,
        artifact_store=artifact_store,
        adapters=adapters,
        cache_dir=cache_dir,
        telemetry_recorder=None,
    )
