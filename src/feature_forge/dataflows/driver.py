"""Hamilton driver factory for the optional dataflow profile."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from feature_forge.config import Settings
from feature_forge.dataflows.profile import ExecutionProfile, get_profile_policy
from feature_forge.storage.local import LocalArtifactStore

if TYPE_CHECKING:
    from feature_forge.experiment.lifecycle import LocalRunRepository

SILVER_FINAL_VARS = ["silver_materialization"]
GOLD_FINAL_VARS = ["gold_materialization"]
PLATINUM_FINAL_VARS = ["execute_platinum"]
SILVER_CACHEABLE_NODES = [
    "bronze_snapshot",
    "bronze_checks",
    "source_metadata",
    "dataset_fingerprint_value",
    "canonical_features",
    "canonical_target",
    "row_ids",
    "fold_assignments",
    "dataset_profile",
    "silver_checks",
]


def build_driver(
    *,
    profile: ExecutionProfile,
    settings: Settings | None = None,
    artifact_store: LocalArtifactStore | None = None,
    adapters: tuple[Any, ...] = (),
    modules: tuple[ModuleType, ...] = (),
    cache_dir: str | Path | None = None,
    lifecycle_repository: LocalRunRepository | None = None,
    run_id: str | None = None,
    case_id: str | None = None,
) -> Any:
    """Build one Hamilton driver with an isolated profile context.

    Hamilton remains an optional dependency. Importing this module is safe in a
    standard Feature Forge installation; calling this function requires the
    ``pipeline`` extra.
    """
    policy = get_profile_policy(profile)
    if policy.requires_artifact_store and artifact_store is None:
        raise ValueError(f"Profile '{profile.value}' requires an artifact store")

    try:
        from hamilton import driver
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Hamilton is optional; install the pipeline extra with `uv sync --extra pipeline`"
        ) from exc

    from feature_forge.dataflows import gold, platinum, silver

    builder = driver.Builder().with_modules(
        silver,
        gold,
        platinum,
        *modules,
    )
    config: dict[str, Any] = {
        "execution_profile": profile,
        "artifact_store": artifact_store,
    }
    if settings is not None:
        config.update(
            {
                "task": settings.task,
                "metric": settings.metric,
                "cv_folds": settings.evaluation.cv_folds,
            }
        )
    builder = builder.with_config(config)
    observed = None
    resolved_adapters = list(adapters)
    if lifecycle_repository is not None:
        if run_id is None or case_id is None:
            raise ValueError("lifecycle_repository requires run_id and case_id")
        from feature_forge.observability.hamilton_adapter import (
            HamiltonObservedDriver,
            HamiltonTelemetryRecorder,
            create_hamilton_lifecycle_adapter,
            create_run_event_sink,
        )

        observed = HamiltonTelemetryRecorder(
            run_id=run_id,
            case_id=case_id,
            sink=create_run_event_sink(lifecycle_repository),
        )
        resolved_adapters.append(create_hamilton_lifecycle_adapter(observed))
    if resolved_adapters:
        builder = builder.with_adapters(*resolved_adapters)
    if policy.cache_enabled:
        builder = builder.with_cache(
            path=cache_dir or ".feature_forge_artifacts/hamilton-cache",
            default=SILVER_CACHEABLE_NODES,
            default_behavior="disable",
        )
    built = builder.build()
    if observed is not None:
        return HamiltonObservedDriver(built, observed)
    return built
