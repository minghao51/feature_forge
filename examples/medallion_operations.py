"""Runnable orchestration helpers for the opt-in medallion path.

The application must provide a module-level ``LayerExecutor`` that writes and
returns valid Bronze, Silver, Gold, and Platinum packages. Feature Forge does
not ship a turnkey durable materializer.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from feature_forge.contracts import GoldPackage, ManifestRef
from feature_forge.contracts.orchestration import ResumePlan, ResumePolicy
from feature_forge.contracts.stages import Layer
from feature_forge.dataflows.gold import replay_gold_package
from feature_forge.dataflows.silver import load_silver_package
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.experiment.case_executor import LayerExecutor
from feature_forge.experiment.resume import build_resume_plan
from feature_forge.platform import ExperimentalPlatform
from feature_forge.storage.local import LocalArtifactStore


def layer_fingerprints(
    run_id: str,
    *,
    bronze: str,
    silver: str,
    gold: str,
    platinum: str,
) -> dict[str, dict[str, str]]:
    """Build the exact per-run fingerprint shape required by the platform."""
    return {
        run_id: {
            Layer.BRONZE.value: bronze,
            Layer.SILVER.value: silver,
            Layer.GOLD.value: gold,
            Layer.PLATINUM.value: platinum,
        }
    }


def plan_durable_run(
    artifact_root: str | Path,
    fingerprints: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    """Resolve a side-effect-free durable plan for the built-in Titanic case."""
    return ExperimentalPlatform().run(
        datasets=["titanic"],
        methods=["openfe"],
        models=["xgboost"],
        seeds=[42],
        artifact_policy="layer_boundaries",
        artifact_root=artifact_root,
        layer_fingerprints=fingerprints,
        dry_run=True,
        progress=False,
    )


def run_durable(
    artifact_root: str | Path,
    fingerprints: dict[str, dict[str, str]],
    layer_executor: LayerExecutor,
) -> list[dict[str, object]]:
    """Run with an application-owned package materializer."""
    return ExperimentalPlatform().run(
        datasets=["titanic"],
        methods=["openfe"],
        models=["xgboost"],
        seeds=[42],
        artifact_policy="layer_boundaries",
        artifact_root=artifact_root,
        layer_fingerprints=fingerprints,
        layer_executor=layer_executor,
        resume_policy=ResumePolicy(enabled=True),
        progress=False,
    )


def inspect_resume(
    artifact_root: str | Path,
    run_id: str,
    expected_fingerprints: Mapping[Layer, str],
) -> ResumePlan:
    """Inspect the verified contiguous reusable prefix without writing."""
    return build_resume_plan(
        store=LocalArtifactStore(artifact_root),
        run_id=run_id,
        expected_fingerprints=expected_fingerprints,
        policy=ResumePolicy(enabled=True),
        dry_run=True,
    )


def replay_gold(
    artifact_root: str | Path,
    silver_ref: ManifestRef,
    gold_ref: ManifestRef,
) -> GoldPackage:
    """Replay persisted Gold code offline against verified Silver evidence."""
    store = LocalArtifactStore(artifact_root)
    silver = load_silver_package(store, silver_ref)
    return replay_gold_package(
        store,
        gold_ref,
        silver,
        SandboxedExecutor(timeout_seconds=5),
    )


def recover_abandoned(
    artifact_root: str | Path,
    *,
    retain_staging_seconds: float = 86_400,
    stale_lock_seconds: float = 3_600,
) -> dict[str, list[str]]:
    """Remove only old local staging/lock owners proven dead by the store."""
    return LocalArtifactStore(artifact_root).recover_abandoned(
        retain_staging_seconds=retain_staging_seconds,
        stale_lock_seconds=stale_lock_seconds,
    )
