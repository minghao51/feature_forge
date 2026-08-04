"""Verified contiguous layer reuse and side-effect-free resume planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from feature_forge.contracts.artifacts import ManifestRef
from feature_forge.contracts.orchestration import (
    ResumePlan,
    ResumePolicy,
    ResumeStageDecision,
    StageDisposition,
)
from feature_forge.contracts.stages import Layer, RunState
from feature_forge.exceptions import DatasetError
from feature_forge.storage.local import LocalArtifactStore

_LAYERS = (Layer.BRONZE, Layer.SILVER, Layer.GOLD, Layer.PLATINUM)


class ResumeExecutionError(RuntimeError):
    """Layer execution failed after producing a durable partial resume plan."""

    def __init__(self, plan: ResumePlan, failed_layer: Layer, cause: Exception) -> None:
        super().__init__(f"{failed_layer.value} resume execution failed: {cause}")
        self.plan = plan
        self.failed_layer = failed_layer
        self.cause = cause


def _load_semantic_package(
    store: LocalArtifactStore,
    ref: ManifestRef,
    chain: Mapping[Layer, ManifestRef],
) -> None:
    """Run the layer loader and require the exact authoritative upstream chain."""
    if ref.layer is Layer.BRONZE:
        manifest = store.load_manifest(ref)
        if manifest.upstream_manifests:
            raise ValueError("Bronze package must not declare upstream manifests")
        return
    if ref.layer is Layer.SILVER:
        from feature_forge.dataflows.silver import load_silver_package

        silver_package = load_silver_package(store, ref)
        expected = chain.get(Layer.BRONZE)
        if expected is None or silver_package.manifest.upstream_manifests != [expected]:
            raise ValueError("Silver package does not reference the exact planned Bronze package")
        return
    if ref.layer is Layer.GOLD:
        from feature_forge.dataflows.gold import load_gold_package

        gold_package = load_gold_package(store, ref)
        expected = chain.get(Layer.SILVER)
        if expected is None or (
            gold_package.request.silver_manifest != expected
            or gold_package.manifest.upstream_manifests != [expected]
        ):
            raise ValueError("Gold package does not reference the exact planned Silver package")
        return
    from feature_forge.dataflows.platinum import load_platinum_package

    platinum_package = load_platinum_package(store, ref)
    expected_silver = chain.get(Layer.SILVER)
    expected_gold = chain.get(Layer.GOLD)
    if (
        expected_silver is None
        or expected_gold is None
        or (
            platinum_package.request.silver_manifest != expected_silver
            or platinum_package.request.gold_manifest != expected_gold
            or platinum_package.manifest.upstream_manifests != [expected_silver, expected_gold]
        )
    ):
        raise ValueError("Platinum package does not reference the exact planned Silver/Gold chain")


def build_resume_plan(
    *,
    store: LocalArtifactStore,
    run_id: str,
    expected_fingerprints: Mapping[Layer, str],
    policy: ResumePolicy,
    dry_run: bool = False,
    semantic_validator: Callable[
        [LocalArtifactStore, ManifestRef, Mapping[Layer, ManifestRef]], None
    ] = _load_semantic_package,
) -> ResumePlan:
    """Select only a fully verified contiguous prefix of compatible packages."""
    decisions: list[ResumeStageDecision] = []
    upstream = None
    chain: dict[Layer, ManifestRef] = {}
    reusable_prefix = policy.enabled
    next_layer: Layer | None = None

    for layer in _LAYERS:
        expected = expected_fingerprints.get(layer)
        if expected is None:
            continue
        ref = None
        reason = "resume disabled"
        if reusable_prefix:
            try:
                ref = store.find_reusable(
                    expected,
                    layer,
                    upstream=upstream,
                    preferred_run_id=run_id,
                    allow_cross_run=policy.allow_cross_run_reuse,
                )
                if ref is not None:
                    semantic_validator(store, ref, chain)
            except (DatasetError, OSError, ValueError) as exc:
                ref = None
                if not policy.recompute_invalid:
                    raise
                reason = f"invalid package will be recomputed: {exc}"
            else:
                reason = "verified fingerprint and upstream lineage match"
        if ref is not None:
            decisions.append(
                ResumeStageDecision(
                    layer=layer,
                    expected_fingerprint=expected,
                    disposition=StageDisposition.REUSED,
                    state=None if dry_run else RunState.SUCCEEDED,
                    manifest_ref=ref,
                    reason=reason,
                )
            )
            upstream = ref
            chain[layer] = ref
            continue

        reusable_prefix = False
        if next_layer is None:
            next_layer = layer
        decisions.append(
            ResumeStageDecision(
                layer=layer,
                expected_fingerprint=expected,
                disposition=StageDisposition.EXECUTED,
                reason=reason
                if reason != "verified fingerprint and upstream lineage match"
                else "no reusable package",
            )
        )

    return ResumePlan(
        dry_run=dry_run,
        decisions=decisions,
        next_layer=next_layer,
        complete=bool(decisions)
        and all(item.disposition is StageDisposition.REUSED for item in decisions),
    )


def execute_resume_plan(
    *,
    store: LocalArtifactStore,
    plan: ResumePlan,
    execute_stage: Callable[[Layer, Mapping[Layer, ManifestRef]], ManifestRef],
    semantic_validator: Callable[
        [LocalArtifactStore, ManifestRef, Mapping[Layer, ManifestRef]], None
    ] = _load_semantic_package,
) -> ResumePlan:
    """Execute exactly the non-reused suffix and verify every new boundary."""
    if plan.dry_run:
        return plan
    completed: list[ResumeStageDecision] = []
    upstream: ManifestRef | None = None
    chain: dict[Layer, ManifestRef] = {}
    for decision in plan.decisions:
        if decision.disposition is StageDisposition.REUSED:
            if decision.manifest_ref is None:
                raise ValueError("reused stage is missing its manifest reference")
            upstream = decision.manifest_ref
            semantic_validator(store, upstream, chain)
            chain[decision.layer] = upstream
            completed.append(decision)
            continue
        try:
            ref = execute_stage(decision.layer, dict(chain))
            report = store.verify(ref)
            if not report.valid:
                raise ValueError(
                    f"executed {decision.layer.value} package failed verification: "
                    f"{'; '.join(report.errors)}"
                )
            manifest = store.load_manifest(ref)
            if manifest.layer_fingerprint != decision.expected_fingerprint:
                raise ValueError(f"executed {decision.layer.value} fingerprint does not match plan")
            if upstream is not None and upstream not in manifest.upstream_manifests:
                raise ValueError(
                    f"executed {decision.layer.value} upstream lineage does not match plan"
                )
            semantic_validator(store, ref, chain)
        except Exception as exc:
            failed = decision.model_copy(
                update={
                    "state": RunState.FAILED,
                    "reason": f"execution failed: {type(exc).__name__}: {exc}",
                }
            )
            partial = plan.model_copy(
                update={
                    "decisions": [*completed, failed],
                    "next_layer": decision.layer,
                    "complete": False,
                }
            )
            raise ResumeExecutionError(partial, decision.layer, exc) from exc
        upstream = ref
        chain[decision.layer] = ref
        completed.append(
            decision.model_copy(
                update={
                    "manifest_ref": ref,
                    "state": RunState.SUCCEEDED,
                    "reason": "executed and verified against planned fingerprint and lineage",
                }
            )
        )
    return plan.model_copy(update={"decisions": completed, "next_layer": None, "complete": True})
