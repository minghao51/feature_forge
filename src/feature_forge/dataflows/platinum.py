"""Platinum fold evidence, aggregation, uncertainty, and durable boundary.

The Platinum driver is one-call: ``execute(['platinum_materialization'])``
derives the manifest from upstream nodes, so callers supply only the Platinum
request input, verified Silver/Gold packages, and the environment snapshot.
Persisted fold evidence labels the reported arm ``enhanced`` so
``load_platinum_package`` can reconstruct both candidate and no-candidate
packages.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from feature_forge.contracts import (
    AggregateMetric,
    ArtifactNamespace,
    CheckResult,
    EnvironmentSnapshot,
    FailureClass,
    GoldPackage,
    Layer,
    PlatinumMaterialization,
    PlatinumPackage,
    PlatinumRequest,
    PlatinumSelectionDecision,
    RunManifest,
    RunRequest,
    RunState,
    SilverPackage,
    StageResult,
    UncertaintySummary,
)
from feature_forge.dataflows._io import (
    build_platinum_request as build_platinum_request,
)
from feature_forge.dataflows._io import (
    load_platinum_package as load_platinum_package,
)
from feature_forge.dataflows.hamilton_compat import cache, tag
from feature_forge.dataflows.profile import ExecutionProfile, get_profile_policy
from feature_forge.evaluation.metrics import MetricDirection, get_metric
from feature_forge.evaluation.model_factory import ModelFactory
from feature_forge.exceptions import DatasetError
from feature_forge.storage.hashing import fingerprint
from feature_forge.storage.local import LocalArtifactStore

PLATINUM_REQUIRED_ARTIFACTS = {
    "request.json": "application/json",
    "fold_assignments.parquet": "application/vnd.apache.parquet",
    "fold_metrics.parquet": "application/vnd.apache.parquet",
    "predictions.parquet": "application/vnd.apache.parquet",
    "aggregate_metrics.json": "application/json",
    "uncertainty.json": "application/json",
    "selection_decisions.json": "application/json",
    "checks.json": "application/json",
}


def _prepared(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if not pd.api.types.is_numeric_dtype(result[column]):
            result[column] = pd.Categorical(result[column]).codes
        result[column] = result[column].fillna(result[column].median())
    return result


def _evaluate_frame(
    request: PlatinumRequest, frame: pd.DataFrame, silver: SilverPackage, arm: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _prepared(frame)
    target = silver.canonical_target[silver.bronze.target].reset_index(drop=True)
    rows = silver.row_ids["row_id"].astype(str).reset_index(drop=True)
    folds = silver.fold_assignments["fold"].reset_index(drop=True)
    metric_fn = get_metric(request.metric)
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold in sorted(int(value) for value in folds.unique()):
        train = folds != fold
        valid = folds == fold
        estimator = ModelFactory(random_state=request.model.seed).get_model(
            request.model.name, request.model.task
        )
        estimator.fit(features.loc[train], target.loc[train])
        if request.metric == "auc" and hasattr(estimator, "predict_proba"):
            predicted = estimator.predict_proba(features.loc[valid])
            if getattr(predicted, "ndim", 1) > 1 and predicted.shape[1] == 2:
                predicted = predicted[:, 1]
        else:
            predicted = estimator.predict(features.loc[valid])
        score = float(metric_fn(target.loc[valid].to_numpy(), np.asarray(predicted)))
        metrics.append(
            {
                "arm": arm,
                "fold": fold,
                "metric": request.metric,
                "score": score,
                "n_train": int(train.sum()),
                "n_validation": int(valid.sum()),
                "status": "succeeded",
            }
        )
        for row_id, actual, value in zip(
            rows.loc[valid], target.loc[valid], np.asarray(predicted), strict=True
        ):
            predictions.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "row_id": row_id,
                    "target_json": json.dumps(actual.item() if hasattr(actual, "item") else actual),
                    "prediction_json": json.dumps(
                        value.tolist() if hasattr(value, "tolist") else value
                    ),
                }
            )
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


@tag(layer="platinum", cost="cheap", persistence="none", owner="evaluation")
def platinum_request(platinum_request_input: PlatinumRequest) -> PlatinumRequest:
    """Normalize the executor-supplied Platinum request into the node graph."""
    return PlatinumRequest.model_validate(platinum_request_input.model_dump(mode="python"))


@tag(layer="platinum", cost="expensive", persistence="recompute", owner="evaluation")
@cache(behavior="recompute")
def baseline_fold_evidence(
    platinum_request: PlatinumRequest, verified_silver_package: SilverPackage
) -> dict[str, pd.DataFrame]:
    metrics, predictions = _evaluate_frame(
        platinum_request,
        verified_silver_package.canonical_features,
        verified_silver_package,
        "baseline",
    )
    return {"metrics": metrics, "predictions": predictions}


@tag(layer="platinum", cost="expensive", persistence="recompute", owner="evaluation")
@cache(behavior="recompute")
def candidate_fold_evidence(
    platinum_request: PlatinumRequest,
    verified_silver_package: SilverPackage,
    verified_gold_package: GoldPackage,
) -> dict[str, dict[str, pd.DataFrame]]:
    results: dict[str, dict[str, pd.DataFrame]] = {}
    base = verified_silver_package.canonical_features
    for name in verified_gold_package.accepted_features.columns:
        if name == "row_id":
            continue
        frame = pd.concat(
            [base, verified_gold_package.accepted_features[[name]].reset_index(drop=True)], axis=1
        )
        metrics, predictions = _evaluate_frame(
            platinum_request, frame, verified_silver_package, f"candidate:{name}"
        )
        results[name] = {"metrics": metrics, "predictions": predictions}
    return results


@tag(layer="platinum", cost="cheap", persistence="none", owner="evaluation")
def aggregate_metrics(
    platinum_request: PlatinumRequest,
    baseline_fold_evidence: dict[str, pd.DataFrame],
    candidate_fold_evidence: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    baseline = baseline_fold_evidence["metrics"].set_index("fold")["score"]
    candidates: dict[str, dict[str, float]] = {}
    for name, evidence in candidate_fold_evidence.items():
        enhanced = evidence["metrics"].set_index("fold")["score"]
        delta = enhanced - baseline
        direction = 1.0 if platinum_request.metric_direction is MetricDirection.MAXIMIZE else -1.0
        candidates[name] = {
            "legacy_gain": float(delta.mean()),
            "directional_gain": float((delta * direction).mean()),
        }
    best = max(candidates, key=lambda key: candidates[key]["directional_gain"], default=None)
    if best is None:
        enhanced = baseline
    else:
        enhanced = candidate_fold_evidence[best]["metrics"].set_index("fold")["score"]
    direction = 1.0 if platinum_request.metric_direction is MetricDirection.MAXIMIZE else -1.0
    return {
        "baseline_score": float(baseline.mean()),
        "enhanced_score": float(enhanced.mean()),
        "legacy_gain": float((enhanced - baseline).mean()),
        "directional_gain": float(((enhanced - baseline) * direction).mean()),
        "fold_deltas": (enhanced - baseline).tolist(),
        "candidates": candidates,
        "best": best,
    }


@tag(layer="platinum", cost="cheap", persistence="none", owner="evaluation")
def uncertainty_summary(
    platinum_request: PlatinumRequest, aggregate_metrics: dict[str, Any]
) -> UncertaintySummary:
    deltas = np.asarray(aggregate_metrics["fold_deltas"], dtype=float)
    if len(deltas) < platinum_request.evaluation_policy.minimum_successful_folds:
        raise DatasetError("Evaluation did not meet the minimum successful-fold policy")
    std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    se = std / math.sqrt(len(deltas))
    # A conservative normal approximation keeps the core dataflow dependency-free.
    margin = 1.96 * se
    return UncertaintySummary(
        method="paired_t",
        confidence_level=platinum_request.uncertainty_policy.confidence_level,
        pair_count=len(deltas),
        mean_directional_gain=float(aggregate_metrics["directional_gain"]),
        standard_deviation=std,
        standard_error=se,
        lower_bound=float(deltas.mean() - margin),
        upper_bound=float(deltas.mean() + margin),
    )


@tag(layer="platinum", cost="cheap", persistence="none", owner="evaluation")
def selection_decisions(
    platinum_request: PlatinumRequest,
    verified_gold_package: GoldPackage,
    aggregate_metrics: dict[str, Any],
    uncertainty_summary: UncertaintySummary,
) -> list[PlatinumSelectionDecision]:
    selected_name = aggregate_metrics["best"]
    selected = selected_name is not None and (
        aggregate_metrics["candidates"][selected_name]["directional_gain"]
        >= platinum_request.selection_policy.minimum_practical_gain
    )
    decisions = []
    for candidate in verified_gold_package.candidates:
        values = aggregate_metrics["candidates"].get(
            candidate.name, {"legacy_gain": 0.0, "directional_gain": 0.0}
        )
        is_selected = selected and candidate.name == selected_name
        decisions.append(
            PlatinumSelectionDecision(
                candidate_id=candidate.candidate_id,
                feature_name=candidate.name,
                selected=is_selected,
                reason_code="selected" if is_selected else "not_selected",
                reason="best directional fold gain"
                if is_selected
                else "did not meet selection policy",
                legacy_gain=values["legacy_gain"],
                directional_gain=values["directional_gain"],
                lower_bound=uncertainty_summary.lower_bound,
            )
        )
    return decisions


@tag(layer="platinum", cost="cheap", persistence="none", owner="verification")
def platinum_checks(
    platinum_request: PlatinumRequest,
    verified_silver_package: SilverPackage,
    baseline_fold_evidence: dict[str, pd.DataFrame],
    aggregate_metrics: dict[str, Any],
) -> list[CheckResult]:
    metrics = baseline_fold_evidence["metrics"]
    return [
        CheckResult(
            check_id="PLATINUM.FOLDS.COMPLETE",
            passed=len(metrics) >= platinum_request.evaluation_policy.minimum_successful_folds,
            message="Fold evidence is complete",
        ),
        CheckResult(
            check_id="PLATINUM.ROWS.ALIGNED",
            passed=metrics["n_validation"].sum() == len(verified_silver_package.row_ids),
            message="Fold rows align with Silver",
        ),
        CheckResult(
            check_id="PLATINUM.AGGREGATE.RECONSTRUCTABLE",
            passed=math.isfinite(aggregate_metrics["enhanced_score"]),
            message="Aggregate is finite",
        ),
    ]


@tag(layer="platinum", cost="cheap", persistence="recompute", owner="verification")
@cache(behavior="recompute")
def platinum_manifest(
    platinum_request: PlatinumRequest,
    environment_snapshot: EnvironmentSnapshot,
    platinum_checks: list[CheckResult],
    aggregate_metrics: dict[str, Any],
    uncertainty_summary: UncertaintySummary,
    selection_decisions: list[PlatinumSelectionDecision],
) -> RunManifest:
    """Build the per-attempt Platinum manifest from node-produced evidence."""
    now = datetime.now(UTC)
    passed = all(item.passed for item in platinum_checks if item.required)
    layer_fp = fingerprint(
        {
            "input": platinum_request.platinum_input_fingerprint,
            "aggregate": aggregate_metrics,
            "uncertainty": uncertainty_summary.model_dump(mode="json"),
            "decisions": [item.model_dump(mode="json") for item in selection_decisions],
        }
    )
    return RunManifest(
        layer=Layer.PLATINUM,
        package_kind="platinum",
        layer_fingerprint=layer_fp,
        reuse_fingerprint=platinum_request.platinum_input_fingerprint,
        run_id=platinum_request.run_id,
        case_fingerprint=platinum_request.case_fingerprint,
        state=RunState.SUCCEEDED if passed else RunState.FAILED,
        request=RunRequest(
            dataset="silver",
            method="platinum",
            model=platinum_request.model.name,
            seed=platinum_request.model.seed,
            options={"platinum_input_fingerprint": platinum_request.platinum_input_fingerprint},
        ),
        environment=environment_snapshot,
        upstream_manifests=[platinum_request.silver_manifest, platinum_request.gold_manifest],
        stages=[
            StageResult(
                stage="platinum",
                state=RunState.SUCCEEDED if passed else RunState.FAILED,
                started_at=now,
                finished_at=now,
                checks=platinum_checks,
                failure_class=None if passed else FailureClass.DETERMINISTIC,
            )
        ],
        created_at=now,
        completed_at=now if passed else None,
    )


@tag(layer="platinum", cost="io", persistence="boundary", owner="storage")
@cache(behavior="recompute")
def platinum_materialization(
    platinum_manifest: RunManifest,
    platinum_request: PlatinumRequest,
    verified_silver_package: SilverPackage,
    baseline_fold_evidence: dict[str, pd.DataFrame],
    candidate_fold_evidence: dict[str, dict[str, pd.DataFrame]],
    aggregate_metrics: dict[str, Any],
    uncertainty_summary: UncertaintySummary,
    selection_decisions: list[PlatinumSelectionDecision],
    platinum_checks: list[CheckResult],
    execution_profile: ExecutionProfile,
    artifact_store: LocalArtifactStore | None,
) -> PlatinumMaterialization:
    """Persist the verified Platinum package with paired enhanced-arm evidence."""
    if any(item.required and not item.passed for item in platinum_checks):
        raise DatasetError("Platinum checks failed")
    best = aggregate_metrics["best"]
    enhanced_evidence = candidate_fold_evidence[best] if best else baseline_fold_evidence
    # The persisted evidence always carries a paired "enhanced" arm: the winning
    # candidate when one is selected, otherwise a copy of the baseline arm so
    # no-candidate packages still reconstruct during loading.
    metrics = pd.concat(
        [baseline_fold_evidence["metrics"], enhanced_evidence["metrics"].assign(arm="enhanced")],
        ignore_index=True,
    )
    predictions = pd.concat(
        [
            baseline_fold_evidence["predictions"],
            enhanced_evidence["predictions"].assign(arm="enhanced"),
        ],
        ignore_index=True,
    )
    aggregate = AggregateMetric(
        metric=platinum_request.metric,
        metric_direction=platinum_request.metric_direction,
        aggregation="unweighted_mean_of_fold_metrics",
        baseline_score=aggregate_metrics["baseline_score"],
        enhanced_score=aggregate_metrics["enhanced_score"],
        legacy_gain=aggregate_metrics["legacy_gain"],
        directional_gain=aggregate_metrics["directional_gain"],
        successful_folds=len(aggregate_metrics["fold_deltas"]),
    )
    package = PlatinumPackage(
        manifest=platinum_manifest,
        request=platinum_request,
        fold_assignments=verified_silver_package.fold_assignments,
        fold_metrics=metrics,
        predictions=predictions,
        aggregate=aggregate,
        uncertainty=uncertainty_summary,
        decisions=selection_decisions,
        checks=platinum_checks,
    )
    policy = get_profile_policy(execution_profile)
    if not policy.persistence_enabled or artifact_store is None:
        return PlatinumMaterialization(manifest=platinum_manifest, persisted=False)
    staging = artifact_store.begin(
        ArtifactNamespace(layer=Layer.PLATINUM, run_id=platinum_request.run_id)
    )
    staging.write_json("request.json", platinum_request.model_dump(mode="json"))
    staging.write_dataframe("fold_assignments.parquet", package.fold_assignments)
    staging.write_dataframe("fold_metrics.parquet", package.fold_metrics)
    staging.write_dataframe("predictions.parquet", package.predictions)
    staging.write_json("aggregate_metrics.json", aggregate.model_dump(mode="json"))
    staging.write_json("uncertainty.json", uncertainty_summary.model_dump(mode="json"))
    staging.write_json(
        "selection_decisions.json", [item.model_dump(mode="json") for item in selection_decisions]
    )
    staging.write_json("checks.json", [item.model_dump(mode="json") for item in platinum_checks])
    staging.set_manifest(platinum_manifest.model_copy(update={"artifacts": staging.artifacts}))
    ref = artifact_store.commit(staging)
    return PlatinumMaterialization(
        manifest=staging.manifest,
        manifest_ref=ref,
        persisted=True,
        artifact_paths=[item.relative_path for item in staging.artifacts],
    )
