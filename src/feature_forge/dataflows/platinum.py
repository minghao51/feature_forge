"""Platinum fold evidence, aggregation, uncertainty, and durable boundary.

The Platinum driver is one-call: ``execute(['platinum_materialization'])``
derives the manifest from upstream nodes, so callers supply only the Platinum
request input, verified Silver/Gold packages, and the environment snapshot.
Persisted fold evidence labels the reported arm ``enhanced`` so
``load_platinum_package`` can reconstruct both candidate and no-candidate
packages. The v2 evidence set (ADR 0018 decision 8) additionally persists the
discovery-scope trial arms (``discovery_fold_metrics.parquet``), the greedy
selection transcript (``selection_steps.json``), the per-arm preprocessing
identity (``preprocessing.json``), and the ``evidence.json`` marker that binds
the set and carries the evidence-schema bump; the loader reconstructs and
verifies all of it offline.

Evaluation evidence is partition-aware (ADR 0018): candidate and selection
evidence use Silver discovery folds only, while the reported baseline and
enhanced arms are evaluated on evaluation folds with fold-local
preprocessing; the explicit ``compatibility`` protocol replays all rows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from feature_forge.contracts import (
    PLATINUM_IDENTITY_SCHEMA_VERSION,
    AggregateMetric,
    ArtifactNamespace,
    CheckResult,
    EnvironmentSnapshot,
    EvaluationProtocol,
    FailureClass,
    GoldPackage,
    Layer,
    PlatinumEvidenceIndex,
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
from feature_forge.evaluation.preprocessing import FoldPreprocessor
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
    "discovery_fold_metrics.parquet": "application/vnd.apache.parquet",
    "selection_steps.json": "application/json",
    "preprocessing.json": "application/json",
    "evidence.json": "application/json",
}


def _partition_scope(
    silver: SilverPackage,
    protocol: EvaluationProtocol,
    partition: Literal["discovery", "evaluation"],
) -> np.ndarray:
    """Positional boolean row mask for ``partition`` in Silver row order.
    The ``compatibility`` protocol covers every row (legacy all-row
    evaluation, ADR 0018). Under ``holdout`` the scope comes solely from the
    ``partition`` column: verified Silver packages generated under the
    holdout protocol always carry labels — the Silver checks reject a
    missing ``partition`` column — so a missing column means a pre-ADR-0018
    package and fails closed instead of silently evaluating discovery and
    reported evidence on all rows (mirroring the executor's fit-time guard).
    """
    if protocol == "compatibility":
        return np.ones(len(silver.row_ids), dtype=bool)
    if "partition" not in silver.fold_assignments.columns:
        raise DatasetError(
            "Silver fold_assignments lacks a 'partition' column; regenerate the "
            "Silver package under a partition-aware evaluation protocol (ADR 0018) "
            "before evaluating platinum evidence"
        )
    return (silver.fold_assignments["partition"] == partition).to_numpy()


def _evaluate_frame(
    request: PlatinumRequest,
    frame: pd.DataFrame,
    silver: SilverPackage,
    arm: str,
    *,
    partition: Literal["discovery", "evaluation"],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate one arm over ``partition`` rows with fold-local preprocessing.

    Rows are scoped to the Silver partition before any fold splitting; per
    fold the ``FoldPreprocessor`` is fitted on that fold's training rows only
    and its specification is retained as evaluation provenance (ADR 0018
    decision 4). Scoped frames keep the original Silver row index.
    """
    scope = _partition_scope(silver, request.evaluation_policy.evaluation_protocol, partition)
    features = frame.loc[scope]
    target = silver.canonical_target[silver.bronze.target].loc[scope]
    rows = silver.row_ids["row_id"].astype(str).loc[scope]
    folds = silver.fold_assignments["fold"].loc[scope]
    metric_fn = get_metric(request.metric)
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    preprocessing: dict[str, Any] = {}
    for fold in sorted(int(value) for value in folds.unique()):
        train = (folds != fold).to_numpy()
        valid = (folds == fold).to_numpy()
        preprocessor = FoldPreprocessor.fit(features.loc[train])
        train_features = preprocessor.transform(features.loc[train])
        valid_features = preprocessor.transform(features.loc[valid])
        estimator = ModelFactory(random_state=request.model.seed).get_model(
            request.model.name, request.model.task
        )
        estimator.fit(train_features, target.loc[train])
        if request.metric == "auc" and hasattr(estimator, "predict_proba"):
            predicted = estimator.predict_proba(valid_features)
            if getattr(predicted, "ndim", 1) > 1 and predicted.shape[1] == 2:
                predicted = predicted[:, 1]
        else:
            predicted = estimator.predict(valid_features)
        score = float(metric_fn(target.loc[valid].to_numpy(), np.asarray(predicted)))
        preprocessing[str(fold)] = preprocessor.specification()
        metrics.append(
            {
                "arm": arm,
                "fold": fold,
                "metric": request.metric,
                "score": score,
                "n_train": int(train.sum()),
                "n_validation": int(valid.sum()),
                "status": "succeeded",
                "partition": partition,
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
    return pd.DataFrame(metrics), pd.DataFrame(predictions), preprocessing


@tag(layer="platinum", cost="cheap", persistence="none", owner="evaluation")
def platinum_request(platinum_request_input: PlatinumRequest) -> PlatinumRequest:
    """Normalize the executor-supplied Platinum request into the node graph."""
    return PlatinumRequest.model_validate(platinum_request_input.model_dump(mode="python"))


@tag(layer="platinum", cost="expensive", persistence="recompute", owner="evaluation")
@cache(behavior="recompute")
def baseline_fold_evidence(
    platinum_request: PlatinumRequest, verified_silver_package: SilverPackage
) -> dict[str, Any]:
    """Evaluate the baseline arm on evaluation folds (ADR 0018 decision 6)."""
    metrics, predictions, preprocessing = _evaluate_frame(
        platinum_request,
        verified_silver_package.canonical_features,
        verified_silver_package,
        "baseline",
        partition="evaluation",
    )
    return {"metrics": metrics, "predictions": predictions, "preprocessing": preprocessing}


@dataclass(frozen=True)
class GreedySelectionOutcome:
    """Frozen discovery-fold greedy selection result (ADR 0018 §5).

    Attributes:
        selected: Feature names in frozen selection order.
        steps: Per-greedy-step records (gains, lower bounds, ranking).
        baseline_fold_scores: Discovery-fold baseline scores the first
            greedy step is compared against.
        step_arm_metrics: Fold-metrics frames for every greedy-labeled arm
            (``greedy:...``) evaluated across the steps, in step order, so
            the discovery trial evidence can be persisted verbatim. Step-1
            arms reuse the per-candidate evidence and are not duplicated
            here; the discovery baseline arm is carried separately by
            ``CandidateFoldEvidence.discovery_arms``.
    """

    selected: tuple[str, ...]
    steps: tuple[dict[str, Any], ...]
    baseline_fold_scores: dict[int, float]
    step_arm_metrics: tuple[pd.DataFrame, ...]


class CandidateFoldEvidence(dict[str, dict[str, pd.DataFrame]]):
    """Per-candidate discovery-fold evidence plus frozen selection artifacts.

    Mapping keys are candidate feature names only, so selection consumers
    iterating the mapping can never read frozen-selection artifacts; the
    outcome and the post-freeze evaluation arm ride as attributes.
    """

    def __init__(
        self,
        evidence: dict[str, dict[str, pd.DataFrame]],
        *,
        selection: GreedySelectionOutcome,
        selected_arm: dict[str, Any] | None,
        discovery_arms: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(evidence)
        self.selection = selection
        self.selected_arm = selected_arm
        self.discovery_arms = discovery_arms


def _t_critical(confidence_level: float, degrees_of_freedom: int) -> float:
    """Two-sided paired Student-t critical value (ADR 0018 decision 7).

    Derived from the configured confidence level and ``pair_count - 1``
    degrees of freedom; replaces the pre-PR-4 1.96 normal-approximation
    margin so paired-fold uncertainty reflects small sample sizes.
    """
    return float(student_t.ppf((1 + confidence_level) / 2, degrees_of_freedom))


def _directional_lower_bound(deltas: np.ndarray, confidence_level: float) -> float:
    """Lower bound of the directional paired Student-t interval (ADR 0018 decision 7).

    The margin is the Student-t critical value for ``confidence_level`` with
    ``len(deltas) - 1`` degrees of freedom applied to the standard error of
    the mean. Fewer than two paired folds cannot establish a positive bound
    (the critical value is undefined at zero degrees of freedom), so the
    bound fails closed to ``-inf``; the empty case has no evidence at all.
    """
    if len(deltas) < 2:
        return float("-inf")
    se = float(deltas.std(ddof=1)) / math.sqrt(len(deltas))
    return float(deltas.mean() - _t_critical(confidence_level, len(deltas) - 1) * se)


def _arm_fold_scores(metrics: pd.DataFrame) -> dict[int, float]:
    """Fold → score mapping extracted from one arm's fold-metrics frame."""
    return {
        int(fold): float(score)
        for fold, score in zip(metrics["fold"], metrics["score"], strict=True)
    }


def _greedy_forward_selection(
    platinum_request: PlatinumRequest,
    silver: SilverPackage,
    gold: GoldPackage,
    candidate_evidence: dict[str, dict[str, pd.DataFrame]],
    baseline_fold_scores: dict[int, float],
) -> GreedySelectionOutcome:
    """Greedy forward selection on discovery folds (ADR 0018 §5).

    Each step scores every remaining candidate appended to the already
    selected set on discovery folds, ranks the candidates by mean directional
    gain with a stable feature-name tie-break, and elects the first ranked
    candidate meeting ``minimum_practical_gain`` and — when configured — a
    strictly positive directional lower confidence bound. Selection stops at
    ``max_selected_features`` or when no remaining candidate qualifies.
    """
    policy = platinum_request.selection_policy
    direction = 1.0 if platinum_request.metric_direction is MetricDirection.MAXIMIZE else -1.0
    selected: list[str] = []
    current_scores = dict(baseline_fold_scores)
    remaining = sorted(candidate_evidence)
    steps: list[dict[str, Any]] = []
    step_arm_metrics: list[pd.DataFrame] = []
    while True:
        if (
            policy.max_selected_features is not None
            and len(selected) >= policy.max_selected_features
        ):
            break
        if not remaining:
            break
        selected_before = list(selected)
        ranked: list[tuple[dict[str, Any], dict[int, float]]] = []
        for name in remaining:
            if selected:
                frame = pd.concat(
                    [
                        silver.canonical_features,
                        *(
                            gold.accepted_features[[feature]].reset_index(drop=True)
                            for feature in [*selected, name]
                        ),
                    ],
                    axis=1,
                )
                metrics, _, _ = _evaluate_frame(
                    platinum_request,
                    frame,
                    silver,
                    arm=f"greedy:{'+'.join([*selected, name])}",
                    partition="discovery",
                )
                step_arm_metrics.append(metrics)
                candidate_scores = _arm_fold_scores(metrics)
            else:
                # The base+single-candidate arm is exactly the per-candidate
                # discovery evidence; reuse it instead of re-evaluating.
                candidate_scores = _arm_fold_scores(candidate_evidence[name]["metrics"])
            deltas = np.asarray(
                [
                    direction * (candidate_scores[fold] - current_scores[fold])
                    for fold in sorted(current_scores)
                ],
                dtype=float,
            )
            entry = {
                "feature": name,
                "directional_gain": float(deltas.mean()),
                "lower_bound": _directional_lower_bound(
                    deltas, platinum_request.uncertainty_policy.confidence_level
                ),
            }
            ranked.append((entry, candidate_scores))
        ranked.sort(key=lambda item: (-item[0]["directional_gain"], item[0]["feature"]))
        chosen: dict[str, Any] | None = None
        chosen_scores: dict[int, float] | None = None
        for candidate_entry, candidate_scores in ranked:
            if candidate_entry["directional_gain"] < policy.minimum_practical_gain:
                continue
            if policy.require_positive_lower_bound and not candidate_entry["lower_bound"] > 0.0:
                continue
            chosen, chosen_scores = candidate_entry, candidate_scores
            break
        if chosen is None or chosen_scores is None:
            steps.append(
                {
                    "step": len(steps) + 1,
                    "selected_before": selected_before,
                    "ranking": [item for item, _ in ranked],
                    "chosen": None,
                    "fold_scores": {},
                }
            )
            break
        selected.append(chosen["feature"])
        remaining.remove(chosen["feature"])
        current_scores = chosen_scores
        steps.append(
            {
                "step": len(steps) + 1,
                "selected_before": selected_before,
                "ranking": [item for item, _ in ranked],
                "chosen": chosen["feature"],
                "fold_scores": {str(fold): score for fold, score in chosen_scores.items()},
            }
        )
    return GreedySelectionOutcome(
        selected=tuple(selected),
        steps=tuple(steps),
        baseline_fold_scores=dict(baseline_fold_scores),
        step_arm_metrics=tuple(step_arm_metrics),
    )


@tag(layer="platinum", cost="expensive", persistence="recompute", owner="evaluation")
@cache(behavior="recompute")
def candidate_fold_evidence(
    platinum_request: PlatinumRequest,
    verified_silver_package: SilverPackage,
    verified_gold_package: GoldPackage,
) -> CandidateFoldEvidence:
    """Evaluate accepted candidates on discovery folds, then freeze selection.

    Per-candidate arms use discovery-partition rows only (ADR 0018 decision
    3). The greedy outcome freezes the selected set before the combined set
    is scored once on evaluation folds as ``selected_arm``; an empty
    selection yields no combined arm. The discovery baseline arm plus every
    greedy-step arm ride as ``discovery_arms`` for v2 persistence.
    """
    results: dict[str, dict[str, pd.DataFrame]] = {}
    base = verified_silver_package.canonical_features
    accepted = verified_gold_package.accepted_features
    for name in accepted.columns:
        if name == "row_id":
            continue
        frame = pd.concat([base, accepted[[name]].reset_index(drop=True)], axis=1)
        metrics, predictions, _ = _evaluate_frame(
            platinum_request,
            frame,
            verified_silver_package,
            f"candidate:{name}",
            partition="discovery",
        )
        results[name] = {"metrics": metrics, "predictions": predictions}
    baseline_metrics, _, _ = _evaluate_frame(
        platinum_request,
        verified_silver_package.canonical_features,
        verified_silver_package,
        "baseline",
        partition="discovery",
    )
    baseline_fold_scores = {
        int(fold): float(score)
        for fold, score in zip(baseline_metrics["fold"], baseline_metrics["score"], strict=True)
    }
    selection = _greedy_forward_selection(
        platinum_request,
        verified_silver_package,
        verified_gold_package,
        results,
        baseline_fold_scores,
    )
    selected_arm: dict[str, Any] | None = None
    if selection.selected:
        frame = pd.concat(
            [base, *(accepted[[name]].reset_index(drop=True) for name in selection.selected)],
            axis=1,
        )
        metrics, predictions, preprocessing = _evaluate_frame(
            platinum_request, frame, verified_silver_package, "enhanced", partition="evaluation"
        )
        selected_arm = {
            "metrics": metrics,
            "predictions": predictions,
            "preprocessing": preprocessing,
        }
    discovery_arms = pd.concat([baseline_metrics, *selection.step_arm_metrics], ignore_index=True)
    return CandidateFoldEvidence(
        results,
        selection=selection,
        selected_arm=selected_arm,
        discovery_arms=discovery_arms,
    )


@tag(layer="platinum", cost="cheap", persistence="none", owner="evaluation")
def aggregate_metrics(
    platinum_request: PlatinumRequest,
    baseline_fold_evidence: dict[str, Any],
    candidate_fold_evidence: CandidateFoldEvidence | dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    """Aggregate evaluation-scope arms plus discovery-fold candidate gains.

    Candidate gains summarize discovery-fold evidence against the discovery
    baseline; the reported baseline/enhanced arms come from evaluation-fold
    scores (``selected_arm`` when selection froze a non-empty set, otherwise
    a baseline mirror with zero gain).

    The plain-``dict`` fallback (no frozen selection attribute) pairs
    candidate scores with the baseline evidence by bare fold id, so it is
    only sound for single-scope (all-row / unpartitioned) evidence — the
    real pipeline always passes ``CandidateFoldEvidence``.
    """
    direction = 1.0 if platinum_request.metric_direction is MetricDirection.MAXIMIZE else -1.0
    baseline = baseline_fold_evidence["metrics"].set_index("fold")["score"]
    selection: GreedySelectionOutcome | None = getattr(candidate_fold_evidence, "selection", None)
    selected_arm: dict[str, Any] | None = getattr(candidate_fold_evidence, "selected_arm", None)
    if selection is not None:
        discovery_baseline = dict(selection.baseline_fold_scores)
    else:
        discovery_baseline = {int(fold): float(score) for fold, score in baseline.items()}
    candidates: dict[str, dict[str, float]] = {}
    for name, evidence in candidate_fold_evidence.items():
        scores = evidence["metrics"].set_index("fold")["score"]
        aligned = [int(fold) for fold in scores.index if int(fold) in discovery_baseline]
        legacy_gain = (
            float(np.mean([float(scores[fold]) - discovery_baseline[fold] for fold in aligned]))
            if aligned
            else 0.0
        )
        candidates[name] = {
            "legacy_gain": float(legacy_gain),
            "directional_gain": float(legacy_gain * direction),
        }
    best: str | None = None
    if candidates:
        best = max(sorted(candidates), key=lambda key: candidates[key]["directional_gain"])
    enhanced = (
        selected_arm["metrics"].set_index("fold")["score"] if selected_arm is not None else baseline
    )
    fold_deltas = enhanced - baseline
    baseline_preprocessing = baseline_fold_evidence.get("preprocessing")
    enhanced_preprocessing = selected_arm.get("preprocessing") if selected_arm is not None else None
    result: dict[str, Any] = {
        "baseline_score": float(baseline.mean()),
        "enhanced_score": float(enhanced.mean()),
        "legacy_gain": float(fold_deltas.mean()),
        "directional_gain": float((fold_deltas * direction).mean()),
        "fold_deltas": fold_deltas.tolist(),
        "candidates": candidates,
        "best": best,
    }
    if selection is not None:
        result["selected"] = list(selection.selected)
    result["preprocessing"] = {
        "baseline": baseline_preprocessing,
        "enhanced": enhanced_preprocessing
        if enhanced_preprocessing is not None
        else baseline_preprocessing,
    }
    return result


@tag(layer="platinum", cost="cheap", persistence="none", owner="evaluation")
def uncertainty_summary(
    platinum_request: PlatinumRequest, aggregate_metrics: dict[str, Any]
) -> UncertaintySummary:
    """Paired directional Student-t interval over fold deltas (ADR 0018 decision 7).

    Raw fold deltas (enhanced - baseline) are sign-adjusted by the metric
    direction so every reported statistic describes directional gain; the
    margin applies the Student-t critical value for the configured
    confidence level with ``pair_count - 1`` degrees of freedom.
    """
    raw_deltas = np.asarray(aggregate_metrics["fold_deltas"], dtype=float)
    if len(raw_deltas) < platinum_request.evaluation_policy.minimum_successful_folds:
        raise DatasetError("Evaluation did not meet the minimum successful-fold policy")
    direction = 1.0 if platinum_request.metric_direction is MetricDirection.MAXIMIZE else -1.0
    deltas = raw_deltas * direction
    std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    se = std / math.sqrt(len(deltas))
    margin = _t_critical(platinum_request.uncertainty_policy.confidence_level, len(deltas) - 1) * se
    mean = float(deltas.mean())
    return UncertaintySummary(
        method="paired_t",
        confidence_level=platinum_request.uncertainty_policy.confidence_level,
        pair_count=len(deltas),
        mean_directional_gain=mean,
        standard_deviation=std,
        standard_error=se,
        lower_bound=float(mean - margin),
        upper_bound=float(mean + margin),
    )


@tag(layer="platinum", cost="cheap", persistence="none", owner="evaluation")
def selection_decisions(
    platinum_request: PlatinumRequest,
    verified_gold_package: GoldPackage,
    aggregate_metrics: dict[str, Any],
    uncertainty_summary: UncertaintySummary,
) -> list[PlatinumSelectionDecision]:
    """Gate candidate election over the aggregate evidence (ADR 0018 5-6).

    Real-pipeline aggregates carry the ``selected`` list frozen by greedy
    selection on discovery folds, which already enforced every selection
    policy field; exactly that set is elected. Aggregates without a frozen
    selection (hand-built or reconstructed) fall back to electing the
    deterministic best candidate, gated by the selection policy.
    """
    policy = platinum_request.selection_policy
    frozen = aggregate_metrics.get("selected")
    elected: set[str]
    if frozen is not None:
        elected = set(frozen)
    else:
        elected = set()
        best = aggregate_metrics["best"]
        if best is not None:
            best_gain = aggregate_metrics["candidates"][best]["directional_gain"]
            meets_bound = (
                not policy.require_positive_lower_bound or uncertainty_summary.lower_bound > 0.0
            )
            if best_gain >= policy.minimum_practical_gain and meets_bound:
                elected = {best}
    decisions = []
    for candidate in verified_gold_package.candidates:
        values = aggregate_metrics["candidates"].get(
            candidate.name, {"legacy_gain": 0.0, "directional_gain": 0.0}
        )
        is_selected = candidate.name in elected
        decisions.append(
            PlatinumSelectionDecision(
                candidate_id=candidate.candidate_id,
                feature_name=candidate.name,
                selected=is_selected,
                reason_code="selected" if is_selected else "not_selected",
                reason="met the selection policy on discovery folds"
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
    baseline_fold_evidence: dict[str, Any],
    aggregate_metrics: dict[str, Any],
) -> list[CheckResult]:
    metrics = baseline_fold_evidence["metrics"]
    evaluation_rows = _partition_scope(
        verified_silver_package,
        platinum_request.evaluation_policy.evaluation_protocol,
        "evaluation",
    )
    return [
        CheckResult(
            check_id="PLATINUM.FOLDS.COMPLETE",
            passed=len(metrics) >= platinum_request.evaluation_policy.minimum_successful_folds,
            message="Fold evidence is complete",
        ),
        CheckResult(
            check_id="PLATINUM.ROWS.ALIGNED",
            passed=int(metrics["n_validation"].sum()) == int(evaluation_rows.sum()),
            message="Fold validation rows align with the evaluation partition",
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
    options: dict[str, Any] = {
        "platinum_input_fingerprint": platinum_request.platinum_input_fingerprint,
        "platinum_identity_schema_version": PLATINUM_IDENTITY_SCHEMA_VERSION,
    }
    preprocessing = aggregate_metrics.get("preprocessing")
    if preprocessing is not None:
        # ADR 0018 decision 4: fitted preprocessing stays in the provenance.
        options["evaluation_provenance"] = {"preprocessing": preprocessing}
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
            options=options,
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
    baseline_fold_evidence: dict[str, Any],
    candidate_fold_evidence: CandidateFoldEvidence | dict[str, dict[str, pd.DataFrame]],
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
    selected_arm: dict[str, Any] | None = getattr(candidate_fold_evidence, "selected_arm", None)
    enhanced_evidence = selected_arm if selected_arm is not None else baseline_fold_evidence
    # The persisted evidence always carries a paired "enhanced" arm: the frozen
    # selected feature set evaluated on evaluation folds when selection froze a
    # non-empty set, otherwise a copy of the baseline arm so no-candidate
    # packages still reconstruct during loading.
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
    # v2 evidence set (ADR 0018 decision 8): discovery-fold trial arms, the
    # greedy selection transcript, the preprocessing identity, and the typed
    # marker binding the set. Durable evidence without discovery-fold
    # selection evidence cannot be reconstructed offline, so fail closed
    # rather than persisting a v2-shaped package that cannot verify.
    discovery_arms: pd.DataFrame | None = getattr(candidate_fold_evidence, "discovery_arms", None)
    selection: GreedySelectionOutcome | None = getattr(candidate_fold_evidence, "selection", None)
    if discovery_arms is None or selection is None:
        raise DatasetError(
            "Platinum v2 durable evidence requires discovery-fold selection evidence"
        )
    discovery_metrics = pd.concat(
        [
            discovery_arms,
            *(evidence["metrics"] for evidence in candidate_fold_evidence.values()),
        ],
        ignore_index=True,
    )
    staging.write_dataframe("discovery_fold_metrics.parquet", discovery_metrics)
    staging.write_json("selection_steps.json", list(selection.steps))
    staging.write_json("preprocessing.json", aggregate_metrics["preprocessing"])
    staging.write_json(
        "evidence.json",
        PlatinumEvidenceIndex(
            selected=list(selection.selected),
            selection_partition=platinum_request.evaluation_policy.selection_partition,
            evaluation_protocol=platinum_request.evaluation_policy.evaluation_protocol,
            confidence_level=platinum_request.uncertainty_policy.confidence_level,
        ).model_dump(mode="json"),
    )
    staged_artifacts = {item.relative_path: item.media_type for item in staging.artifacts}
    if staged_artifacts != PLATINUM_REQUIRED_ARTIFACTS:
        # Fail closed under ``python -O`` (a bare assert would be stripped).
        raise DatasetError(
            f"Platinum package staged artifacts do not match contract: {sorted(staged_artifacts)}"
        )
    staging.set_manifest(platinum_manifest.model_copy(update={"artifacts": staging.artifacts}))
    ref = artifact_store.commit(staging)
    return PlatinumMaterialization(
        manifest=staging.manifest,
        manifest_ref=ref,
        persisted=True,
        artifact_paths=[item.relative_path for item in staging.artifacts],
    )
