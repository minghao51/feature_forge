"""Platinum fold evidence, uncertainty, selection, and durable verification."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import t as student_t  # type: ignore[import-untyped]

from feature_forge.contracts import (
    AggregateMetric,
    ArtifactNamespace,
    ArtifactRef,
    CheckResult,
    EnvironmentSnapshot,
    FailureClass,
    GoldPackage,
    Layer,
    ManifestRef,
    ModelSpecification,
    PlatinumExecutionResult,
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
    platinum_input_fingerprint,
)
from feature_forge.dataflows.hamilton_compat import tag
from feature_forge.dataflows.profile import ExecutionProfile, get_profile_policy
from feature_forge.evaluation.cv import CVEvaluator, FoldEvaluation
from feature_forge.evaluation.metrics import get_metric, get_metric_direction
from feature_forge.exceptions import DatasetError
from feature_forge.experiment.execution import ExperimentResult
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
    "report.json": "application/json",
}


def build_platinum_request(
    *,
    run_id: str,
    case_fingerprint: str,
    silver: SilverPackage,
    gold: GoldPackage,
    gold_ref: ManifestRef,
    evaluator: CVEvaluator,
    model_name: str,
    evaluation_policy: Any = None,
    uncertainty_policy: Any = None,
    selection_policy: Any = None,
) -> PlatinumRequest:
    """Resolve a request from verified packages and effective runtime policy."""
    from feature_forge.contracts import EvaluationPolicy, SelectionPolicy, UncertaintyPolicy

    eval_policy = evaluation_policy or EvaluationPolicy()
    unc_policy = uncertainty_policy or UncertaintyPolicy()
    select_policy = selection_policy or SelectionPolicy()
    if evaluator.config.task != silver.bronze.task:
        raise DatasetError("Evaluator task does not match authoritative Silver task")
    if evaluator.config.random_state != evaluator.model_factory.random_state:
        raise DatasetError("Evaluator and model factory seeds differ")
    if gold.request.silver_manifest.run_id != silver.manifest.run_id:
        raise DatasetError("Gold does not reference the authoritative Silver package")
    model_data = evaluator.model_factory.describe_model(model_name, evaluator.config.task)
    resolved_params = dict(model_data["resolved_params"])
    for key in ("n_jobs", "thread_count"):
        if key in resolved_params:
            resolved_params[key] = eval_policy.estimator_threads
    model_data["resolved_params"] = resolved_params
    model = ModelSpecification.model_validate(model_data)
    fold_fp = fingerprint(
        {
            "silver_fingerprint": silver.manifest.layer_fingerprint,
            "folds": silver.fold_assignments.to_dict(orient="records"),
        }
    )
    direction = get_metric_direction(evaluator.config.metric)
    evaluation_identity = {
        **eval_policy.model_dump(mode="json"),
        "metric_direction": direction.value,
        "selection_policy": select_policy.model_dump(mode="json"),
    }
    input_fp = platinum_input_fingerprint(
        gold_fingerprint_value=gold.manifest.layer_fingerprint,
        model_name=model.name,
        model_version=model.distribution_version,
        model_config=model.model_dump(mode="json"),
        metric=evaluator.config.metric,
        fold_fingerprint=fold_fp,
        evaluation_policy=evaluation_identity,
        uncertainty_policy=unc_policy.model_dump(mode="json"),
    )
    return PlatinumRequest(
        run_id=run_id,
        case_fingerprint=case_fingerprint,
        silver_manifest=gold.request.silver_manifest,
        gold_manifest=gold_ref,
        gold_fingerprint=gold.manifest.layer_fingerprint,
        fold_fingerprint=fold_fp,
        platinum_input_fingerprint=input_fp,
        model=model,
        metric=evaluator.config.metric,
        metric_direction=direction,
        evaluation_policy=eval_policy,
        uncertainty_policy=unc_policy,
        selection_policy=select_policy,
    )


def _paired_summary(
    baseline: pd.DataFrame,
    enhanced: pd.DataFrame,
    request: PlatinumRequest,
) -> tuple[AggregateMetric, UncertaintySummary]:
    baseline_scores = baseline.set_index("fold")["score"].sort_index()
    enhanced_scores = enhanced.set_index("fold")["score"].sort_index()
    if not baseline_scores.index.equals(enhanced_scores.index):
        raise DatasetError("Baseline and enhanced fold IDs differ")
    legacy_delta = enhanced_scores - baseline_scores
    sign = 1.0 if request.metric_direction.value == "maximize" else -1.0
    directional = legacy_delta * sign
    count = len(directional)
    if count < request.evaluation_policy.minimum_successful_folds:
        raise DatasetError("Evaluation did not meet the minimum successful-fold policy")
    mean = float(directional.mean())
    std = float(directional.std(ddof=1))
    if math.isnan(std):
        std = 0.0
    se = std / math.sqrt(count)
    alpha = 1.0 - request.uncertainty_policy.confidence_level
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=count - 1))
    margin = critical * se
    aggregate = AggregateMetric(
        metric=request.metric,
        metric_direction=request.metric_direction,
        aggregation=request.evaluation_policy.aggregation,
        baseline_score=float(baseline_scores.mean()),
        enhanced_score=float(enhanced_scores.mean()),
        legacy_gain=float(legacy_delta.mean()),
        directional_gain=mean,
        successful_folds=count,
    )
    uncertainty = UncertaintySummary(
        method=request.uncertainty_policy.method,
        confidence_level=request.uncertainty_policy.confidence_level,
        pair_count=count,
        mean_directional_gain=mean,
        standard_deviation=std,
        standard_error=se,
        lower_bound=mean - margin,
        upper_bound=mean + margin,
    )
    return aggregate, uncertainty


def _with_arm(evidence: FoldEvaluation, arm: str) -> FoldEvaluation:
    metrics = evidence.fold_metrics.copy()
    predictions = evidence.predictions.copy()
    metrics["arm"] = arm
    predictions["arm"] = arm
    return FoldEvaluation(metrics, predictions)


def _evaluate(
    request: PlatinumRequest,
    silver: SilverPackage,
    gold: GoldPackage,
    evaluator: CVEvaluator,
) -> tuple[
    pd.DataFrame, pd.DataFrame, AggregateMetric, UncertaintySummary, list[PlatinumSelectionDecision]
]:
    row_ids = silver.row_ids["row_id"].astype(str).reset_index(drop=True)
    target = silver.canonical_target[silver.bronze.target].reset_index(drop=True)
    base = silver.canonical_features.reset_index(drop=True)
    gold_row_ids = gold.accepted_features["row_id"].astype(str).tolist()
    if request.evaluation_policy.alignment == "strict_order" and gold_row_ids != row_ids.tolist():
        raise DatasetError("Gold row order differs from authoritative Silver order")
    accepted = gold.accepted_features.set_index("row_id").loc[row_ids.tolist()].reset_index()
    context = {"run": request.run_id, "stage": "platinum", "case": request.case_fingerprint}
    baseline = evaluator.evaluate_on_folds(
        base,
        target,
        row_ids,
        silver.fold_assignments,
        arm="baseline",
        model_name=request.model.name,
        error_context=context,
        estimator_threads=request.evaluation_policy.estimator_threads,
        blas_threads=request.evaluation_policy.blas_threads,
    )
    candidates_by_name = {item.name: item for item in gold.candidates}
    candidate_evidence: dict[str, FoldEvaluation] = {}
    decisions: list[PlatinumSelectionDecision] = []
    for name in accepted.columns[1:]:
        enhanced_frame = pd.concat([base, accepted[[name]].reset_index(drop=True)], axis=1)
        evidence = evaluator.evaluate_on_folds(
            enhanced_frame,
            target,
            row_ids,
            silver.fold_assignments,
            arm=f"candidate:{name}",
            model_name=request.model.name,
            error_context={**context, "feature": name},
            estimator_threads=request.evaluation_policy.estimator_threads,
            blas_threads=request.evaluation_policy.blas_threads,
        )
        aggregate, uncertainty = _paired_summary(
            baseline.fold_metrics, evidence.fold_metrics, request
        )
        candidate_evidence[name] = evidence
        if request.selection_policy.profile == "compatibility":
            selected_candidate = aggregate.legacy_gain > 0
            reason_code = (
                "legacy_gain_positive" if selected_candidate else "legacy_gain_not_positive"
            )
        else:
            practical = (
                aggregate.directional_gain >= request.selection_policy.minimum_practical_gain
            )
            lower_ok = (
                not request.selection_policy.require_positive_lower_bound
                or uncertainty.lower_bound > 0
            )
            selected_candidate = practical and lower_ok
            reason_code = (
                "recommended_evidence_passed"
                if selected_candidate
                else "practical_gain_below_threshold"
                if not practical
                else "lower_bound_not_positive"
            )
        candidate = candidates_by_name[name]
        decisions.append(
            PlatinumSelectionDecision(
                candidate_id=candidate.candidate_id,
                feature_name=name,
                selected=selected_candidate,
                reason_code=reason_code,
                reason=f"Selection profile {request.selection_policy.profile}: {reason_code}",
                legacy_gain=aggregate.legacy_gain,
                directional_gain=aggregate.directional_gain,
                lower_bound=uncertainty.lower_bound,
            )
        )
    selected = [item for item in decisions if item.selected]
    cap = request.selection_policy.max_selected_features
    if cap is not None and len(selected) > cap:
        ranked = sorted(selected, key=lambda item: item.directional_gain, reverse=True)
        keep = {item.candidate_id for item in ranked[:cap]}
        decisions = [
            item
            if not item.selected or item.candidate_id in keep
            else item.model_copy(
                update={
                    "selected": False,
                    "reason_code": "max_selected_features",
                    "reason": "Excluded by max_selected_features after evidence ranking",
                }
            )
            for item in decisions
        ]
    selected_names = [item.feature_name for item in decisions if item.selected]
    if not selected_names:
        enhanced = _with_arm(baseline, "enhanced")
    elif len(selected_names) == 1:
        enhanced = _with_arm(candidate_evidence[selected_names[0]], "enhanced")
    else:
        final_frame = pd.concat([base, accepted[selected_names].reset_index(drop=True)], axis=1)
        enhanced = evaluator.evaluate_on_folds(
            final_frame,
            target,
            row_ids,
            silver.fold_assignments,
            arm="enhanced",
            model_name=request.model.name,
            error_context=context,
            estimator_threads=request.evaluation_policy.estimator_threads,
            blas_threads=request.evaluation_policy.blas_threads,
        )
    aggregate, uncertainty = _paired_summary(baseline.fold_metrics, enhanced.fold_metrics, request)
    all_metrics = pd.concat(
        [
            baseline.fold_metrics,
            *(item.fold_metrics for item in candidate_evidence.values()),
            enhanced.fold_metrics,
        ],
        ignore_index=True,
    )
    all_predictions = pd.concat(
        [
            baseline.predictions,
            *(item.predictions for item in candidate_evidence.values()),
            enhanced.predictions,
        ],
        ignore_index=True,
    )
    return all_metrics, all_predictions, aggregate, uncertainty, decisions


def _checks(
    request: PlatinumRequest,
    silver: SilverPackage,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    aggregate: AggregateMetric,
) -> list[CheckResult]:
    baseline = metrics[metrics["arm"] == "baseline"].sort_values("fold")
    enhanced = metrics[metrics["arm"] == "enhanced"].sort_values("fold")
    pred_base = predictions[predictions["arm"] == "baseline"]
    pred_enhanced = predictions[predictions["arm"] == "enhanced"]
    expected_ids = set(silver.row_ids["row_id"].astype(str))
    reconstructed = (
        abs(float(baseline["score"].mean()) - aggregate.baseline_score) < 1e-12
        and abs(float(enhanced["score"].mean()) - aggregate.enhanced_score) < 1e-12
    )
    values = [
        ("PLATINUM.FOLDS.IDENTICAL", baseline["fold"].tolist() == enhanced["fold"].tolist()),
        (
            "PLATINUM.PREDICTIONS.JOINABLE",
            set(pred_base["row_id"]) == expected_ids == set(pred_enhanced["row_id"]),
        ),
        ("PLATINUM.METRICS.RECONSTRUCTABLE", reconstructed),
        (
            "PLATINUM.UPSTREAM.VERIFIED",
            request.gold_fingerprint != "" and request.fold_fingerprint != "",
        ),
    ]
    return [
        CheckResult(
            check_id=check_id,
            passed=passed,
            required=True,
            message=check_id.replace(".", " ").title(),
        )
        for check_id, passed in values
    ]


def _manifest(
    request: PlatinumRequest,
    environment: EnvironmentSnapshot,
    checks: list[CheckResult],
    aggregate: AggregateMetric,
    uncertainty: UncertaintySummary,
    decisions: list[PlatinumSelectionDecision],
) -> RunManifest:
    now = datetime.now(UTC)
    passed = all(item.passed for item in checks if item.required)
    state = RunState.SUCCEEDED if passed else RunState.FAILED
    materialization_fp = fingerprint(
        {
            "platinum_input_fingerprint": request.platinum_input_fingerprint,
            "aggregate": aggregate,
            "uncertainty": uncertainty,
            "decisions": decisions,
        }
    )
    return RunManifest(
        layer=Layer.PLATINUM,
        package_kind="platinum",
        layer_fingerprint=materialization_fp,
        run_id=request.run_id,
        case_fingerprint=request.case_fingerprint,
        state=state,
        request=RunRequest(
            dataset="silver",
            method="platinum",
            model=request.model.name,
            seed=request.model.seed,
            options={"platinum_input_fingerprint": request.platinum_input_fingerprint},
        ),
        environment=environment,
        upstream_manifests=[request.silver_manifest, request.gold_manifest],
        stages=[
            StageResult(
                stage="platinum",
                state=state,
                started_at=now,
                finished_at=now,
                checks=checks,
                failure_class=None if passed else FailureClass.DETERMINISTIC,
            )
        ],
        created_at=now,
        completed_at=now if passed else None,
    )


def _json_artifact(store: LocalArtifactStore, ref: ManifestRef, path: str) -> Any:
    resolved = store.resolve(
        ArtifactRef(layer=Layer.PLATINUM, run_id=ref.run_id, relative_path=path)
    )
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_platinum_package(store: LocalArtifactStore, ref: ManifestRef) -> PlatinumPackage:
    """Load and semantically verify a complete Platinum package."""
    if ref.layer is not Layer.PLATINUM:
        raise DatasetError("Platinum loading requires a Platinum manifest reference")
    try:
        manifest = store.load_manifest(ref)
        by_path = {item.relative_path: item for item in manifest.artifacts}
        if set(by_path) != set(PLATINUM_REQUIRED_ARTIFACTS):
            raise ValueError("Platinum artifact set does not match its contract")
        for path, media_type in PLATINUM_REQUIRED_ARTIFACTS.items():
            item = by_path[path]
            if (
                not item.required
                or item.media_type != media_type
                or item.layer is not Layer.PLATINUM
            ):
                raise ValueError(f"invalid required Platinum artifact: {path}")
        request = PlatinumRequest.model_validate(_json_artifact(store, ref, "request.json"))
        from feature_forge.dataflows.gold import load_gold_package
        from feature_forge.dataflows.silver import load_silver_package

        silver = load_silver_package(store, request.silver_manifest)
        gold = load_gold_package(store, request.gold_manifest)
        silver_manifest = silver.manifest
        gold_manifest = gold.manifest
        folds = pd.read_parquet(
            store.resolve(
                ArtifactRef(
                    layer=Layer.PLATINUM,
                    run_id=ref.run_id,
                    relative_path="fold_assignments.parquet",
                )
            )
        )
        metrics = pd.read_parquet(
            store.resolve(
                ArtifactRef(
                    layer=Layer.PLATINUM, run_id=ref.run_id, relative_path="fold_metrics.parquet"
                )
            )
        )
        predictions = pd.read_parquet(
            store.resolve(
                ArtifactRef(
                    layer=Layer.PLATINUM, run_id=ref.run_id, relative_path="predictions.parquet"
                )
            )
        )
        aggregate = AggregateMetric.model_validate(
            _json_artifact(store, ref, "aggregate_metrics.json")
        )
        uncertainty = UncertaintySummary.model_validate(
            _json_artifact(store, ref, "uncertainty.json")
        )
        decisions = [
            PlatinumSelectionDecision.model_validate(item)
            for item in _json_artifact(store, ref, "selection_decisions.json")
        ]
        checks = [
            CheckResult.model_validate(item) for item in _json_artifact(store, ref, "checks.json")
        ]
    except Exception as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"Unable to load verified Platinum package: {exc}") from exc
    baseline = metrics[metrics["arm"] == "baseline"].sort_values("fold")
    enhanced = metrics[metrics["arm"] == "enhanced"].sort_values("fold")
    expected_aggregate, expected_uncertainty = _paired_summary(baseline, enhanced, request)
    keys = ["fold", "row_id"]
    base_keys = (
        predictions[predictions["arm"] == "baseline"][keys].sort_values(keys).reset_index(drop=True)
    )
    enhanced_keys = (
        predictions[predictions["arm"] == "enhanced"][keys].sort_values(keys).reset_index(drop=True)
    )
    recomputed_fp = fingerprint(
        {
            "platinum_input_fingerprint": request.platinum_input_fingerprint,
            "aggregate": aggregate,
            "uncertainty": uncertainty,
            "decisions": decisions,
        }
    )
    expected_fold_fp = fingerprint(
        {
            "silver_fingerprint": silver_manifest.layer_fingerprint,
            "folds": folds.to_dict(orient="records"),
        }
    )
    try:
        pd.testing.assert_frame_equal(
            folds.reset_index(drop=True),
            silver.fold_assignments.reset_index(drop=True),
            check_dtype=True,
        )
    except AssertionError as exc:
        raise DatasetError(
            "Platinum fold assignments differ from authoritative Silver evidence"
        ) from exc

    required_metric_columns = {
        "arm",
        "fold",
        "metric",
        "score",
        "n_train",
        "n_validation",
        "status",
    }
    required_prediction_columns = {
        "arm",
        "fold",
        "row_id",
        "target_json",
        "prediction_json",
    }
    if not required_metric_columns.issubset(
        metrics.columns
    ) or not required_prediction_columns.issubset(predictions.columns):
        raise DatasetError("Platinum fold evidence schema is incomplete")
    if metrics.duplicated(["arm", "fold"]).any() or predictions.duplicated(["arm", "row_id"]).any():
        raise DatasetError("Platinum fold evidence contains duplicate arm/fold/row keys")

    expected_candidate_names = gold.accepted_features.columns[1:].tolist()
    expected_candidate_arms = {f"candidate:{name}" for name in expected_candidate_names}
    expected_arms = {"baseline", "enhanced", *expected_candidate_arms}
    actual_metric_arms = {str(arm) for arm in metrics["arm"].unique()}
    actual_prediction_arms = {str(arm) for arm in predictions["arm"].unique()}
    actual_candidate_arms = {
        str(arm) for arm in metrics["arm"].unique() if str(arm).startswith("candidate:")
    }
    decision_names = [item.feature_name for item in decisions]
    if (
        actual_candidate_arms != expected_candidate_arms
        or actual_metric_arms != expected_arms
        or actual_prediction_arms != expected_arms
        or set(decision_names) != set(expected_candidate_names)
        or len(decision_names) != len(set(decision_names))
    ):
        raise DatasetError("Platinum candidate evidence does not match verified Gold candidates")

    for arm in expected_arms:
        arm_keys = (
            predictions[predictions["arm"] == arm][keys].sort_values(keys).reset_index(drop=True)
        )
        if not arm_keys.equals(base_keys):
            raise DatasetError("Platinum arms do not share identical fold/row prediction keys")

    metric_fn = get_metric(request.metric)
    for metric_row in metrics.itertuples(index=False):
        arm_rows = predictions[
            (predictions["arm"] == metric_row.arm) & (predictions["fold"] == metric_row.fold)
        ]
        targets = np.asarray([json.loads(value) for value in arm_rows["target_json"]])
        predicted_values = [json.loads(value) for value in arm_rows["prediction_json"]]
        predicted = np.asarray(predicted_values)
        reconstructed_score = float(metric_fn(targets, predicted))
        if (
            metric_row.metric != request.metric
            or metric_row.status != "succeeded"
            or len(arm_rows) != metric_row.n_validation
            or not math.isclose(
                reconstructed_score,
                cast(float, metric_row.score),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise DatasetError("Platinum fold score does not reconstruct from predictions")

    target_values = predictions.pivot(index=["fold", "row_id"], columns="arm", values="target_json")
    if target_values.nunique(axis=1).gt(1).any():
        raise DatasetError("Platinum arms disagree on authoritative targets")

    decision_by_name = {item.feature_name: item for item in decisions}
    candidate_summaries: dict[str, tuple[AggregateMetric, UncertaintySummary]] = {}
    for name in expected_candidate_names:
        candidate_metrics = metrics[metrics["arm"] == f"candidate:{name}"].sort_values("fold")
        candidate_aggregate, candidate_uncertainty = _paired_summary(
            baseline, candidate_metrics, request
        )
        candidate_summaries[name] = (candidate_aggregate, candidate_uncertainty)
        decision = decision_by_name[name]
        if not (
            math.isclose(decision.legacy_gain, candidate_aggregate.legacy_gain, abs_tol=1e-12)
            and math.isclose(
                decision.directional_gain,
                candidate_aggregate.directional_gain,
                abs_tol=1e-12,
            )
            and math.isclose(decision.lower_bound, candidate_uncertainty.lower_bound, abs_tol=1e-12)
        ):
            raise DatasetError("Platinum selection decision does not match candidate evidence")

    policy_selected = {
        name
        for name, (candidate_aggregate, candidate_uncertainty) in candidate_summaries.items()
        if (
            candidate_aggregate.legacy_gain > 0
            if request.selection_policy.profile == "compatibility"
            else candidate_aggregate.directional_gain
            >= request.selection_policy.minimum_practical_gain
            and (
                not request.selection_policy.require_positive_lower_bound
                or candidate_uncertainty.lower_bound > 0
            )
        )
    }
    final_selected = set(policy_selected)
    cap = request.selection_policy.max_selected_features
    if cap is not None and len(final_selected) > cap:
        final_selected = set(
            sorted(
                final_selected,
                key=lambda name: candidate_summaries[name][0].directional_gain,
                reverse=True,
            )[:cap]
        )
    if {item.feature_name for item in decisions if item.selected} != final_selected:
        raise DatasetError("Platinum selection outcomes do not match the declared policy")
    gold_candidates = {item.name: item for item in gold.candidates}
    for decision in decisions:
        if decision.feature_name in policy_selected and decision.feature_name not in final_selected:
            expected_code = "max_selected_features"
            expected_reason = "Excluded by max_selected_features after evidence ranking"
        elif request.selection_policy.profile == "compatibility":
            expected_code = (
                "legacy_gain_positive"
                if decision.feature_name in policy_selected
                else "legacy_gain_not_positive"
            )
            expected_reason = f"Selection profile compatibility: {expected_code}"
        else:
            candidate_aggregate, candidate_uncertainty = candidate_summaries[decision.feature_name]
            practical = (
                candidate_aggregate.directional_gain
                >= request.selection_policy.minimum_practical_gain
            )
            expected_code = (
                "recommended_evidence_passed"
                if decision.feature_name in policy_selected
                else "practical_gain_below_threshold"
                if not practical
                else "lower_bound_not_positive"
            )
            expected_reason = f"Selection profile recommended: {expected_code}"
        if (
            decision.candidate_id != gold_candidates[decision.feature_name].candidate_id
            or decision.reason_code != expected_code
            or decision.reason != expected_reason
        ):
            raise DatasetError("Platinum selection decision metadata does not reconstruct")
    if (
        manifest.layer is not Layer.PLATINUM
        or manifest.package_kind != "platinum"
        or manifest.run_id != request.run_id
        or manifest.upstream_manifests != [request.silver_manifest, request.gold_manifest]
        or manifest.layer_fingerprint != recomputed_fp
        or gold_manifest.layer_fingerprint != request.gold_fingerprint
        or expected_fold_fp != request.fold_fingerprint
        or request.silver_manifest != gold.request.silver_manifest
        or request.model.task != silver.bronze.task
        or request.metric != aggregate.metric
        or request.model.seed != manifest.request.seed
        or request.model.name != manifest.request.model
        or any(
            request.model.resolved_params.get(key) != request.evaluation_policy.estimator_threads
            for key in ("n_jobs", "thread_count")
            if key in request.model.resolved_params
        )
        or aggregate != expected_aggregate
        or uncertainty != expected_uncertainty
        or not base_keys.equals(enhanced_keys)
        or folds["row_id"].duplicated().any()
        or set(base_keys["row_id"]) != set(folds["row_id"].astype(str))
        or any(item.required and not item.passed for item in checks)
    ):
        raise DatasetError("Verified Platinum package contains inconsistent semantic evidence")
    return PlatinumPackage(
        manifest=manifest,
        request=request,
        fold_assignments=folds,
        fold_metrics=metrics,
        predictions=predictions,
        aggregate=aggregate,
        uncertainty=uncertainty,
        decisions=decisions,
        checks=checks,
    )


def experiment_result_from_package(
    package: PlatinumPackage,
    *,
    silver: SilverPackage,
    gold: GoldPackage,
    manifest_uri: str | None,
) -> dict[str, Any]:
    """Build the append-only public result exclusively from verified evidence."""
    aggregate = package.aggregate
    uncertainty = package.uncertainty
    selected_count = sum(item.selected for item in package.decisions)
    return asdict(
        ExperimentResult(
            dataset=silver.manifest.request.dataset,
            method=gold.request.method_name,
            model=package.request.model.name,
            seed=package.request.model.seed,
            cv_score=aggregate.enhanced_score,
            gain=aggregate.legacy_gain,
            baseline_score=aggregate.baseline_score,
            num_features_generated=gold.counts.accepted_output_columns,
            run_id=package.request.run_id,
            case_fingerprint=package.request.case_fingerprint,
            state=package.manifest.state.value,
            manifest_uri=manifest_uri,
            uncertainty={
                "mean_directional_gain": uncertainty.mean_directional_gain,
                "standard_deviation": uncertainty.standard_deviation,
                "standard_error": uncertainty.standard_error,
                "lower_bound": uncertainty.lower_bound,
                "upper_bound": uncertainty.upper_bound,
                "confidence_level": uncertainty.confidence_level,
            },
            num_candidate_features=gold.counts.candidates_proposed,
            num_executed_features=gold.counts.candidates_executed,
            num_accepted_features=selected_count,
            num_accepted_output_columns=selected_count,
            feature_failure_counts=gold.counts.failure_counts,
            silver_fingerprint=silver.manifest.layer_fingerprint,
            gold_fingerprint=gold.manifest.layer_fingerprint,
            platinum_fingerprint=package.manifest.layer_fingerprint,
            platinum_manifest_uri=manifest_uri,
            directional_gain=aggregate.directional_gain,
            selection_profile=package.request.selection_policy.profile,
            metric=package.request.metric,
            metric_direction=package.request.metric_direction.value,
        )
    )


@tag(layer="platinum", cost="expensive", persistence="boundary", owner="evaluation")
def execute_platinum(
    *,
    artifact_store: LocalArtifactStore,
    silver_ref: ManifestRef,
    gold_ref: ManifestRef,
    evaluator: CVEvaluator,
    model_name: str,
    run_id: str,
    case_fingerprint: str,
    environment_snapshot: EnvironmentSnapshot,
    evaluation_policy: Any = None,
    uncertainty_policy: Any = None,
    selection_policy: Any = None,
    execution_profile: ExecutionProfile = ExecutionProfile.PRODUCTION,
) -> PlatinumExecutionResult:
    """Evaluate verified Gold evidence and atomically publish Platinum evidence."""
    from feature_forge.dataflows.gold import load_gold_package
    from feature_forge.dataflows.silver import load_silver_package

    silver = load_silver_package(artifact_store, silver_ref)
    gold = load_gold_package(artifact_store, gold_ref)
    if gold.request.silver_manifest != silver_ref:
        raise DatasetError("Gold and Platinum do not reference the same Silver package")
    request = build_platinum_request(
        run_id=run_id,
        case_fingerprint=case_fingerprint,
        silver=silver,
        gold=gold,
        gold_ref=gold_ref,
        evaluator=evaluator,
        model_name=model_name,
        evaluation_policy=evaluation_policy,
        uncertainty_policy=uncertainty_policy,
        selection_policy=selection_policy,
    )
    policy = get_profile_policy(execution_profile)
    namespace = ArtifactNamespace(layer=Layer.PLATINUM, run_id=run_id)
    if policy.persistence_enabled:
        existing = artifact_store.get_manifest_ref(namespace)
        if existing is not None:
            package = load_platinum_package(artifact_store, existing)
            if package.request.platinum_input_fingerprint != request.platinum_input_fingerprint:
                raise DatasetError("Platinum run ID already contains different evidence")
            uri = f"04_platinum/runs/{request.run_id}/manifest.json"
            return PlatinumExecutionResult(
                materialization=PlatinumMaterialization(
                    manifest=package.manifest,
                    manifest_ref=existing,
                    persisted=True,
                    artifact_paths=[item.relative_path for item in package.manifest.artifacts],
                ),
                package=package,
                experiment_result=experiment_result_from_package(
                    package, silver=silver, gold=gold, manifest_uri=uri
                ),
            )

    metrics, predictions, aggregate, uncertainty, decisions = _evaluate(
        request, silver, gold, evaluator
    )
    checks = _checks(request, silver, metrics, predictions, aggregate)
    manifest = _manifest(request, environment_snapshot, checks, aggregate, uncertainty, decisions)
    if not policy.persistence_enabled:
        materialization = PlatinumMaterialization(manifest=manifest, persisted=False)
        package = PlatinumPackage(
            manifest=manifest,
            request=request,
            fold_assignments=silver.fold_assignments,
            fold_metrics=metrics,
            predictions=predictions,
            aggregate=aggregate,
            uncertainty=uncertainty,
            decisions=decisions,
            checks=checks,
        )
        return PlatinumExecutionResult(
            materialization=materialization,
            package=package,
            experiment_result=experiment_result_from_package(
                package, silver=silver, gold=gold, manifest_uri=None
            ),
        )
    staging = artifact_store.begin(namespace)
    staging.write_json("request.json", request.model_dump(mode="json"))
    staging.write_dataframe("fold_assignments.parquet", silver.fold_assignments)
    staging.write_dataframe("fold_metrics.parquet", metrics)
    staging.write_dataframe("predictions.parquet", predictions)
    staging.write_json("aggregate_metrics.json", aggregate.model_dump(mode="json"))
    staging.write_json("uncertainty.json", uncertainty.model_dump(mode="json"))
    staging.write_json(
        "selection_decisions.json", [item.model_dump(mode="json") for item in decisions]
    )
    staging.write_json("checks.json", [item.model_dump(mode="json") for item in checks])
    staging.write_json(
        "report.json",
        {
            "aggregate": aggregate.model_dump(mode="json"),
            "uncertainty": uncertainty.model_dump(mode="json"),
            "selection_decisions": [item.model_dump(mode="json") for item in decisions],
            "model": request.model.model_dump(mode="json"),
        },
    )
    staging.set_manifest(manifest)
    manifest_ref = artifact_store.commit(staging)
    package = load_platinum_package(artifact_store, manifest_ref)
    return PlatinumExecutionResult(
        materialization=PlatinumMaterialization(
            manifest=cast(RunManifest, staging.manifest),
            manifest_ref=manifest_ref,
            persisted=True,
            artifact_paths=[item.relative_path for item in package.manifest.artifacts],
        ),
        package=package,
        experiment_result=experiment_result_from_package(
            package,
            silver=silver,
            gold=gold,
            manifest_uri=f"04_platinum/runs/{request.run_id}/manifest.json",
        ),
    )
