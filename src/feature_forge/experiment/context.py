"""Resolution of immutable configuration for one experiment case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from feature_forge.config import EvaluationConfig, Settings
from feature_forge.dataflows.profile import ExecutionProfile
from feature_forge.evaluation import CVEvaluator, ModelFactory
from feature_forge.evaluation.kit import EvaluationKit
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import EvaluationError, LLMError
from feature_forge.experiment.execution import ExperimentCase
from feature_forge.observability.structlog_config import get_logger
from feature_forge.storage.hashing import fingerprint

if TYPE_CHECKING:
    from feature_forge.llm.base import LLMClient

logger = get_logger(__name__)

TaskType = Literal["classification", "regression"]

_TASK_METRICS: dict[TaskType, frozenset[str]] = {
    "classification": frozenset({"auc", "acc", "f1"}),
    "regression": frozenset({"rmse", "mae", "r2", "nrmse"}),
}
_DEFAULT_METRIC: dict[TaskType, str] = {
    "classification": "auc",
    "regression": "r2",
}


def validate_metric_for_task(metric: str, task: TaskType, dataset_name: str) -> None:
    """Reject known built-in metrics that contradict dataset task metadata."""
    if metric in set().union(*_TASK_METRICS.values()) and metric not in _TASK_METRICS[task]:
        raise EvaluationError(
            f"Metric '{metric}' is incompatible with {task} task for dataset '{dataset_name}'"
        )


@dataclass(frozen=True)
class CaseExecutionContext:
    """Fully resolved execution policy shared by a method and outer evaluator."""

    case: ExperimentCase
    task: TaskType
    metric: str
    seed: int
    cv_folds: int
    settings: Settings
    evaluation: EvaluationConfig
    model_factory: ModelFactory
    evaluator: CVEvaluator
    run_id: str
    case_fingerprint: str
    execution_profile: ExecutionProfile
    artifact_policy: str
    # Pre-built collaborators shared with every method adapter so methods do
    # not have to self-build (and so `CaseComputation` no longer needs to
    # reflect on each constructor's signature). `llm_client` is None when no
    # API key is configured; LLM-dependent methods (CAAFE unified, LLMFE,
    # Malmus, MALMAS) raise a clearer error at fit time in that case.
    llm_client: LLMClient | None
    sandbox: SandboxedExecutor
    eval_kit: EvaluationKit


def resolve_dataset_task(
    *,
    dataset_name: str,
    loaded: dict[str, Any],
    registry_metadata: dict[str, Any],
    fallback: str,
) -> TaskType:
    """Resolve task from dataset-owned metadata and reject contradictory declarations."""
    loaded_metadata = loaded.get("metadata")
    candidates = [
        loaded.get("task"),
        loaded_metadata.get("task") if isinstance(loaded_metadata, dict) else None,
        registry_metadata.get("task"),
    ]
    declared = {value for value in candidates if value is not None}
    invalid = declared - {"classification", "regression"}
    if invalid:
        raise EvaluationError(
            f"Dataset '{dataset_name}' declares unsupported task values: {sorted(invalid)}"
        )
    if len(declared) > 1:
        raise EvaluationError(
            f"Dataset '{dataset_name}' has conflicting task metadata: {sorted(declared)}"
        )
    resolved = next(iter(declared), fallback)
    if resolved not in {"classification", "regression"}:
        raise EvaluationError(f"Unsupported task: {resolved}")
    return cast(TaskType, resolved)


def resolve_case_context(
    *,
    case: ExperimentCase,
    base_settings: Settings,
    loaded_dataset: dict[str, Any],
    registry_metadata: dict[str, Any],
) -> CaseExecutionContext:
    """Resolve one case before constructing its method or any provider client."""
    task = resolve_dataset_task(
        dataset_name=case.dataset,
        loaded=loaded_dataset,
        registry_metadata=registry_metadata,
        fallback=base_settings.task,
    )
    if case.metric is not None:
        validate_metric_for_task(case.metric, task, case.dataset)
        metric = case.metric
    elif base_settings.metric in _TASK_METRICS[task]:
        metric = base_settings.metric
    else:
        metric = _DEFAULT_METRIC[task]

    settings = base_settings.model_copy(deep=True)
    settings.task = task
    settings.metric = metric
    settings.random_state = case.seed
    if case.cv_folds is not None:
        settings.evaluation.cv_folds = case.cv_folds

    model_factory = ModelFactory(
        random_state=case.seed,
        model_threads=settings.resources.model_threads,
    )
    evaluator = CVEvaluator(config=settings, model_factory=model_factory)
    sandbox = SandboxedExecutor(
        timeout_seconds=settings.evaluation.sandbox_timeout_seconds,
        max_memory_mb=settings.evaluation.sandbox_max_memory_mb,
    )
    eval_kit = EvaluationKit(
        sandbox=sandbox,
        evaluator=evaluator,
        model_factory=model_factory,
    )
    llm_client = _build_llm_client(settings)
    run_id = case.effective_run_id
    case_identity = fingerprint(
        {
            "kind": "case",
            "dataset": case.dataset,
            "method": case.method,
            "model": case.model,
            "task": task,
            "metric": metric,
            "seed": case.seed,
            "cv_folds": settings.evaluation.cv_folds,
            "mode": case.mode,
            "execution_profile": case.execution_profile.value,
            "artifact_policy": case.artifact_policy,
            "settings": settings,
        }
    )
    return CaseExecutionContext(
        case=case,
        task=task,
        metric=metric,
        seed=case.seed,
        cv_folds=settings.evaluation.cv_folds,
        settings=settings,
        evaluation=settings.evaluation,
        model_factory=model_factory,
        evaluator=evaluator,
        run_id=run_id,
        case_fingerprint=case_identity,
        execution_profile=case.execution_profile,
        artifact_policy=case.artifact_policy,
        llm_client=llm_client,
        sandbox=sandbox,
        eval_kit=eval_kit,
    )


def _build_llm_client(settings: Settings) -> LLMClient | None:
    """Best-effort single LLM client shared by all method adapters for a case.

    Mirrors the defensive construction in ``FeatureForge.__init__``: if no API
    key is configured (common for OpenFE-only runs or notebook exploration),
    we log a warning and return None. LLM-dependent methods raise a clearer
    error at fit time via their own validation.
    """
    from feature_forge.llm.factory import create_llm_client

    try:
        return create_llm_client(settings.llm, retry_config=settings.retry)
    except (ValueError, KeyError, TypeError, ImportError, AttributeError, LLMError) as exc:
        logger.warning(
            "case_llm_client_init_failed",
            error=str(exc),
            hint="Set an LLM API key or pass llm_client= explicitly",
        )
        return None
