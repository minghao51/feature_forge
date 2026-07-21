"""Deterministic resolution and application of nested resource limits."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from feature_forge.config import Settings
from feature_forge.contracts.orchestration import EffectiveResourcePlan, ResourceConfig


def resolve_resource_plan(
    requested: ResourceConfig,
    *,
    experiment_workers: int | None = None,
) -> EffectiveResourcePlan:
    """Resolve one safe plan with at most one heavy parallel layer."""
    outer = experiment_workers or requested.experiment_workers
    llm = requested.llm_concurrency
    cv = requested.cv_workers
    sandbox = requested.sandbox_workers
    batch = requested.candidate_batch_size
    backend = requested.cv_backend
    model_threads = requested.model_threads
    blas_threads = requested.blas_threads
    adjustments: list[str] = []
    heavy: Literal["outer", "llm", "cv", "none"]

    if outer > 1:
        if any(value > 1 for value in (llm, cv, sandbox, batch)):
            adjustments.append("outer parallelism serializes all per-case heavy work")
        llm = cv = sandbox = batch = 1
        model_threads = blas_threads = 1
        backend = "threading"
        heavy = "outer"
    elif llm > 1:
        if cv > 1:
            adjustments.append("LLM parallelism takes precedence over CV parallelism")
        cv = 1
        backend = "threading"
        heavy = "llm"
    elif cv > 1:
        sandbox = 1
        batch = 1
        model_threads = blas_threads = 1
        heavy = "cv"
    else:
        heavy = "none"

    if model_threads > 1 and blas_threads > 1:
        blas_threads = 1
        adjustments.append("model threading disables nested BLAS threading")
    if sandbox > 1 and batch > 1:
        batch = 1
        adjustments.append("sandbox concurrency disables nested candidate batching")

    return EffectiveResourcePlan(
        experiment_workers=outer,
        llm_concurrency=llm,
        sandbox_workers=sandbox,
        candidate_batch_size=batch,
        cv_workers=cv,
        cv_backend=backend,
        model_threads=model_threads,
        blas_threads=blas_threads,
        memory_budget_mb=requested.memory_budget_mb,
        max_attempts=requested.max_attempts,
        backoff_base_seconds=requested.backoff_base_seconds,
        backoff_max_seconds=requested.backoff_max_seconds,
        heavy_parallel_layer=heavy,
        adjustments=adjustments,
    )


def safer_resource_plan(plan: EffectiveResourcePlan) -> EffectiveResourcePlan | None:
    """Return a strictly safer allocation for a resource retry, if possible."""
    values = plan.model_dump()
    changed = False
    for key in (
        "experiment_workers",
        "llm_concurrency",
        "sandbox_workers",
        "candidate_batch_size",
        "cv_workers",
        "model_threads",
        "blas_threads",
    ):
        if values[key] > 1:
            values[key] = max(1, values[key] // 2)
            changed = True
    if not changed:
        return None
    values["cv_backend"] = "threading"
    values["heavy_parallel_layer"] = next(
        (
            name
            for name, key in (
                ("outer", "experiment_workers"),
                ("llm", "llm_concurrency"),
                ("cv", "cv_workers"),
            )
            if values[key] > 1
        ),
        "none",
    )
    values["adjustments"] = [*values["adjustments"], "resource retry reduced concurrency"]
    return EffectiveResourcePlan.model_validate(values)


def settings_for_plan(settings: Settings, plan: EffectiveResourcePlan) -> Settings:
    """Return a per-case settings copy carrying the effective inner limits."""
    resolved = settings.model_copy(deep=True)
    resolved.llm.max_concurrent_calls = plan.llm_concurrency
    resolved.evaluation.max_cv_workers = plan.cv_workers
    resolved.evaluation.max_sandbox_workers = plan.sandbox_workers
    resolved.evaluation.feature_eval_backend = plan.cv_backend
    resolved.resources = ResourceConfig(
        experiment_workers=plan.experiment_workers,
        llm_concurrency=plan.llm_concurrency,
        sandbox_workers=plan.sandbox_workers,
        candidate_batch_size=plan.candidate_batch_size,
        cv_workers=plan.cv_workers,
        cv_backend=plan.cv_backend,
        model_threads=plan.model_threads,
        blas_threads=plan.blas_threads,
        memory_budget_mb=plan.memory_budget_mb,
        max_attempts=plan.max_attempts,
        backoff_base_seconds=plan.backoff_base_seconds,
        backoff_max_seconds=plan.backoff_max_seconds,
    )
    if plan.memory_budget_mb is not None:
        resolved.evaluation.sandbox_max_memory_mb = plan.memory_budget_mb
    return resolved


@contextmanager
def thread_limits(plan: EffectiveResourcePlan) -> Iterator[None]:
    """Constrain BLAS/OpenMP libraries and restore the caller environment."""
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = str(plan.blas_threads)
    try:
        try:
            from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
        except ImportError:
            yield
        else:
            with threadpool_limits(limits=plan.blas_threads):
                yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
