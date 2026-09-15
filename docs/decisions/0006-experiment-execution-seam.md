# ADR 0006: Experiment execution seam behind the platform facade

- **Status:** Accepted
- **Date:** 2026-08-29
- **Related:** `docs/plan/10_experimental_platform_refactor.md`;
  `docs/plan/11_platform_refactor_review.md`;
  `docs/plan/03_key_design_decisions.md` (§8); ADR 0001, ADR 0004

## Context

Users had to hand-wire `ExperimentMatrix` + `ExperimentRunner` +
registries + `CVEvaluator` (~15 lines) to run a comparison, and
orchestration side effects (tracker lifecycle, registry wiring) were
entangled with pure case computation, so process-pool execution and
sequential execution could not share a result shape. The original
matrix-runner classes have since been removed; the platform is the only
supported entry point for experiment matrices.

This record was written retroactively on 2026-08-29, after
implementation; see Related for the originating plan documents.

## Decision

1. **`ExperimentalPlatform` facade.** `src/feature_forge/platform.py`
   exposes `run(datasets, methods, models, ...)` plus
   `report`/`report_best`/`to_dataframe`, `list_*` introspection, and
   instance-local `register_method`/`register_dataset`/`register_model`/
   `register_metric` (programmatic registration never mutates global
   registry state).
2. **Normalized case/result shapes.** An experiment is a list of frozen
   `ExperimentCase` dataclasses (`dataset`, `method`, `model`, `seed`,
   optional `mode`, `cv_folds`, `run_id`) producing `ExperimentResult`
   records with a nullable `error` — one shape for every backend.
3. **Execution-backend seam.** `ExperimentCaseExecutor` owns tracker
   lifecycle and case behavior in the main process; an `ExecutionBackend`
   ABC has `SequentialExecutionAdapter` (in-process) and
   `ProcessPoolExecutionAdapter` (workers call the top-level
   `run_case(payload)` seam).
4. **Explicit serialization boundary.** `parallel=True` requires
   pickleable payloads, so only registry/entry-point-discovered methods
   may run in parallel; instance-local classes from
   `platform.register_method(...)` are rejected with a clear error and
   must run sequentially.
5. **Opt-in experiment tracking.** Trackers are chosen by
   `TrackerConfig.backend` (`wandb` | `mlflow` | `none`) via
   `create_tracker_from_config`, defaulting to `none` — a deliberate
   deviation from the plan's WandB default so default runs never require
   tracker credentials (see `REPORT_LOG.md` R11/R17).
6. **Registry-backed plugin hooks.** Datasets, models, and metrics are
   discoverable via `feature_forge.datasets`/`.models`/`.metrics`
   entry-point groups; all registries share one
   `discover_entry_points` helper with load-error handling and
   duplicate-name warnings.

## Consequences

- One-liner experiment runs with a uniform result shape regardless of
  sequential or process-pool execution; side effects are isolated in the
  main process.
- The parallel constraint is real: ad-hoc registered methods cannot use
  the process pool, and worker failures surface as errors on the result
  rather than crashing the matrix.
- Safe-by-default tracking means WandB/MLflow must be opted into
  explicitly; users who want tracking by default configure it once in
  `settings.yaml`.
- Adding a new orchestration concern (e.g. remote backends, queuing)
  means a new `ExecutionBackend` implementation — not edits scattered
  across the facade.
