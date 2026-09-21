# Medallion Refactor: PR Handoff Index

**Repository:** `minghao51/feature_forge`
**Date:** 2026-07-14
**Status:** PR 0-7 implemented within their recorded boundaries; PR 7 requires an
application-owned `LayerExecutor` for full durable materialization; PR 8 remains deferred
**Master specification:** [`2026-07-13-medallion-refactor.md`](2026-07-13-medallion-refactor.md)

## Purpose

This index converts the master Hamilton + medallion-lite specification and the
2026-07-14 implementation review into independently executable PR handoffs.
Each PR has a narrow contract, explicit dependencies, acceptance tests, and a
required closing handoff so progress can be reviewed without rereading the full
master document.

## Current baseline

PR 0-2 established:

- legacy `ExperimentResult` characterization;
- architecture decision records;
- versioned run, stage, dataset, and artifact contracts;
- secret-free fingerprint helpers;
- atomic local artifact packages;
- Bronze/Silver Hamilton nodes and profiles;
- offline verified Silver loading;
- generated Silver DAG and optional pipeline CI coverage.

The default `ExperimentalPlatform` path remains the compatible imperative path.
The opt-in layer-boundary orchestration spans Bronze through Platinum with resumable
execution, verification, derived catalog indexing, and canonical operations docs. The library
does not ship a turnkey layer materializer: full durable execution remains application-integrated
through a required module-level `LayerExecutor`.

## Required PR sequence

| Phase | Handoff | Status | Depends on | Primary outcome |
|---|---|---|---|---|
| PR 2.5 | [`2026-07-14-pr2.5-foundation-stabilization.md`](2026-07-14-pr2.5-foundation-stabilization.md) | Complete | PR 0-2 | Correct per-case configuration, artifact integrity, fingerprints, dependency/security baseline |
| PR 3 | [`2026-07-14-pr3-gold-offline-replay.md`](2026-07-14-pr3-gold-offline-replay.md) | Implemented | PR 2.5 | Gold method adapter, candidate/accepted features, zero-provider replay |
| PR 4 | [`2026-07-14-pr4-platinum-evidence.md`](2026-07-14-pr4-platinum-evidence.md) | Implemented 2026-07-14 | PR 3 | Durable fold evidence, uncertainty, compatible results |
| PR 5 | [`2026-07-14-pr5-scheduler-resume-resources.md`](2026-07-14-pr5-scheduler-resume-resources.md) | Implemented 2026-07-14 | PR 4 | Failure-aware scheduler, resume, resource policy |
| PR 6 | [`2026-07-14-pr6-catalog-verification-observability.md`](2026-07-14-pr6-catalog-verification-observability.md) | Implemented and verified 2026-07-15 | PR 5 | Rebuildable DuckDB catalog, verification CLI, lifecycle telemetry |
| PR 7 | [`2026-07-14-pr7-docs-operations.md`](2026-07-14-pr7-docs-operations.md) | Verified within library boundary 2026-07-15; turnkey materializer not shipped | PR 6 | Canonical operational docs, generated references, freshness checks, explicit application executor boundary |
| PR 8 gate | [`2026-07-14-pr8-decision-readiness.md`](2026-07-14-pr8-decision-readiness.md) | Evidence gathering planned; implementation not authorized | PR 7 | Interaction, scale, schema, security, ownership, hosting, and decision evidence |
| PR 8 implementation | [`2026-07-14-pr8-optional-astro-catalog.md`](2026-07-14-pr8-optional-astro-catalog.md) | Deferred; blocked pending an approved PR 8 gate | PR 8 gate approval | Optional read-only browser catalog |

## Cross-PR invariants

Every implementation must preserve these rules:

1. `ExperimentalPlatform.run()` remains compatible until an explicitly
   documented migration changes it.
2. Hamilton is optional for legacy users and remains an inner case dataflow,
   not the outer experiment scheduler.
3. Third-party methods do not need to import Hamilton.
4. Cache entries are disposable; only verified manifests and `_SUCCESS`
   packages are durable evidence.
5. Secrets never enter fingerprints, manifests, logs, catalog rows, generated
   docs, or browser snapshots.
6. Replay must not initialize or call an LLM provider.
7. Row IDs and persisted fold assignments are authoritative after Silver.
8. Baseline and enhanced evaluation must consume identical folds.
9. Layer reuse is controlled by layer-specific fingerprints, not a monolithic
   case fingerprint.
10. Required CI and tests remain offline and deterministic.
11. Unrelated worktree changes remain untouched; no phase may absorb another
    phase for convenience.
12. Performance-sensitive changes require a recorded before/after measurement.

## Layer identity model

| Layer | Reuse fingerprint must include | Must not include |
|---|---|---|
| Bronze | source identity/checksum, source contract version | method, prompt, model, metric |
| Silver | Bronze identity, target/task, canonicalization, split policy/seed | method, prompt, model, metric |
| Gold | Silver identity, method/version/config, prompt bundle, generated code/spec contract, selection policy when it changes accepted output | evaluation model and reporting-only configuration |
| Platinum | Gold identity, model/version/config, metric, persisted folds, evaluation/uncertainty policy | tracker backend, presentation configuration |
| Run/case | references all selected stage identities plus orchestration metadata | raw secret values |

## Integration gates

No phase is complete until:

- its focused tests pass;
- `uv run ruff check .` passes;
- `uv run mypy src` passes;
- the relevant full offline pytest lane passes;
- `uv lock --check` and `git diff --check` pass;
- repository and documentation hygiene pass;
- strict MkDocs passes when docs or generated diagrams change;
- `uv run python scripts/run_pip_audit.py` passes or an explicit, reviewed,
  time-bounded exception is documented;
- `.planning/STATE.md` and the next PR handoff reflect the actual checkout.

## Ownership and status update convention

At the end of each PR, update its handoff with:

- status and completion date;
- commits/PR URL if applicable;
- files and contracts changed;
- compatibility and migration behavior;
- exact verification commands and results;
- performance measurements;
- risks and unresolved decisions;
- the next PR's confirmed prerequisites.

Do not mark a PR complete based only on green unit tests. Its acceptance
criteria and cross-PR invariants must all be satisfied.
