# ADR 0016: Remove the legacy execution engine after qualification

- **Status:** Accepted
- **Date:** 2026-09-14
- **Supersedes:** ADR 0012
- **Related:** ADR 0006, ADR 0013, ADR 0014, ADR 0015,
  `docs/plan/21_hamilton_default_execution_handoff.md`,
  `experiments/legacy_removal/2026-09-14/qualification_report.json`,
  `REPORT_LOG.md`

## Context

ADR 0012 selected Hamilton as the default case dataflow engine but retained the
imperative `legacy` engine as an explicit rollback path until six removal gates
were satisfied. Maintaining two execution paths now duplicates orchestration,
evaluation, configuration, tests, and documentation, and allows their semantics
to drift.

The technical gates are complete:

1. The full suite passes on Python 3.11, 3.12, and 3.13 (954 passed and 9
   expected optional-XGBoost skips on each interpreter).
2. Sequential execution and real process-worker recomputation agree under both
   the Linux-default and explicit `spawn` contexts.
3. A two-seed, bundled breast-cancer matrix produced Hamilton/legacy baseline,
   enhanced, and gain deltas no larger than `3.71e-17`, within the documented
   absolute tolerance of `1e-9`.
4. Resume, Hamilton-cache deletion/corruption, package-corruption recovery, and
   provider-free replay drills pass.
5. The real matrix persisted resolved secret-free configuration, runtime
   provenance, package verification, cache status, and reproducibility commands
   under `experiments/legacy_removal/2026-09-14/`.

On 2026-09-14 the maintainer accepted the sixth gate and chose removal rather
than extending the compatibility period.

## Decision

Hamilton is the sole supported case execution engine. A dedicated follow-up
change will remove the in-process legacy compatibility path.

1. Remove `ExecutionEngine.LEGACY`, `_run_legacy`, and production-only legacy
   executor wiring once their remaining callers and tests are migrated.
2. Remove the `legacy` artifact-policy branch. Medallion layer boundaries and
   verified packages remain mandatory for every platform run.
3. Reject stale `engine=legacy` configuration with an actionable migration
   error; never silently reinterpret it as Hamilton and never mix engines in one
   attempt.
4. Preserve the qualification reports as the historical parity record. Tests
   that require live dual-engine execution may be replaced by Hamilton invariant
   tests after the removal lands.
5. Preserve `MethodRegistry`, mandatory `LLMClient` DiskCache, sandbox execution,
   cache/package isolation, and worker-local runtime boundaries.
6. Operational rollback after removal is a package/version rollback to the last
   compatibility release, not a runtime engine switch.
7. Implement removal separately from fail-fast/cancellation work so each change
   has a narrow review and rollback surface.

## Consequences

- One execution path becomes authoritative, reducing drift and simplifying
  configuration, scheduling, documentation, and support.
- Users with `FF_DATAFLOW__ENGINE=legacy` or YAML `engine: legacy` must migrate
  their configuration before upgrading to the removal release.
- The immediate runtime rollback switch disappears. Operators must retain the
  prior compatible package/release if they need emergency rollback.
- Historical legacy parity remains auditable from the qualification report, but
  future experiments execute only through verified Hamilton stage packages.
- This ADR authorizes removal; it does not itself delete code. The removal change
  must update tests, migration guidance, operations documentation, plan status,
  and `REPORT_LOG.md` and pass the full repository validation suite.
