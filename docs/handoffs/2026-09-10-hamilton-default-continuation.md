# Hamilton-default continuation handoff

> Superseded for new threads by
> `docs/handoffs/2026-09-14-legacy-removal-and-fail-fast-plan.md`.
> This file is retained as implementation history.

**Updated:** 2026-09-12
**Workspace:** `/home/howt/work/feature_forge`
**Authoritative plan:** `docs/plan/21_hamilton_default_execution_handoff.md`

## Start here

PRs 1–6 from the authoritative plan are implemented in the working tree.
Next collect section 16 legacy-removal evidence without deleting the rollback
engine: representative metric parity, recovery/replay drills, and one real
matrix with complete reproducibility metadata. Removal requires a maintainer
ADR.

## Working-tree progress

- PR 1: ADRs 0012–0015, typed dataflow/cache settings, default-on Hamilton
  cache policy, and core `apache-hamilton>=1.90,<2` dependency.
- PR 2A–2C: strict durable contracts, secret-free hashing, atomic local
  packages, verification, identities, and contiguous resume foundations.
- PR 3A–3E: separate Bronze/Silver/Gold/Platinum drivers, exact node
  inventories, deterministic stage DAGs, provider-free replay, Platinum
  reconstruction, and cache qualification.
- PR 4: worker-local `HamiltonLayerExecutor`; source-once stage handoff;
  incremental verified reuse; exact ordered lineage; lifecycle events; typed
  stage/failure results; public Hamilton default; explicit legacy fallback;
  sequential/process parity; and `run plan --format json`.
- PR 5: redacted bounded Hamilton telemetry; concurrent atomic cache stores;
  cache status/inspection/retention/clear; artifact list/verification CLI;
  generated stage DAGs; cache qualification; and recovery/user documentation.
- PR 6: canonical MALMAS baseline identity; successful unique code replay;
  schema failure evidence; direction-aware selection; fail-closed mode/agent
  validation; explicit warm start; random-forest default; and named heavy extras.

## Validation

- Required sync passed; full standard+Intel suite: 922 passed, 9 expected
  optional-XGBoost skips, with explicit BLAS/OpenMP single-thread limits.
- Clean standard wheel installation excludes heavy extras and fits/predicts
  with the random-forest default.
- Cache qualification reached 10/10 eligible warm hits. The tiny fixture was
  slower warm (73.3 ms vs 21.6 ms cold); representative performance follow-up
  is tracked in `.planning/STATE.md`, and no speedup claim is made.
- `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`,
  and `uv run mypy tests` pass.
- Repository hygiene and documentation reference checks pass.
- Public full-case integration verifies all four packages, complete second-run
  reuse, process parity, lifecycle events, and Gold/Platinum invalidation when
  LLM identity changes.

## Constraints and next assignment

- The working tree remains intentionally dirty with unrelated prior changes.
  Preserve them; never reset, clean, or stash broadly.
- Treat `origin/feat/medallion-refactor` / `690a764` as read-only reference.
- Keep fresh temporary Hamilton cache paths in tests.
- Keep `engine=legacy` as the explicit rollback route; never silently fall back
  within a Hamilton attempt.
- PRs 1–6 are complete in the working tree.
- Keep `engine=legacy` until all plan section 16 gates and a maintainer ADR pass.
