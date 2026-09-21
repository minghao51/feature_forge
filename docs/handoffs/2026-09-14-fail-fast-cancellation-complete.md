# Fail-fast and cooperative cancellation — implementation complete

**Updated:** 2026-09-14  
**Workspace:** `/home/howt/work/feature_forge`  
**Authoritative plan:** `docs/plan/22_fail_fast_cancellation_contract.md`  
**Plan index:** `docs/plan/00_index.md`  
**Accepted decision:** `docs/decisions/0017-failure-policy-and-cancellation-contract.md`  
**Status:** Plan 22 PRs 1–4 are landed in the working tree; PR 4 (operator
documentation + completion audit) is complete pending only the final
validation sweep. No commit created.

## Start here

Plan 22 is functionally complete: the default `continue` behavior is
preserved, fail-fast and token cancellation have typed deterministic result
semantics on both adapters, never-started cases have redacted lifecycle
evidence, no live service crosses the process boundary, public documentation
is accurate, and the plan 21 completion audit no longer overstates
cancellation or retry coverage (plan 22 §9).

The remaining open item is the final validation sweep (below), plus two
explicitly open follow-ups: optional in-case cancellation (only with
measurements; requires a new ADR) and the dormant case-retry fields.

## Working-tree constraints

- The checkout contains substantial intentional pre-existing uncommitted work.
- Never reset, clean, broadly stash, or overwrite unrelated changes.
- Never commit unless explicitly requested.
- Never modify supplied files under `data/`; write experiment evidence under
  `experiments/`.
- Preserve the mandatory `LLMClient` DiskCache and the sandbox execution
  boundary. Cancellation never deletes Hamilton cache entries, LLM cache
  entries, staging, or verified packages.

## Implemented work (plan 22 PRs 1–4)

### PR 1: ADR and typed contracts

- ADR 0017 recorded and accepted by the maintainer on 2026-09-14.
- `FailurePolicy` (`continue` / `fail_fast`) and `ExecutionPolicyConfig` in
  `Settings.execution` (env `FF_EXECUTION__FAILURE_POLICY`); the
  `config/settings.yaml` `execution:` section.
- `ExperimentResult.state` with `resolved_state` derivation, serialized
  additively into public result rows.
- Parent-process thread-safe `CancellationToken` (idempotent
  `cancel(reason)`, first-reason-wins, `is_cancelled()`, read-only `reason`),
  plus the redacted cancelled `FailureRecord` factory (`CaseCancelled`).

### PR 2: Sequential fail-fast and lifecycle

- `ExperimentalPlatform.run(failure_policy=..., cancellation_token=...)`
  resolved run-scoped into a deep-copied Settings snapshot — cached/global
  Settings are never mutated; the effective policy is recorded in tracker
  provenance.
- Sequential scheduling (plan 22 §2.4): token checked before each case;
  `fail_fast` stops scheduling after the first terminal failed result
  (`resolved_state == FAILED`). Never-started cases drain in matrix order
  into typed cancelled results (parent-allocated identity, no stage packages,
  no fabricated scores, redacted record in `error`).
- One redacted `case_cancelled` lifecycle event per unstarted case; tracker
  effects stay exactly-once for cases that actually started; cancellation
  touches no cache/staging/package bytes (digest-snapshot integration test).

### PR 3: Bounded process scheduling

- `ProcessPoolExecutionAdapter` no longer eagerly submits the full matrix:
  bounded window of at most `max_workers` futures, refilled only while a
  parent-side `stop_on_result` verdict and the token permit. Queued futures
  are cancelled with `Future.cancel()`; the never-submitted tail becomes
  typed CANCELLED rows; already-running workers finish and keep their real
  results. Original order/cardinality are preserved via index-keyed
  reassembly; the executor is shut down and joined on every exit path.
- KeyboardInterrupt hygiene (plan 22 §2.6) on both paths: stop submission,
  cancel pending futures, join/close resources, journal redacted cancellation
  for proven-never-started cases, re-raise — never swallowed, never converted
  into a failed experiment.
- Sequential/process semantic parity for deterministic schedules
  (`tests/integration/test_hamilton_execution_parity.py`), Linux-default and
  explicit `spawn` contexts.

### PR 4: Operator documentation and completion audit

- `docs/operations.md`: new "Execution failure policy and cancellation"
  section (settings/env/per-run forms, cooperative token example, bounded
  window vs. hard termination, interrupt behavior, `state` field, resume
  note, explicit retry gap) + `FF_EXECUTION__FAILURE_POLICY` config row.
- `docs/api_reference.md`: `run()` keyword params with precedence and
  no-mutation guarantee, `ExperimentResult.state`/`resolved_state`,
  `CancellationToken` API, cancelled `FailureRecord` shape,
  `ProcessPoolExecutionAdapter` bounded-window note.
- `docs/migration_guide.md`: PR-1 subsection extended to implemented status
  (per-run override, token, additive `state`, no migration action).
- `README.md` / `docs/index.md`: brief failure-policy + cancellation mention
  pointing to operations docs; `config/settings.yaml` comment enriched.
- Completion audit: plan 21 claims audited literally; retry/cancellation
  overstatements corrected with `[Audited 2026-09-14: ...]` notes (see the
  audit table in `REPORT_LOG.md`); plan index and `.planning/STATE.md`
  refreshed. The `ResourceConfig.max_attempts` retry gap remains explicit
  everywhere.

## Important files

- Contracts/config: `src/feature_forge/config.py`,
  `src/feature_forge/contracts/orchestration.py`,
  `src/feature_forge/contracts/stages.py`, `config/settings.yaml`
- Scheduling: `src/feature_forge/experiment/execution.py`
  (`CancellationToken`, `cancelled_failure_record`, bounded
  `ProcessPoolExecutionAdapter`), `src/feature_forge/platform.py`
  (run-scoped policy resolution, cancelled-result factories, lifecycle
  journaling)
- Tests: `tests/unit/test_fail_fast_sequential.py`,
  `tests/unit/test_fail_fast_process.py`,
  `tests/integration/test_hamilton_execution_parity.py`,
  `tests/integration/test_platform_e2e.py` (fail-fast resume/redaction),
  `tests/unit/test_platform.py` (run() plumbing)
- Docs: `docs/operations.md`, `docs/api_reference.md`,
  `docs/migration_guide.md`, `README.md`, `docs/index.md`
- Status/evidence: `REPORT_LOG.md`, `.planning/STATE.md`,
  `docs/plan/00_index.md`, `docs/plan/22_fail_fast_cancellation_contract.md`,
  `docs/decisions/0017-failure-policy-and-cancellation-contract.md`

## Validation evidence

As reported by the PR workers (see `REPORT_LOG.md` entries):

- PR 3 worker reported the full suite as 1007 passed / 9 skipped; the
  post-implementation audit re-ran the suite and corrected the true count to
  **998 passed / 9 skipped** (the PR 3 worker's arithmetic included the 9
  skips twice). After the audit's own fixes (mypy-strict test typing, fork
  deprecation-warning filters, the §6 test 4 process-path mid-run token test)
  the final validated count is **1000 passed / 9 skipped** (skips are the
  expected optional-XGBoost set) under explicit BLAS/OpenMP single-thread
  limits, plus ruff, ruff format, `mypy src`, `mypy tests`, repo hygiene,
  docs references, generated-doc freshness,
  strict MkDocs, `uv lock --check`, and `git diff --check`.
- PR 2 worker: sequential fail-fast/cancellation suites and the real-chain
  fail-fast resume/redaction integration test passed.
- PR 4 worker: docs-focused checks re-run after the documentation edits —
  `scripts/check_docs_references.py`, `scripts/check_repo_hygiene.py`,
  `mkdocs build --strict`, `scripts/generate_stage_dag_docs.py --check`,
  and `tests/unit/test_documentation.py`.

## Confirmed decisions

- ADR 0017 accepted 2026-09-14: `continue` is the default; `fail_fast`
  triggers only on a terminal failed case result; feature-level rejections,
  warnings, cache misses, and partial successes never trigger it.
- Cancellation is cooperative at case boundaries. Up to `max_workers`
  already-running cases may complete after a stop condition — bounded,
  documented behavior, never hard termination. Already-running work is
  never killed or relabelled.
- Result cardinality is exact: one ordered row per requested case with the
  additive `state` field; cancelled rows carry no scores/stages.
- The token is parent-process-only and never serialized; process payloads
  stay free of tokens and live services.
- KeyboardInterrupt is re-raised after cleanup, never swallowed.
- Resume from verified prefixes produced before cancellation works.
- Case-level retry is NOT implemented: `ResourceConfig.max_attempts` stays
  dormant, and fail-fast evaluates only after terminal results. Docs and
  plans must not claim retry coverage.

## Suggested next steps

1. Run the final validation sweep and record results in `REPORT_LOG.md`:
   ```bash
   uv sync --all-groups --extra intel
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   NUMEXPR_NUM_THREADS=1 uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src && uv run mypy tests
   uv run python scripts/check_repo_hygiene.py
   uv run python scripts/check_docs_references.py
   uv run python scripts/generate_stage_dag_docs.py --check
   uv run mkdocs build --strict
   uv lock --check && git diff --check
   ```
2. Optional (requires measurements + a new ADR per plan 22 §7): revisit
   in-case cooperative cancellation only if case-boundary cancellation proves
   operationally inadequate.
3. Open gap: implement case-level retry (`ResourceConfig.max_attempts` is
   dormant) in its own plan if an experiment demonstrates the need.
4. Plan 22 §8 asks for re-validation of the process-scheduling slice on
   Python 3.11, 3.12, and 3.13 and both the Linux-default and explicit
   `spawn` contexts before calling the program fully closed; PR 3 covered
   both contexts on the default interpreter.

## Suggested start for the next thread

- `docs/plan/00_index.md` (Current Phase / Next Steps)
- `docs/plan/22_fail_fast_cancellation_contract.md` (§9 definition of complete)
- `docs/decisions/0017-failure-policy-and-cancellation-contract.md`
- `docs/operations.md` ("Execution failure policy and cancellation")
- `REPORT_LOG.md` (top entries, including the PR 4 audit table)
- `.planning/STATE.md` (Active execution-runtime sequence)
