# PR 6 Handoff: Catalog, Verification CLI, and Observability

**Status:** Implemented and verified 2026-07-15
**Depends on:** Stable manifests, fingerprints, lifecycle events, and resume
**Index:** [`2026-07-14-medallion-pr-index.md`](2026-07-14-medallion-pr-index.md)

## Objective

Add a rebuildable local catalog, read-only verification commands, and lifecycle
telemetry without making DuckDB or an external tracker the source of truth.

## Scope

### DuckDB catalog

- Define a versioned schema for runs, stages, manifests, artifacts, checks,
  features, decisions, fold metrics, and lineage edges.
- Index only committed, verified manifests.
- Apply one package/catalog update transactionally.
- Store manifest/artifact references rather than copying large payloads.
- Keep the catalog derived: deleting it and rebuilding from artifact roots must
reproduce the same logical index.

## Confirmed PR 5 inputs

- `RunEvent` schema version 1 carries event/run/case identity, timestamp,
  state, stage/layer, executed/reused/skipped disposition, structured failure,
  manifest reference, fingerprint, attempt, and secret-free details.
- `LocalRunRepository` stores authoritative append-only `events.jsonl` and an
  atomic `state.json` snapshot under `control/runs/<run-id>/`.
- The platform currently emits case planned/running/terminal events. PR 6 must
  add Hamilton node timing/cache events and catalog indexing events without
  changing the event source of truth.
- `LocalArtifactStore.find_reusable()` and resume planning accept only verified
  committed packages with exact fingerprint and upstream lineage. Catalog
  acceleration must preserve those semantics and filesystem rebuild parity.
- Conservative recovery reports removed and retained staging/lock paths; PR 6
  should surface these as operational findings, never perform implicit
  destructive cleanup during read-only verification.
- Detect missing, corrupt, duplicate, incompatible, and orphaned records.

### Rebuild and reconciliation

- Scan allowed layer roots deterministically.
- Verify packages before indexing.
- Report, but do not silently delete, incomplete staging or corrupt packages.
- Support incremental indexing and full rebuild.
- Record catalog schema/version and migration policy.
- Add stale-run detection from lifecycle state and timestamps.

### CLI

Provide stable commands equivalent to:

```text
feature-forge verify run <run-id>
feature-forge verify artifact <uri>
feature-forge verify catalog
feature-forge catalog rebuild
feature-forge catalog status
feature-forge run plan ...
```

Verification commands must be read-only. Rebuild/catalog mutations must be
explicit. Exit codes and machine-readable output must be documented.

### Hamilton and tracker observability

- Add a Hamilton lifecycle adapter recording node/stage timing, state, cache
  outcome, and failure context.
- Include run, case, stage, layer, and node identifiers in errors/logs.
- Keep artifact manifests authoritative; trackers receive metrics and artifact
  references only.
- Apply required/optional tracker failure policy.
- Prevent secret or raw sensitive artifact content from entering telemetry.

## Expected file areas

- catalog contracts/schema and repository implementation
- CLI entrypoints
- `src/feature_forge/dataflows/` lifecycle adapter
- `src/feature_forge/experiment/tracker.py` and backends
- configuration for catalog path/policy
- verification/rebuild tests and operational docs

## Tests to add

- Empty catalog rebuild equals incremental indexing of the same roots.
- Rebuild is idempotent and deterministically ordered.
- Corrupt/missing packages are reported and not indexed as healthy.
- Verification commands perform no writes.
- CLI exit codes distinguish valid, invalid, missing, and operational failure.
- Catalog transactions do not expose partial package indexing.
- Node failures identify node/layer/case/run without secrets.
- Optional tracker failure does not invalidate artifact publication; required
  tracker failure follows explicit policy.
- Catalog can be deleted without losing authoritative evidence.

## Acceptance criteria

- Rebuild produces the same logical catalog as normal indexing.
- Every catalog row traces back to a verified manifest.
- Read-only verification does not mutate artifacts, catalog, caches, or state.
- Catalog detects missing/corrupt artifacts and stale runs.
- Trackers cannot become the sole location of a result or artifact.
- Default installations do not import DuckDB until catalog functionality is
  requested through the pipeline extra.

## Explicitly out of scope

- Remote/multi-user catalog service.
- Catalog as a write authority.
- Browser UI or Astro.
- Automated destructive cleanup.

## Verification

```bash
uv run ruff check .
uv run mypy src
uv run --extra pipeline pytest tests/unit/test_catalog.py tests/unit/test_verification_cli.py -q
uv run --extra pipeline pytest -m "not llm" -q
uv run python scripts/run_pip_audit.py
uv run mkdocs build --strict
git diff --check
```

## Completion handoff requirements

Report schema/version, transaction boundaries, rebuild equivalence evidence,
CLI surface/exit codes, telemetry redaction, tracker failure policy, and the
exact generated references PR 7 must document.

## Completion record

- Catalog schema version 1 is manifest-centric and contains `manifests`,
  `runs`, `datasets`, `stages`, `artifacts`, `validation_checks`, `features`,
  `feature_decisions`, `fold_metrics`, `lineage_edges`, and `run_events`.
  Evidence tables retain the verified manifest SHA-256; large payloads remain
  in authoritative packages.
- `LocalCatalog.index_package()` verifies the package and required upstream
  references before opening a transaction. Replacing a manifest and all of its
  dependent rows is one DuckDB transaction. Fault injection after inserts
  proves rollback leaves no partial package rows.
- Full rebuild scans layer/run candidates deterministically, reports corrupt or
  incomplete packages without indexing them, builds a sibling database under
  an exclusive rebuild lock, closes and fsyncs it, then atomically replaces the
  old catalog. Incremental and rebuilt logical dumps have identical stable
  digests; repeated rebuild is idempotent. Version 1 uses rebuild-only migration
  and refuses unknown catalog versions.
- `feature-forge` now exposes `verify run`, `verify artifact`, `verify catalog`,
  `catalog rebuild`, `catalog status`, and `run plan`. JSON output is one
  schema-versioned object. Stable exits are `0` success, `10` invalid, `11`
  missing, `12` incompatible, `20` operational, and `64` usage.
- Verification and status use read-only DuckDB connections and authoritative
  filesystem/lifecycle reads. Tests snapshot bytes and mtimes and prove that
  missing catalog paths are not created. Reconciliation reports corrupt,
  missing, undeclared, unindexed, stale, incomplete-staging, and stale-run
  findings without cleanup.
- CLI and dataflow public exports are lazy. Default import tests prove that
  neither DuckDB nor Hamilton is imported by `feature_forge.cli`; catalog and
  Hamilton attachment fail with actionable `uv sync --extra pipeline` guidance
  when the optional dependencies are absent.
- The Hamilton adapter emits bounded run/case/Hamilton-run/task/node timing and
  state with allowlisted tags. It never emits node inputs/results. Errors are
  redacted and truncated. Cache hit/miss state is read from Hamilton 1.90 cache
  events rather than inferred from duration.
- Trackers receive scalar metrics and durable artifact references through
  `log_artifact_reference()`. Optional tracker degradation yields `PARTIAL`;
  required degradation yields a run-level `FAILED`. In either case committed
  verified evidence remains valid and reusable.
- Focused PR 6, tracker, scheduler/resume, and Silver/Gold/Platinum tests pass.
  Ruff passes; strict mypy passes across 110 source files; lock, diff, repository
  hygiene, docs-reference, strict MkDocs, and dependency-audit gates pass. The
  audit reports no known vulnerabilities and skips only the editable local
  distribution.
- After the delta remediation stopped, one authoritative full non-LLM lane
  passed 857 tests with 0 failures and 0 errors at 88% coverage. Ruff,
  strict mypy across 110 source files, lockfile consistency, diff whitespace,
  repository hygiene, docs references, strict MkDocs, and dependency audit all
  pass. The audit reports no known vulnerabilities and skips only the editable
  local distribution.

### PR 7 generated-reference inputs

PR 7 must freshness-check the live DDL in `storage/catalog.py`, CLI help and
exit definitions in `cli.py` and `contracts/catalog.py`, telemetry tag/redaction
policy in `observability/hamilton_adapter.py`, tracker failure policy in
`config.py`, and the operational baseline in `docs/catalog_verification.md`.

### Adversarial remediation (2026-07-15)

- Artifact verification now constructs a validated `ArtifactRef` and calls the
  artifact store resolver. Traversal, metadata, undeclared, and missing paths
  cannot bypass manifest authorization.
- Indexing and run/catalog verification invoke layer semantic loaders. Gold
  also loads and matches its authoritative Silver lineage; Platinum reconstructs
  its Silver/Gold chain.
- Catalog verification reconstructs and compares every logical table, detecting
  dependent-row tampering even when the manifest row is unchanged.
- Staleness means an aged `case_running` without a later `case_terminal`; later
  stage/node events cannot hide a crashed case.
- The production driver binds redacted Hamilton node/cache telemetry to the PR 5
  run journal. Redaction includes authorization/bearer values, provider keys,
  tokens, generic secrets, errors, and allowlisted tag values.
- Lifecycle indexing retains disposition, attempt, structured failure,
  fingerprint, and details. Incremental/event writers and rebuild share one
  lock; rebuild fsyncs the temporary database before atomic replacement.
- Catalog verification reports a malformed lifecycle journal as a stable
  `invalid_journal` result and does not reparse it during reconciliation.
  Telemetry also redacts standalone provider-key values matching the execution
  failure boundary's `sk-`, `rk-`, and `pk-` credential pattern.
