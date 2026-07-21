# Catalog, verification, and lifecycle telemetry

Feature Forge keeps durable manifests and `_SUCCESS` packages authoritative.
The DuckDB catalog is a disposable local index: deleting it does not delete or
invalidate experiment evidence.

## Installation and configuration

Catalog commands require the optional pipeline dependencies:

```bash
uv sync --extra pipeline
```

The default catalog is
`.feature_forge_artifacts/control/catalog/catalog.duckdb`. Override nested
settings with `FF_CATALOG__PATH` and `FF_CATALOG__STALE_RUN_SECONDS`, or pass
`--catalog-path`, `--artifact-root`, and `--lifecycle-root` to the CLI.
Importing Feature Forge or its CLI does not import DuckDB or Hamilton.

## Catalog schema and authority

Catalog schema version 1 contains these manifest-derived tables:

- `manifests`, `runs`, and `datasets` for package identity;
- `stages`, `artifacts`, and `validation_checks` for package evidence;
- `features`, `feature_decisions`, and `fold_metrics` for normalized Gold and
  Platinum records;
- `lineage_edges` for exact upstream manifest references;
- `run_events` for the derived lifecycle-event index.

Every domain row includes or is keyed by a verified manifest SHA-256. Large
frames, predictions, generated code, and other artifact payloads remain in the
artifact package; the catalog stores descriptors or compact record summaries.
Unknown catalog schema versions are refused. Version 1 uses rebuild-only
migration: build a sibling database from evidence, close it, and atomically
replace the previous catalog only after the rebuild succeeds.

Incremental indexing verifies and normalizes a package before opening its
transaction. Replacing the manifest and all dependent rows is one DuckDB
transaction, so readers cannot observe a partially indexed package.

## Commands

```bash
uv run feature-forge verify run <run-id>
uv run feature-forge verify artifact bronze:<run-id>:records.jsonl.gz
uv run feature-forge verify catalog
uv run --extra pipeline feature-forge catalog rebuild
uv run feature-forge catalog status
uv run feature-forge run plan --dataset titanic --method openfe
```

Use `--format json` before the command for one compact, schema-versioned JSON
object on stdout. The stable exit codes are:

| Code | Meaning |
|---:|---|
| 0 | valid or successful |
| 10 | invalid or corrupt evidence |
| 11 | requested evidence is missing |
| 12 | incompatible catalog/schema |
| 20 | operational failure |
| 64 | command usage error |

`verify run`, `verify artifact`, `verify catalog`, and `catalog status` are
read-only. They do not create missing databases or directories and do not
initialize providers, trackers, Hamilton drivers, or caches. `catalog rebuild`
is the explicit mutation command. It reports corrupt packages and non-package,
dot-prefixed candidates generically as incomplete candidates, but does not
classify stale-lock ownership or delete anything. Only explicit artifact-store
recovery applies owner-age and dead-process checks to staging directories and
commit locks.

Artifact verification accepts only a normalized `ArtifactRef` declared by the
verified manifest, then resolves it through the artifact store. Package
metadata, undeclared paths, and traversal paths are not artifact references.
Run/catalog verification and indexing additionally invoke the Silver, Gold,
and Platinum semantic loaders with exact upstream lineage reconstruction;
hashes alone do not authorize evidence.

`run plan` calls the PR 5 dry-run path. A `layer_boundaries` plan also requires
`--fingerprints-json` containing the resolved per-run layer fingerprints.

## Reconciliation and stale runs

Catalog verification compares committed packages with every logical catalog
table, including dependent feature, decision, metric, check, lineage, and typed
lifecycle rows. It reports
unindexed packages, stale rows, corrupt or missing files, undeclared files,
incomplete staging, incompatible schemas, and stale lifecycle runs. A run is
stale when a `case_running` event has aged beyond the configured threshold and
no later `case_terminal` exists. Later node or stage events do not mask a stale
case. Read-only checks never perform recovery or cleanup.

## Telemetry and trackers

The production Hamilton driver attaches the lifecycle adapter to the PR 5 local
run journal and flushes Hamilton cache events after execution/materialization.
The adapter records run/case/Hamilton-run/task/node identity,
allowlisted tags, timing, state, cache outcome, and a bounded redacted failure.
It never records node inputs, results, dataframes, generated code, environment
configuration, or provider payloads. Authorization values, provider-specific
keys (including standalone `sk-`, `rk-`, and `pk-` values), bearer credentials,
errors, and allowlisted tag values are redacted.
Cache outcomes come from Hamilton's cache
event records rather than timing inference.

Trackers receive scalar metrics and durable manifest/artifact references only
on the medallion path. `tracker.failure_policy` is `optional` by default:
tracker degradation makes an otherwise successful run `PARTIAL`. With
`required`, the run-level outcome is `FAILED`. In both cases, already committed
verified artifacts remain valid and reusable; tracker state is never evidence
and never participates in fingerprints.
