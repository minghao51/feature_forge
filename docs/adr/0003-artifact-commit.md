# ADR 0003: Use typed artifacts with atomic package commits

- Status: Accepted
- Date: 2026-07-13

## Decision

Durable packages use Parquet for tabular data, JSON for manifests, compressed
JSONL for record streams, and human-diffable Python files for generated code.
Each package is staged below its layer directory and validated against typed
descriptors. The manifest and `_SUCCESS` marker are written last inside the
staging directory, after which the complete directory is atomically renamed
into its run-scoped namespace. Readers therefore never observe a published
package without its completion marker.

Manifests contain hashes, sizes, paths, and schema metadata. A package without
`_SUCCESS` is incomplete and must not be reused. Existing in-memory and
descriptive-key artifact storage remains available; supplying a `run_id` opts
that adapter into a run-scoped directory.

## Consequences

- Concurrent run IDs cannot overwrite one another.
- Corruption is detectable without rerunning an experiment.
- A crash before the completion marker leaves a non-reusable package.
- Catalogs can later be rebuilt from manifests rather than becoming the source
  of truth.
