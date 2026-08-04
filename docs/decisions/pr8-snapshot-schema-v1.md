# PR 8 public snapshot schema v1 proposal

**Status:** `PROPOSED / SECURITY APPROVAL OUTSTANDING`
**Schema:** [`schemas/pr8-public-snapshot-v1.schema.json`](schemas/pr8-public-snapshot-v1.schema.json)
**Governance gate:** GV-02 and GV-05

This is a technology-neutral proposal for a derived, read-only browser snapshot.
It is not an implementation authorization and it is not a replacement for the
rebuildable catalog or verified manifests.

## Authority and lifecycle

- The local DuckDB catalog and verified manifest packages remain authoritative.
- A future exporter must read catalog rows only; it must never read raw artifacts,
  prompts, predictions, code, environment dumps, or agent memory.
- Snapshot deletion must not remove authoritative state. Rebuilding the catalog
  must be sufficient to regenerate the snapshot.
- `schema_version` is the public format version and is currently exactly `"1"`.
  Consumers must reject future versions rather than guessing.
- Records must be sorted by their pseudonymous IDs and arrays must be emitted in
  stable order. The same approved catalog and test-only salt must produce byte-
  stable output.
- A snapshot must have a bounded total byte size and bounded per-array record
  count. The partitioning and limits require owner approval before implementation.

## Allowlist

| Area | Exported | Transform/constraint |
|---|---|---|
| Snapshot identity | `schema_version`, `snapshot_id`, catalog schema version | Version constants and digest of the approved snapshot content |
| Dataset summary | Pseudonymous dataset ID, run count, feature count | Keyed pseudonym; counts only |
| Run summary | Pseudonymous run/dataset/method/model IDs, seed, state, stage count, selected feature count | Keyed pseudonyms; aggregate counts; no timestamps or options |
| Feature decisions | Pseudonymous feature/run IDs, selected/rejected disposition, bounded reason code | Keyed pseudonyms and closed enum; no names, descriptions, code, or errors |
| Lineage | Pseudonymous parent/child run IDs, parent layer, ordinal | Closed layer enum; no paths or manifest URIs |
| Checks | Pseudonymous check ID, severity, pass/required flags | Closed severity enum; no message text |
| Metrics | Pseudonymous metric/run IDs, baseline/enhanced arm, closed metric name, scalar value | No raw fold rows, predictions, uncertainty payloads, or tracker links in v1 |

The exact JSON Schema is deny-by-default: every object uses
`additionalProperties: false`, and all top-level arrays are required even when
empty. Field names outside the schema are not implicitly safe.

## Omitted by default

The exporter must omit provider keys and tokens, authorization headers, raw
prompts/responses, raw datasets and rows, predictions, generated code, feature
names and descriptions, raw error messages, absolute paths and filesystem roots,
private URLs, tracker references, artifact URIs, environment dumps, memory
contents, arbitrary manifest metadata, free-form options, and timestamps.

Identifiers are not exported in their source form. If a future owner approves
identifiers at all, they must be keyed pseudonyms produced with a secret kept
outside the snapshot. The key must never be embedded in the browser bundle or
the repository.

## Compatibility and safety tests required before approval

1. Unknown fields are rejected at every object level.
2. Future `schema_version` and future catalog versions are rejected.
3. The same synthetic catalog and test-only key produce identical bytes across
   clean processes and different `PYTHONHASHSEED` values.
4. Negative fixtures containing provider keys, authorization headers, absolute
   paths, private URLs, prompts, raw rows, predictions, and memory are absent
   from the exported bytes.
5. Every exported record can be traced to an allowlisted catalog row and then to
   a verified manifest without exposing the source reference.
6. The exporter is explicit and read-only; verification and catalog rebuild do
   not invoke it implicitly.

The current synthetic-only fixture can be regenerated with:

```bash
uv run python scripts/emit_pr8_snapshot_spike.py \
  --output /tmp/feature-forge-pr8-snapshot-v1.json
```

This fixture is a schema-shape and byte-stability spike only. It does not read a
catalog and cannot be used as production evidence.

## Approval record

| Reviewer | Role | Decision | Date | Evidence |
|---|---|---|---|---|
| `TBD` | Security | `NOT_REVIEWED` | `TBD` | `TBD` |
| `TBD` | Catalog/data owner | `NOT_REVIEWED` | `TBD` | `TBD` |
| `TBD` | Decision owner | `NOT_REVIEWED` | `TBD` | `TBD` |

Until these rows are completed with independent dated approvals, GV-02 and
GV-05 remain `NOT_MET`, and no browser export or frontend implementation is
authorized.
