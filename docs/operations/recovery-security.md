# Recovery, retention, security, backup, and rollback

## Crash recovery and retention

`LocalArtifactStore.recover_abandoned(...)` is the only cleanup surface. Operators provide
minimum staging and stale-lock ages. Cleanup removes a staging directory or commit lock only
when its owner metadata is old and its process is provably dead on the current host. Live or
ambiguous owners are retained. Committed namespaces are never removed.

Verification and catalog reconciliation report generic incomplete candidates, corrupt
packages, and stale lifecycle runs but perform no cleanup. They do not prove that a candidate
is an abandoned lock. Only `recover_abandoned` classifies staging directories and commit locks
using owner metadata, age, host, and process liveness. Feature Forge has no automatic
committed-data retention policy. Back up or remove large committed matrices only under an
explicit external policy after preserving manifests, metrics, generated code, and required
lineage.

## Tracker failure

The default tracker failure policy is `optional`: degradation makes an otherwise successful
run `PARTIAL`. `required` makes the run outcome `FAILED`. Already committed verified packages
remain valid in either case. Trackers receive only scalar metrics and durable references.

## Sensitive-data boundary

Never emit provider credentials, authorization headers, tokens, raw prompts/responses, source
rows, candidate matrices, predictions with sensitive row IDs, generated code, agent memories,
private tracker URLs, or absolute local roots into telemetry or generated documentation.
Hamilton telemetry records bounded identity/timing/state/cache metadata and allowlisted tags;
inputs and results are excluded.

## Backup and rollback

Back up the artifact roots and `control/runs` journals. The DuckDB catalog and documentation
outputs can be rebuilt. Restore packages as immutable directories with their original
`manifest.json` and `_SUCCESS`, then run catalog rebuild and verification.

Rollback boundaries are independent: keep legacy execution as the default, disable the
Hamilton profile, resume from the last verified layer, rebuild/delete the catalog, or revert a
default-profile migration. Never edit a historical manifest or reinterpret legacy descriptive
storage as verified evidence.
