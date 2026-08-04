# Migrating to the medallion runtime

The migration is a strangler, not a rewrite. Existing `ExperimentalPlatform.run()` behavior
continues while the manifest-backed path is opt-in.

| Existing surface | Medallion surface | Compatibility and rollback |
|---|---|---|
| Imperative case execution | Hamilton inner case dataflow | Switch the profile/policy back; Hamilton is not the outer scheduler |
| Nine legacy `ExperimentResult` fields | Optional run, layer, manifest, uncertainty, and decision fields | Reporter accepts mixed old/new shapes; legacy field order remains |
| `DataFrameStorage` and descriptive keys | Run/layer package store with manifest and `_SUCCESS` | Legacy directories remain unverified; do not reinterpret them |
| `generated_scripts` and `feature_metadata` | Gold adapter and optional staged batches | Existing plugins keep working and never import Hamilton |
| Recomputed folds/summary-only metrics | Persisted Silver folds and Platinum evidence | Keep compatibility selection while parity is measured |
| Tracker-centric inspection | Manifest authority plus derived catalog | Delete/rebuild the catalog; tracker loss does not lose evidence |

## Recommended sequence

1. Install the `pipeline` extra and keep legacy execution as default.
2. Produce deterministic Silver evidence and verify row/fold identity.
3. Persist and replay Gold without a provider.
4. Reconstruct Platinum aggregates from fold records and measure parity.
5. Enable failure-aware scheduling and verified resume.
6. Rebuild and verify the derived catalog.
7. Move the default only after parity and one documented deprecation window.

Each stage can roll back independently by selecting legacy execution or the last verified
upstream manifest. Current artifact readers accept schema version 1 and reject unknown or
future versions. Feature Forge ships no automatic or in-place persisted-artifact migration.
Upgrade evidence by reading it with a compatible release/tool and writing a new package/run
namespace, or rebuild it from the authoritative upstream source. Never mutate historical
manifests in place. Catalog schema migration is rebuild-only.

The optional Astro catalog remains deferred. MkDocs plus generated static references are
canonical until at least two real interaction requirements and every owner/security/hosting
governance condition are satisfied.
