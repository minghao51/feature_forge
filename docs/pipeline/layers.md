# Layers, identity, and invalidation

| Layer | Durable responsibility | Reuse identity changes when | Does not change for |
|---|---|---|---|
| Bronze | Reproducible source input | source identity/checksum or source contract | method, prompt, model, metric |
| Silver | Canonical features/target, row IDs, folds | Bronze, target/task, canonicalization, split policy/seed | method, prompt, model, metric |
| Gold | Generated code/specifications, candidate and accepted feature products | Silver, method/version/config, prompt bundle, code/spec contract, output-changing validation policy | evaluation model and reporting-only options |
| Platinum | Fold predictions/metrics, uncertainty, final decisions | Gold, model/version/config, metric, folds, evaluation/selection/uncertainty policy | tracker and presentation settings |

Layer fingerprints are versioned independently. A changed downstream policy must not force
upstream regeneration. Resume accepts only a verified manifest with the expected fingerprint,
schema, and upstream lineage; invalid or incomplete packages are never treated as cache hits.

Silver row IDs and persisted folds become authoritative. Baseline and enhanced evaluation must
use the same fold records. Gold replay loads stored code and specifications and must not
initialize a provider. Platinum aggregates must reconstruct from stored fold evidence.

## Source policy

Stable local inputs may use reference-plus-checksum Bronze evidence. Mutable, remote,
generated, or difficult-to-retrieve inputs should be snapshotted. Existing
`.feature_forge_artifacts` descriptive storage is not silently reclassified as verified
medallion evidence because it has no committed manifest lineage.
