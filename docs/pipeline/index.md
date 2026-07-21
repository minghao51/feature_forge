# Medallion pipeline

Feature Forge uses a two-level execution model. `ExperimentalPlatform` remains the outer
control plane: it expands cases, resolves resources, isolates failures, records lifecycle
events, and decides whether verified layers can be reused. Hamilton is the inner dataflow for
one case. Third-party methods do not need to import Hamilton.

The durable path is opt-in with `artifact_policy="layer_boundaries"`; the legacy imperative
path remains compatible. Hamilton and DuckDB are optional and installed by the `pipeline`
extra.

## Read next

- [Architecture and control boundaries](architecture.md)
- [Layers, identity, and invalidation](layers.md)
- [Contracts and durable evidence](contracts.md)
- [Method and plugin authors](method-authors.md)
- [Generated DAGs, schemas, checks, catalog, and CLI](../generated/index.md)

The [Silver](../silver_dataflow.md), [Gold](../gold_dataflow.md), and
[Platinum](../platinum_evidence.md) pages describe executable details for each layer.
