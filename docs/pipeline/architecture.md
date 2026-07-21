# Architecture and control boundaries

```text
ExperimentalPlatform / scheduler / lifecycle journal
  └── one case
      └── Hamilton documentation, CI, development, production, or replay profile
          ├── Bronze: source identity or snapshot
          ├── Silver: canonical rows, target, IDs, folds
          ├── Gold: generated code, candidates, validation decisions
          └── Platinum: fold evidence, uncertainty, final decisions
```

The outer scheduler owns case ordering, `fail_fast`, process isolation, resource ceilings,
resume planning, and terminal outcomes. Hamilton owns dependency ordering inside a case. It is
not a distributed scheduler and is not required by the legacy API.

The local run journal under `control/runs/<run-id>/` is append-only operational evidence.
Committed layer packages are authoritative data evidence. The DuckDB catalog and Hamilton
cache are derived and disposable. Trackers receive scalar metrics and durable references; a
tracker is never the only location of a result.

## Side-effect boundaries

The `documentation` and `ci` profiles forbid network access and persistence. `replay` forbids
provider construction and completion. Publication is a write-once staging-to-package rename;
only a package whose manifest hash is stored in `_SUCCESS` is committed.

See the [end-to-end generated DAG](../generated/case-dag.png) and the
[live node/tag inventory](../generated/node-inventory.md).
