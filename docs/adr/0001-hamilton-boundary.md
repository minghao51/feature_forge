# ADR 0001: Hamilton is the inner case dataflow

- Status: Accepted
- Date: 2026-07-13

## Decision

Hamilton will own the inspectable dataflow for one experiment case. The
existing `ExperimentalPlatform`, case executor, and execution backends remain
the outer control plane for matrix expansion, scheduling, process boundaries,
trackers, retries, and cancellation.

Generic dataflow nodes will call stable method and evaluation interfaces. They
will not absorb MALMAS internals, and each generated feature will remain a
record rather than becoming a separate Hamilton node.

## Consequences

- Dataset preparation, feature generation, verification, selection, and
  evaluation can acquire explicit lineage and stage contracts.
- Existing methods do not need to depend on Hamilton.
- Hamilton remains optional during the parity phase.
- The legacy execution profile can continue while the new profile is proven.
