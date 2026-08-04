# Architecture decision records

These records capture the decisions that make the Hamilton and medallion-lite
refactor incremental. The existing `ExperimentalPlatform` API remains the
outer control plane while the durable artifact contracts are introduced behind
an opt-in storage boundary.

- [Hamilton boundary](0001-hamilton-boundary.md)
- [Medallion-lite semantics](0002-medallion-lite.md)
- [Artifact formats and atomic commit](0003-artifact-commit.md)
- [Canonical documentation system](0004-documentation-system.md)
