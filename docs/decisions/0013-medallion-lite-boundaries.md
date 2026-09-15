# ADR 0013: Medallion-lite stage boundaries

- **Status:** Accepted
- **Date:** 2026-09-09
- **Related:** ADR 0001, ADR 0012, `docs/plan/21_hamilton_default_execution_handoff.md`

## Context

Experiment evidence needs explicit, verifiable boundaries without introducing a
warehouse or forcing generated feature values through process boundaries.

## Decision

Cases use four independently verifiable stages: Bronze source snapshots,
Silver canonical data and folds, Gold generated-feature evidence, and Platinum
evaluation evidence. Manifest references and hashes form the durable edges.
Source data remains immutable and runtime artifacts are written below the
configured `experiments/` roots.

## Consequences

Stages can be reused only after manifest, lineage, schema, hash, and semantic
verification. A corrupt or incompatible stage causes the remaining suffix to
recompute rather than allowing a later stage to bypass it.
