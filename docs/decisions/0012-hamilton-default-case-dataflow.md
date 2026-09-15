# ADR 0012: Hamilton as the default case dataflow engine

- **Status:** Superseded by ADR 0016
- **Date:** 2026-09-09
- **Related:** ADR 0006, ADR 0016,
  `docs/plan/21_hamilton_default_execution_handoff.md`

## Context

Feature Forge needs dependency-aware execution inside each experiment case
while preserving the platform's scheduling, process, tracking, and cancellation
responsibilities. The existing execution seam must remain usable during the
migration.

## Decision

Hamilton stage drivers will own dependency ordering and durable stage execution
inside a case. `ExperimentalPlatform` remains the outer orchestrator and keeps
the existing `ExecutionBackend` seam. The foundation release exposes an
explicit `ExecutionEngine` setting and temporarily defaults to `legacy`; the
executor release changes that default atomically to `hamilton`. Legacy remains
an explicit rollback path until parity gates are satisfied.

## Consequences

Stage execution can be tested and resumed independently, while callers retain a
single platform API. Runtime services remain worker-local and are not passed
through cached pure nodes.
