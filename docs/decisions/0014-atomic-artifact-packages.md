# ADR 0014: Atomic artifact packages

- **Status:** Accepted
- **Date:** 2026-09-09
- **Related:** ADR 0013, `docs/plan/21_hamilton_default_execution_handoff.md`

## Context

Write-once experiment evidence must never be observed half-published, even when
execution fails or multiple workers operate concurrently.

## Decision

Artifact stores publish each attempt into a staging directory, validate its
contents and SHA-256 descriptors, then atomically publish a completion marker
with the package. Attempt namespaces are write-once. Reuse selects only a
verified package with the requested fingerprint and exact ordered lineage.

## Consequences

Readers either see no package or a complete committed package. Recovery can
discard incomplete staging data without rewriting verified evidence; migration
of legacy roots is a separate operator-controlled action.
