# ADR 0004: Keep MkDocs Material as the canonical documentation system

- Status: Accepted
- Date: 2026-07-13

## Decision

The existing MkDocs Material site remains canonical for architecture,
contracts, operational guidance, and generated pipeline documentation. An
Astro catalog is deferred until an interactive experiment catalog has a
demonstrated need and a stable snapshot schema.

## Consequences

- Architecture decisions live beside the existing implementation plans.
- Documentation builds remain static and do not require a second frontend
  stack during the core refactor.
- Any future browser catalog must consume allowlisted derived snapshots rather
  than raw artifact files.
