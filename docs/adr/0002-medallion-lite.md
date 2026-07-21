# ADR 0002: Use medallion-lite maturity checkpoints

- Status: Accepted
- Date: 2026-07-13

## Decision

Feature Forge will use four durable maturity layers:

- Bronze records reproducible source input.
- Silver records the canonical dataset, stable row IDs, and folds.
- Gold records candidate and accepted feature products plus replay material.
- Platinum records fold-level evaluation evidence and the final summary.

The layers are trust and reuse checkpoints, not a general-purpose lakehouse.
Delta Lake, date-partitioned ingestion tables, merge semantics, and mandatory
persistence of every intermediate Hamilton value are out of scope.

## Consequences

- A layer is reusable only after its required checks pass and its completion
  marker exists.
- Cache entries remain disposable and are never treated as evidence.
- Candidate features are retained where their generation cost justifies it so
  selection policy changes do not require another provider call.
