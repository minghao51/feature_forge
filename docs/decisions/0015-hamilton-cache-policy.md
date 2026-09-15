# ADR 0015: Hamilton cache policy

- **Status:** Accepted
- **Date:** 2026-09-09
- **Related:** ADR 0001, ADR 0012, `docs/plan/21_hamilton_default_execution_handoff.md`

## Context

Hamilton caching can accelerate pure computation, but it is not experiment
evidence and must not weaken Feature Forge's mandatory LLM response cache or
artifact verification.

## Decision

Hamilton caching is enabled by default in every execution profile and uses a
persistent path below the resolved artifact root unless explicitly overridden.
Pure nodes use Hamilton's default cache behavior. Source reads, mutable method
and sandbox boundaries, per-attempt materialization, publication, and external
side effects recompute. Cache hits never establish successful stage completion;
only a verified artifact package does. `enabled=false` is an explicit
diagnostic override.

## Consequences

Cache deletion or corruption is safe and falls back to recomputation. Cache
telemetry must be metadata-only and never contain prompts, inputs, outputs, or
secrets. The LLM `DiskCache` remains mandatory and independently configured.
