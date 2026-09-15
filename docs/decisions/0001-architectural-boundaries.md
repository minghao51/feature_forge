# ADR 0001: Architectural boundaries for the experimentation platform

- **Status:** Accepted
- **Date:** 2026-08-29
- **Related:** `README.md`; `docs/plan/01_architecture.md`;
  `docs/plan/03_key_design_decisions.md`

## Context

Feature Forge is an experiment-first refactor of the MALMAS research
codebase. Its value depends on experiments staying comparable across
methods, datasets, and seeds, and on controlled LLM cost and execution
safety. These boundaries previously lived implicitly in `README.md` and
the plan docs; this record makes them the authoritative list referenced by
the decision order in `AGENTS.md`.

## Decision

1. **Plugin registry as the only extension point.** Every method is a
   first-class, independently runnable experiment unit registered through
   the plugin-based `MethodRegistry` / `BaseMethod` in
   `src/feature_forge/methods/`. New methods must conform to the registry,
   not fork the harness.
2. **Enforced LLM caching.** All LLM calls go through `LLMClient` with
   enforced DiskCache keyed by SHA-256. Bypassing the cache is not an
   option for experiments — an ADR is required to change caching behavior.
3. **Sandboxed execution only.** LLM-generated feature code is executed
   exclusively through the sandbox (AST validation + process isolation).
   No generated code path may skip either stage.
4. **Declarative experiment configuration.** The experiment matrix
   (datasets × methods × seeds × models × rounds) stays configurable via
   pydantic-settings (YAML + env + `.env`); no experiment behavior is
   hard-coded.
5. **Smallest-architecture preference.** Provider abstractions, tracking
   platforms, and infrastructure are added only after an experiment
   demonstrates the need.

## Consequences

- New methods get comparability, caching, and sandboxing for free, but
  must fit the `BaseMethod` contract.
- Accidental API cost from re-running prompts is prevented by design;
  cache-invalidation needs (e.g. prompt changes) must be handled
  deliberately.
  **Correction (2026-08-30):** keys are content-addressed over the full
  rendered `messages`, so prompt changes already invalidate cache entries
  automatically; what must be handled deliberately is orphan clean-up
  (no TTL) and prompt-version provenance. See ADR 0008.
- Experiment results carry resolved configuration, so runs are
  reproducible and auditable.
- Any change to these boundaries (e.g. introducing RAG-style retrieval,
  skipping the sandbox for a trusted provider, or a second extension
  point) now requires a superseding ADR before implementation.
