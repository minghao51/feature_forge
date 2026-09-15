# ADR 0005: Prompt colocation with Pydantic-validated rendering

- **Status:** Accepted (rendering convention superseded by
  [ADR 0008](0008-jinja2-prompt-templating-full-externalization.md);
  colocation and Pydantic validation stand)
- **Date:** 2026-08-29
- **Related:** `docs/plan/16_prompt_colocation_pydantic.md`;
  `docs/plan/13_methods_restructure.md`; ADR 0004

## Context

Prompt text was split between `config/prompts/*.yaml` (eight
MALMAS-specific files in a generically named directory) and hardcoded
f-strings inside `method.py` files. Nothing validated the variables
injected into templates — an `n_features: -1` silently produced a broken
prompt that failed at LLM call time. Prompt text also drifted from the
code that used it. Per `AGENTS.md`, prompts must stay lean, versioned,
and testable.

This record was written retroactively on 2026-08-29, after
implementation; see Related for the originating plan documents.

## Decision

1. **Prompts are colocated with their method.** Each method owns a
   `prompts/` directory inside its package
   (`methods/malmas/prompts/`, `methods/malmus/prompts/`,
   `methods/llmfe/prompts/`, `methods/caafe/prompts/`) holding YAML
   templates (`system` + optional `description`). The shared
   `config/prompts/` directory no longer exists.
2. **Two-layer Pydantic validation.** A `Prompt` model validates the
   YAML structure (non-empty `system`); per-template `*Params` models
   (e.g. `MalmusSingleShotParams`) validate runtime injection variables
   with typed constraints (`n_features: Field(ge=1)`,
   `task: Literal[...]`) and render via `str.format()` — no templating
   dependency.
3. **One shared loader, not per-method copies.** `Prompt` and
   `PromptRegistry` (lazy, cached) live in `methods/_prompting.py` and
   are instantiated per method package — a deliberate deviation from the
   plan's duplicated per-method registries to avoid ~30 lines of copy
   drift.
4. **Tightly coupled prompts stay in Python.** Memory summarization
   prompts (`SummarizeAgentParams`, `SummarizeGlobalParams`) embed their
   text in the Params model's `render_*()` methods rather than YAML;
   YAML externalization is reserved for call-site-agnostic templates.

## Consequences

- Invalid prompt parameters now fail as Pydantic validation errors at
  construction, before any LLM call; prompts are reviewable and
  diffable next to the code that renders them.
- Prompt text changes do not invalidate the LLM response cache key
  automatically — cache-invalidation on prompt edits must be handled
  deliberately (ADR 0001 rule 2).
  **Correction (2026-08-30):** the statement above is wrong. Cache keys
  hash the full rendered `messages`, so prompt edits change the SHA-256
  key automatically (verified empirically); edits orphan old entries but
  never serve stale responses. See ADR 0008 for the corrected
  consequence.
- The `{placeholder}`/`str.format()` convention has no escape hatch
  for literal braces or conditional blocks; richer templating needs a
  new decision.
- A new method must follow the layout (YAML + Params model) to keep the
  pattern uniform; the shared `_prompting.py` seam is where any future
  cross-method prompt change lands.
