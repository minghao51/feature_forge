# ADR 0008: Jinja2 prompt templating and full prompt externalization

- **Status:** Accepted
- **Date:** 2026-08-30
- **Related:** ADR 0005 (supersedes its point 2 rendering convention and
  point 4 Python-embedded exception for memory prompts); ADR 0004

## Context

ADR 0005 colocated prompts as YAML next to each method and validated
injection variables with Pydantic `*Params` models, but rendering used
`str.format()`. That convention has no escape hatch for literal braces
(prompt text containing JSON examples must avoid `{}`), no conditionals
or loops inside templates, and it scattered rendering logic as a
`render()` method on every Params model. Separately, the MALMAS memory
summarization prompts were exempted (ADR 0005 point 4) and lived as
f-strings inside `memory/prompts.py`, so not all LLM prompt text was
reviewable/diffable YAML, and not every prompt had a Pydantic domain
model.

## Decision

1. **Jinja2 is the single prompt templating engine.** `PromptRegistry`
   owns one `jinja2.Environment` per method package with
   `StrictUndefined` (a missing variable fails at render time, preserving
   ADR 0005's fail-fast property), `autoescape=False` (prompts are plain
   text), and `keep_trailing_newline=True`.
2. **Params models are pure domain schemas.** `*Params` classes subclass
   the new `PromptParams` base (`methods/_prompting.py`) and only declare
   + validate variables; `render()` methods are removed. Templates
   render through the registry: `registry.render(name, params)` for a
   system prompt and `registry.render_messages(name, params)` for a
   system+user pair.
3. **All LLM prompt text lives in YAML.** The `Prompt` model gains an
   optional `user:` template (default empty) alongside `system`. The
   MALMAS memory summarization prompts move from
   `memory/prompts.py` f-strings to
   `methods/malmas/prompts/summarize_agent.v1.yaml` and
   `summarize_global.v1.yaml`. YAML placeholders use Jinja syntax
   (`{{ var }}`).
4. **Code-coupled user prompts stay in Python.** The MALMAS agents'
   `_build_user_prompt()` assembles dataset-derived JSON blobs and
   remains a Python builder (per ADR 0005's original rationale); its
   system prompt still comes from the registry. This exception is now
   explicit rather than implicit.

## Consequences

- Every LLM prompt's static text is externalized, reviewable, and
  diffable; every dynamic variable is declared on a Pydantic domain
  model with typed constraints.
- Literal braces in prompt text are safe; templates can use Jinja
  conditionals/loops when a measured need arises, without further
  decisions.
- Prompt text changes still do not invalidate LLM cache keys
  automatically (ADR 0001 rule 2 unchanged).
  **Correction (2026-08-30):** the statement above is wrong and was
  propagated from ADR 0005. Cache keys hash the full rendered
  `messages` payload (`compute_cache_key`), so prompt edits change the
  SHA-256 key automatically — verified empirically. Old entries become
  unreachable orphans (no TTL configured), never staleness hazards.
  What remains deliberate: orphan clean-up and prompt-version
  provenance in run artifacts.
- `jinja2` becomes a direct dependency (it is already in the dependency
  graph transitively via docs tooling).
- Rendering errors move from `KeyError` (str.format) to
  `jinja2.exceptions.UndefinedError`; both surface before any LLM call.
