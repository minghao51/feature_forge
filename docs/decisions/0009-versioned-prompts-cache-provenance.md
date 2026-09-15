# ADR 0009: Versioned prompt files and cache provenance

- **Status:** Accepted
- **Date:** 2026-08-30
- **Related:** ADR 0008 (Jinja2 templating), ADR 0005 (colocation),
  ADR 0001 (DiskCache rule)

## Context

Cache keys are content-addressed over the full rendered `messages`, so
prompt edits already auto-invalidate cache entries (correction recorded
2026-08-30 in ADRs 0001/0005/0008). Two gaps remained:

1. **No provenance.** Nothing recorded *which* prompt template produced
   a cached response or a run artifact — a hit/miss log line shows a key
   prefix, not the template identity. Answering "which prompt version
   generated this response" required diffing git history.
2. **No addressable versions.** Prompt files were mutable-in-place; the
   only history was git. Re-running an old experiment with its exact
   prompt text required a git checkout, and the experimentation platform
   prefers resolved, self-contained run inputs.

## Decision

1. **Versioned files: `<name>.v<N>.yaml`.** Every prompt file carries an
   integer version in its filename (`unary.v1.yaml`). The registry
   resolves a bare `name` to the **highest version present** (latest);
   `get/render/render_messages(name, version=N)` pins an exact version.
   All versions remain loadable for exact reruns; no alias files, no
   index to maintain.
2. **Content identity.** On load, the registry computes
   `sha256(system + user)` and stores it with the `Prompt` alongside its
   filename version. Human-readable version (filename) + exact content
   hash (identity) are recorded together.
3. **Provenance travels with every LLM call.** `PromptProvenance`
   (`prompt_name`, `prompt_version`, `prompt_sha256`) is returned by the
   registry and passed as `prompt_meta=` to
   `LLMClient.complete/complete_json`. It is stored inside the cache
   payload and emitted on the `llm_request` / `llm_cache_hit` /
   `llm_cache_miss` / `llm_response` log events. It is *not* part of the
   cache key — the key already hashes the rendered messages, so
   provenance is metadata, not identity.
4. **Run artifacts record provenance.** Methods already persist the
   rendered prompt text in artifacts; they now persist the provenance
   dict alongside it.

## Consequences

- A cache entry and every artifact can be traced to an exact prompt
  file version and content hash; stale-prompt debugging becomes a
  lookup, not an archaeology dig.
- Bumping a prompt is: create `name.v(N+1).yaml`. Rollback for a rerun
  is: pin `version=N`. Old cache entries stay valid for old versions
  (same rendered text → same key).
- `_prompting.py` gains glob-based version discovery (cached per
  registry); the registry remains the single shared seam (ADR 0005
  point 3).
- `llm` layer stays methods-agnostic: `prompt_meta` is an opaque
  mapping there.
