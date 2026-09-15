# ADR 0011: Strict structured outputs with pydantic and gateway negotiation

- Status: Accepted (2026-09-05)
- Deciders: maintainer + AI assistant (audit follow-up)
- Context: docs/decisions/0010, REPORT_LOG.md (2026-09-05 hy3 `json_mode` incident)

## Context and problem

The 2026-09-05 backend incident exposed two contract gaps in the LLM layer:

1. The OpenCode Go gateway **rejects `response_format: {"type": "json_object"}`**
   for `hy3` (HTTP 400), which is exactly what `OpenAIProvider._json_mode_kwargs()`
   sent on every `complete_json()` call — every agent generation call failed.
2. `complete_json()` takes a *prose* `schema_description` and returns unvalidated
   `JSONValue`; every caller then hand-parses dicts and re-validates field by
   field (router.py, malmus/method.py).

Probing (`scripts/probe_structured_output.py`) established the capability
matrix for the live gateway:

| mode | result |
|------|--------|
| `json_object` | 400 — rejected |
| `json_schema` strict | 200, schema-conformant output |
| prompt-only (schema instruction) | 200, model follows the schema |

## Decision

1. **`LLMClient.complete_structured(messages, response_model)`** is the new
   preferred path for machine-consumed LLM output. It:
   - builds an OpenAI **strict `json_schema` response_format** from a pydantic
     model (`llm/structured.py: build_strict_schema`, all-`required`,
     `additionalProperties: false`);
   - parses tolerantly (`<think>` blocks, markdown fences, prose-wrapped JSON);
   - **validates the reply with the same pydantic model**;
   - performs **exactly one repair round** on validation failure (error fed
     back to the model), then raises `LLMError`.
2. **Gateway negotiation, memoized per client**: any 400 while sending
   `response_format` degrades that client to prompt-only schema enforcement
   for its lifetime (`_json_mode_degraded` / `_structured_mode_degraded`),
   logged as `llm_json_mode_degraded` / `llm_structured_mode_degraded`. No
   configuration knob — behavior is probed, not configured.
3. The legacy `complete_json(schema_description: str)` path keeps its
   signature (all agent call sites still work) but gains the negotiation and
   the tolerant parser. New code should prefer `complete_structured`.
4. Providers may override `_structured_kwargs()`; Anthropic's override returns
   `{}` (no response_format in its API → prompt-mode + validation + repair).
5. Cache semantics unchanged: response_format kwargs are part of the cache
   key, so a mode switch can never serve a stale response; validated content
   is cached under the key of the request shape that produced it.

## Consequences

- Agent outputs become typed and validated at the boundary; hand-rolled
  `response.get(...)` validation can migrate call-site by call-site.
- A gateway that silently ignores `response_format` (no 400) still gets
  schema-conformant replies from the injected instruction, and mismatches are
  caught by pydantic + repaired — correctness does not depend on enforcement.
- The repair round costs one extra LLM call only when the model actually
  violates the schema (rare with strict mode).
- `_json_mode_kwargs()` stays `json_object` for genuine OpenAI/DeepSeek
  endpoints where it is documented; the Go gateway degrades automatically.
