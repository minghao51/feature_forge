# ADR 0020: LLM cache request identity v2

- **Status:** Accepted (maintainer accepted on 2026-09-15)
- **Date:** 2026-09-15
- **Related:** ADR 0001 (enforced mandatory cache), ADR 0009 (cache
  provenance), ADR 0011 (structured outputs and negotiated degradation),
  `docs/plan/23_evaluation_integrity_security_hardening.md` (§5.4),
  `tests/unit/test_plan23_llm_cache_identity.py`, `REPORT_LOG.md`

## Context

Today's cache keys (`feature_forge.llm.cache.compute_cache_key`) hash the
provider name, model, messages, temperature, max_tokens, and explicit request
kwargs only. The provider endpoint (`base_url`), thinking mode, reasoning
effort, and the effective response-format mode — including ADR 0011's
negotiated `json`/`structured` degradations — never enter the key, and
`DeepSeekProvider` injects `extra_body={"thinking": …}` inside `_call_api`,
so thinking mode cannot reach the key even indirectly. Two behaviorally
different requests can therefore share one cached response (demonstrated per
setting in `tests/unit/test_plan23_llm_cache_identity.py`, plan 23 PR 1).

On 2026-09-15 the maintainer accepted this ADR, unlocking the cache runtime
changes (plan 23 PR 6).

## Decision

1. **Secret-free identity.** `LLMClient.cache_identity()` returns a
   JSON-serializable, secret-free dict covering: provider and model;
   normalized endpoint origin/path with credentials and query values removed;
   thinking/reasoning settings; the effective response-format mode and other
   provider behavior flags.
2. **Key schema v2.** `feature_forge.llm.cache` exposes
   `CACHE_KEY_SCHEMA_VERSION = 2`. Every cache key — plain, JSON-mode,
   structured-output, degraded-response-format, and repair calls — is derived
   from the schema version, the identity, temperature, token limit, rendered
   messages, and provider request kwargs.
3. **No secrets.** API keys and any secret material never appear in keys,
   logs, or persisted payloads; cache provenance records the key schema
   version and the secret-free identity.
4. **v1 compatibility.** v1 entries are treated as misses under v2 keys and
   retained untouched until normal cache maintenance removes them; no rewrite
   or migration pass runs.

## Consequences

- Behaviorally identical requests still share cache entries; any
  output-affecting change (endpoint, thinking mode, reasoning effort,
  response mode, material kwargs) produces a distinct key.
- Upgrading causes one intentional cold-miss wave — documented for operators
  as expected, not a defect.
- Key payloads grow slightly; provenance gains the schema version and
  identity, making cache reuse auditable across provider behavior changes.
- Plan 23 PR 6 implements this record and documents the migration.
- **Rollback.** Each plan 23 PR is independently revertible; v1 cache files
  remain untouched under v2 keys, so reverting the key schema restores v1
  hits losslessly at the cost of one further cold-miss wave.
