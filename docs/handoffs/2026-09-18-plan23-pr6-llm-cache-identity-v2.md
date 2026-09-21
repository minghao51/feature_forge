# Plan 23 — PR 6: LLM Cache Request Identity v2

**Date:** 2026-09-18
**Status:** Ready for implementation (ADR 0020 accepted 2026-09-15)
**Spec:** `docs/decisions/0020-llm-cache-request-identity-v2.md` (decisions 1–4),
plan 23 §5.4 (identity contents), §6 PR 6 (scope + acceptance), §7 item 13
(test requirements)
**Executable contract:** the 8 strict xfails in
`tests/unit/test_plan23_llm_cache_identity.py` — this file is the acceptance
record; PR 6 is done when all 8 xfail markers are removed and the assertions
pass unchanged (plus the v1-sharing regression pin keeps passing).

## Context

v1 cache keys (`feature_forge.llm.cache.compute_cache_key`) hash
provider/model/messages/temperature/max_tokens/explicit kwargs only. The
endpoint (`base_url`), thinking mode (DeepSeek injects
`extra_body={"thinking": …}` inside `_call_api`, so it never reaches the
key), reasoning effort, and the effective response-format mode (ADR 0011
negotiated degradations) are all invisible to the key — behaviorally
different requests share cached responses. The 2026-09-18 fix batch added
the two missing §7-item-13 sub-clause stubs (kwargs variation, persisted
provenance secret scan), so the xfail suite now enumerates the complete
PR-6 contract.

Key-derivation architecture (verified 2026-09-18, all sites route through
one method):

- `BaseLLMProvider.build_cache_key(messages, temperature, max_tokens,
  **kwargs)` (`src/feature_forge/llm/base.py:653-665`) delegates to v1
  `compute_cache_key`.
- Call sites: plain `_do_complete` (`base.py:299`, gated `not json_mode`),
  `_do_complete_json` (`:411` miss / `:457` set, passes `json_mode=True` +
  mode kwargs), `_do_complete_structured` (`:495` via `_cache_key` helper,
  same pattern), repair round (`:562` → delegates to `_do_complete` plain
  path). **No caching-behavior change is needed** — every existing path
  already flows through `build_cache_key`; PR 6 changes what the key is
  derived FROM, not which calls cache.
- `DiskCache.get_key` (`cache.py:86`) has no src call sites (tests only) —
  leave it v1 or align it; not part of acceptance.
- Degradation state lives on the provider (`self._json_mode_degraded`,
  `self._structured_mode_degraded`, negotiated per ADR 0011).

## Agent workflow

- **Worker:** `worker` agent (pinned `opencode-go/deepseek-v4.1-flash`).
  Implements src changes + new non-marker tests; runs tests sequentially
  (never in parallel calls).
- **Reviewer:** `reviewer` agent (pinned `zai/glm-5.3-flash`), read-only,
  REQUEST-CHANGES loop per slice.
- **Main agent owns** `tests/unit/test_plan23_llm_cache_identity.py`
  (marker module): removing xfail markers + `type: ignore[attr-defined]`
  comments is done by the main agent after src lands, per the standing
  marker-module rule. The worker must not edit it.
- Standing rules: read before edit; surgical diffs; never commit/reset/
  stash; never print `.env` values; never bypass the enforced DiskCache;
  no new dependencies expected (stdlib `urllib.parse` for endpoint
  normalization).

## Implementation tasks

### T1 — `LLMClient.cache_identity()` (ADR 0020 decision 1)

On `BaseLLMProvider` (`base.py`), returning a JSON-serializable, secret-free
dict. Exact key names are pinned by the xfail suite — implement to match:

- `provider` (e.g. `"deepseek"`), `model`
- `endpoint`: normalized origin + path only — userinfo credentials and the
  entire query string (names AND values) stripped (pinned test uses
  `https://user:secret@gw.example.internal/v1?token=q-value` →
  `https://gw.example.internal/v1`). Use `urllib.parse.urlsplit`; default
  `base_url` when unset must remain secret-free.
- `thinking_enabled`, `reasoning_effort` (provider-specific flags: include
  when the provider supports them; base defaults should not fabricate
  provider-specific state — degrade honestly, e.g. omit or `None`)
- `response_format_mode`: the **effective** mode including negotiated
  degradation (pinned: a client with `_json_mode_degraded = True` must
  produce a different value than a fresh client)
- `cache_identity(**kwargs)`: provider request kwargs fold into the identity
  (`test_cache_identity_reflects_request_kwargs` pins
  `cache_identity()` != `cache_identity(top_p=0.9, frequency_penalty=0.5)`)

Subclass note: DeepSeek carries `thinking_enabled`/`reasoning_effort`;
design `cache_identity()` so each provider contributes its behavior flags
(base builds the common dict; provider hook for extras — avoid isinstance
checks).

### T2 — Schema version + v2 key derivation (decision 2)

- `feature_forge.llm.cache.CACHE_KEY_SCHEMA_VERSION = 2`.
- `build_cache_key` derives from: schema version, the full identity (T1),
  temperature, max_tokens, rendered messages, and provider request kwargs.
  Keep it deterministic and JSON-serializable (existing
  `json.dumps(..., sort_keys=True, default=str)` + SHA-256 pattern).
- **v1 compatibility (decision 4):** v1 entries are misses under v2 keys —
  natural, since v2 keys differ; do NOT rewrite/delete v1 entries and add no
  migration pass. Pinned: v2 key != legacy v1 key for identical args
  (`test_cache_key_schema_version_exists`).
- `compute_cache_key` stays as-is (it IS the v1 function; the legacy-key
  comparison in tests depends on it). `DiskCache.get_key`: not in
  acceptance; align opportunistically or leave with a comment.

### T3 — Persisted provenance (decision 3)

Every `cache.set` payload gains `cache_provenance`:
`{"cache_key_schema_version": 2, "cache_identity": <T1 dict>}` — secret-free
by construction. Touch all three write sites (plain `:328+` region, json
`:457`, structured set path). The persisted-scan xfail pins: payload key
name `cache_provenance`, sub-keys as above, and that neither the API key nor
endpoint userinfo appears anywhere in the raw bytes of the cache directory.

### T4 — Secrets discipline (decision 3)

- No secret material in keys, logs, or persisted payloads. Existing
  `llm_cache_hit`/`llm_cache_miss` logs already truncate keys to 16 chars —
  keep; do not log identity dicts that could carry secrets (they cannot, by
  T1, but verify).
- `LLMClient.api_key`/`SecretStr` handling untouched.

### T5 — Flip the xfails (main agent, after src lands)

`tests/unit/test_plan23_llm_cache_identity.py`:
- Remove `@pytest.mark.xfail(strict=True, ...)` from all 8 and the paired
  `# type: ignore[attr-defined]` comments (mypy `warn_unused_ignores`).
- Assertions stay byte-identical — that is the acceptance bar. If an
  assertion seems wrong, raise it with the maintainer instead of editing.
- The three module-level `_PLAN_54_*`/`_FINDING_8` reason constants become
  unused — remove with the markers.
- Update the module docstring from "does not exist yet" to the landed
  contract.

### T6 — Tests beyond the marker module (worker)

New file (e.g. `tests/unit/test_llm_cache_identity_v2.py`), non-marker:
- v1-miss behavior: write an entry under a v1 key directly via `DiskCache`,
  confirm a v2-keyed lookup misses (and vice versa a v2 entry is not
  readable via legacy `compute_cache_key`).
- Round-trip: plain call cached → hit on identical request; miss when
  endpoint/thinking/effort/mode/kwargs change (integration of T1+T2 through
  the real `_do_complete` path — reuse the offline
  monkeypatched-`_call_api` pattern from the persisted-scan stub).
- Provenance present on json/structured write paths too.
- Degradation negotiation (ADR 0011) still functions — existing
  `test_structured_outputs.py` must stay green.

### T7 — Docs

- `docs/operations.md` (or cache docs section): the upgrade causes one
  intentional cold-miss wave (expected, not a defect); v1 entries retained
  until maintenance evicts; rollback restores v1 hits losslessly (ADR 0020
  rollback note).
- REPORT_LOG entry (AI-assistance disclosure: worker DeepSeek V4.1 Flash,
  reviewer GLM-5.3-Flash, session agent coordination).
- ADR 0020 status line: accepted → implemented (only if that convention
  exists for other ADRs; check 0018/0019 wording first).

## Validation gates (sequential)

```bash
uv sync --all-groups --extra intel
uv run pytest tests/unit/test_plan23_llm_cache_identity.py -q        # 8 pass + 1 pin
uv run pytest tests/unit/test_llm_cache.py tests/unit/test_differential.py \
  tests/unit/test_llm_base.py tests/unit/test_structured_outputs.py \
  tests/unit/test_providers.py tests/unit/test_llm_replay.py -q
uv run pytest                                                       # full suite
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run python scripts/check_repo_hygiene.py
uv run python scripts/check_docs_references.py
uv run mkdocs build --strict && uv lock --check && git diff --check
```

Expect final counts: previous baseline 1106 passed / 9 skipped / 8 xfailed →
xfails 8 → 0 in that file, passed count grows by 8 + new T6 tests.

## Acceptance criteria (plan 23 §6 PR 6 + §7 item 13)

- [ ] All 8 marker-module xfails removed; assertions pass unchanged; the
      identical-requests-share-keys pin still passes (reuse survives v2)
- [ ] Endpoint, thinking, reasoning, response mode, and kwargs each vary the
      key independently (§7 item 13 clause 1)
- [ ] Persisted metadata scanned off-disk is secret-free and versioned
      (§7 item 13 clause 2)
- [ ] v1 entries treated as misses, retained untouched; no migration pass
- [ ] No secrets in keys, logs, or payloads
- [ ] All gates green; suite counts reconcile exactly
- [ ] Operator docs cover the cold-miss wave; REPORT_LOG entry present

## Out of scope

- Extending caching to currently-uncached call paths (none exist — verified
  2026-09-18; plain/json/structured/repair all cache today)
- Cross-client cache sharing semantics beyond the identical-request pin
- LLM replay/tracing changes (ADR 0009 prompt_meta stays as-is)
- PR 7 (documentation, qualification, and claims audit) — next and final
  plan-23 PR
