"""Regression tests encoding finding 8 from docs/plan/23_evaluation_integrity_security_hardening.md §1.

Finding 8 (Medium): LLM cache identity omits the normalized provider endpoint
and behavior settings. ``compute_cache_key(provider, model, messages,
temperature, max_tokens, **kwargs)`` covers only provider/model/messages/
temperature/max_tokens/explicit kwargs — it omits ``base_url``/endpoint,
``thinking_enabled``, ``reasoning_effort``, and the effective response-format
mode (including the negotiated ``_json_mode_degraded`` /
``_structured_mode_degraded`` flags); DeepSeek injects
``extra_body={'thinking': ...}`` inside ``_call_api``, so thinking mode never
reaches the key. Two behaviorally different requests can share one cached
response.

The xfail-strict tests encode the desired post-fix behavior (plan §5.4 and
PR 6): cache keys must incorporate the normalized endpoint, thinking/reasoning
settings, and the effective response-format mode; ``LLMClient.cache_identity()``
must expose a secret-free, JSON-serializable request identity; and
``feature_forge.llm.cache`` must expose ``CACHE_KEY_SCHEMA_VERSION == 2`` so
v1 entries are distinguishable (and treated as misses) after the migration.
Two further §7-item-13 sub-clause stubs pin the remaining coverage: the v2
request identity must vary with provider request kwargs independently, and
persisted cache entries must carry a secret-free, versioned provenance record
that survives being scanned off disk. One test pins the today-valid property
that separately constructed identical clients still share keys — it must keep
passing after the v2 migration.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from feature_forge.llm.cache import DiskCache, compute_cache_key
from feature_forge.llm.providers.deepseek import DeepSeekProvider

_FINDING_8_REASON = (
    "plan 23 §1 finding 8 (LLM cache identity): compute_cache_key omits the "
    "normalized endpoint and behavior settings (base_url, thinking_enabled, "
    "reasoning_effort, effective response-format mode), so behaviorally "
    "different requests share one cached response"
)
_PLAN_54_CACHE_IDENTITY_REASON = (
    "plan 23 §5.4 / PR 6 (LLM cache identity): LLMClient.cache_identity() "
    "does not exist yet (AttributeError today)"
)
_PLAN_54_SCHEMA_VERSION_REASON = (
    "plan 23 §5.4 / PR 6 (LLM cache identity): feature_forge.llm.cache exposes "
    "no CACHE_KEY_SCHEMA_VERSION and keys are not yet versioned (v1 only)"
)
_PLAN_54_KWARGS_REASON = (
    "plan 23 §5.4 / §7 item 13 (LLM cache identity): cache_identity() does not "
    "exist yet (AttributeError today); the v2 identity must vary with provider "
    "request kwargs independently of endpoint/thinking/reasoning/response mode"
)
_PLAN_54_PERSISTED_METADATA_REASON = (
    "plan 23 §5.4 / §7 item 13 (LLM cache identity): persisted cache entries "
    "carry no versioned provenance record, so there is no on-disk metadata to "
    "scan for leaked secrets"
)


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "plan23 finding 8"}]


def _deepseek(**overrides: Any) -> DeepSeekProvider:
    """Offline DeepSeek client with a dummy key and tracing disabled."""
    params: dict[str, Any] = {
        "model": "deepseek-chat",
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "tracing_enabled": False,
    }
    params.update(overrides)
    return DeepSeekProvider(**params)


class TestFinding8CacheIdentityCollisions:
    """Behaviorally different requests currently collide on one cache key."""

    @pytest.mark.xfail(strict=True, reason=_FINDING_8_REASON)
    def test_cache_keys_differ_across_endpoints(self) -> None:
        official = _deepseek()
        gateway = _deepseek(base_url="https://gw.example.internal/v1")
        key_official = official.build_cache_key(_messages(), 0.2, 100)
        key_gateway = gateway.build_cache_key(_messages(), 0.2, 100)
        assert key_official != key_gateway

    @pytest.mark.xfail(strict=True, reason=_FINDING_8_REASON)
    def test_cache_keys_differ_across_thinking_mode(self) -> None:
        # DeepSeek injects extra_body={'thinking': ...} inside _call_api, so
        # the flag changes the wire request but never the cache key. Same
        # reasoning_effort on both sides to isolate the thinking flag.
        plain = _deepseek(thinking_enabled=False, reasoning_effort="medium")
        thinking = _deepseek(thinking_enabled=True, reasoning_effort="medium")
        assert plain.build_cache_key(_messages(), 0.2, 100) != thinking.build_cache_key(
            _messages(), 0.2, 100
        )

    @pytest.mark.xfail(strict=True, reason=_FINDING_8_REASON)
    def test_cache_keys_differ_across_reasoning_effort(self) -> None:
        # thinking_enabled=True on both sides: this is the only mode in which
        # DeepSeek actually sends reasoning_effort, so the effort difference is
        # material on the wire and must split the cache key.
        low = _deepseek(thinking_enabled=True, reasoning_effort="low")
        high = _deepseek(thinking_enabled=True, reasoning_effort="high")
        assert low.build_cache_key(_messages(), 0.2, 100) != high.build_cache_key(
            _messages(), 0.2, 100
        )

    @pytest.mark.xfail(strict=True, reason=_FINDING_8_REASON)
    def test_cache_keys_differ_across_degraded_response_mode(self) -> None:
        strict_mode = _deepseek()
        degraded = _deepseek()
        # v2 identity must cover the EFFECTIVE response-format mode, i.e. the
        # negotiated per-client degradation state (ADR 0011), not just the
        # configured response_format. The flags are private today; PR 6 will
        # expose them properly (e.g. via cache_identity()), so set the
        # attribute directly here. json_mode=True mirrors the real call site
        # (_do_complete_json) where degradation is negotiated.
        degraded._json_mode_degraded = True
        assert strict_mode.build_cache_key(
            _messages(), 0.2, 100, json_mode=True
        ) != degraded.build_cache_key(_messages(), 0.2, 100, json_mode=True)


class TestDesiredV2CacheIdentityContract:
    """Desired §5.4 API surface: cache_identity() and a versioned key schema."""

    @pytest.mark.xfail(strict=True, reason=_PLAN_54_CACHE_IDENTITY_REASON)
    def test_cache_identity_is_secret_free_and_complete(self) -> None:
        client = _deepseek(
            base_url="https://user:secret@gw.example.internal/v1?token=q-value",
            thinking_enabled=True,
            reasoning_effort="high",
        )
        # Desired v2 API (plan §5.4): a secret-free, JSON-serializable request
        # identity. The key names below are the contract this test pins; PR 6
        # implements cache_identity() to match. Remove the type: ignore[attr-
        # defined] together with the xfail marker once the method lands
        # (warn_unused_ignores).
        identity: dict[str, object] = client.cache_identity()  # type: ignore[attr-defined]
        rendered = json.dumps(identity, sort_keys=True)
        assert "test-key" not in rendered, "api key leaked into cache identity"
        assert "secret" not in rendered, "endpoint credentials leaked into cache identity"
        assert identity["provider"] == "deepseek"
        assert identity["model"] == "deepseek-chat"
        # Normalized endpoint: origin + path only — userinfo credentials and
        # the query string (including their values) are stripped.
        assert identity["endpoint"] == "https://gw.example.internal/v1"
        assert identity["thinking_enabled"] is True
        assert identity["reasoning_effort"] == "high"
        # Effective response-format mode: the negotiated degradation state must
        # be reflected in the identity, not just the configured mode.
        plain_client = _deepseek()
        degraded_client = _deepseek()
        degraded_client._json_mode_degraded = True
        plain_identity: dict[str, object] = plain_client.cache_identity()  # type: ignore[attr-defined]
        degraded_identity: dict[str, object] = degraded_client.cache_identity()  # type: ignore[attr-defined]
        assert (
            plain_identity["response_format_mode"] != degraded_identity["response_format_mode"]
        ), "cache identity ignores the effective (degraded) response-format mode"

    @pytest.mark.xfail(strict=True, reason=_PLAN_54_SCHEMA_VERSION_REASON)
    def test_cache_key_schema_version_exists(self) -> None:
        from feature_forge.llm import cache as llm_cache

        # Desired v2 API (plan §5.4): versioned cache keys so v1 entries can
        # be distinguished and treated as misses after the migration. Remove
        # the type: ignore[attr-defined] together with the xfail marker once
        # the constant lands (warn_unused_ignores).
        assert llm_cache.CACHE_KEY_SCHEMA_VERSION == 2  # type: ignore[attr-defined]
        client = _deepseek()
        v2_key = client.build_cache_key(_messages(), 0.2, 100)
        legacy_v1_key = compute_cache_key(client.provider_name, client.model, _messages(), 0.2, 100)
        # The v2 key must not collide with the legacy v1 key on identical
        # args, so stale v1 entries are naturally treated as cold misses.
        assert v2_key != legacy_v1_key

    @pytest.mark.xfail(strict=True, reason=_PLAN_54_KWARGS_REASON)
    def test_cache_identity_reflects_request_kwargs(self) -> None:
        client = _deepseek()
        # Desired v2 API (plan §5.4 / §7 item 13): provider request kwargs are
        # part of the request identity and must vary the identity on their own
        # (independently of endpoint/thinking/reasoning/response mode). PR 6
        # implements cache_identity(**kwargs) to match; remove the type:
        # ignore[attr-defined] together with the xfail marker once it lands
        # (warn_unused_ignores).
        plain: dict[str, object] = client.cache_identity()  # type: ignore[attr-defined]
        with_kwargs: dict[str, object] = client.cache_identity(  # type: ignore[attr-defined]
            top_p=0.9, frequency_penalty=0.5
        )
        assert plain != with_kwargs, "provider request kwargs must be part of the v2 cache identity"

    @pytest.mark.xfail(strict=True, reason=_PLAN_54_PERSISTED_METADATA_REASON)
    def test_persisted_cache_metadata_is_versioned_and_secret_free(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§7 item 13: the secret scan must reach persisted metadata, not dicts in memory."""
        cache = DiskCache(cache_dir=str(tmp_path / "cache"), enabled=True)
        client = _deepseek(
            api_key="sk-plan23-persisted-scan",
            base_url="https://user:scan-pass@gw.example.internal/v1",
            cache=cache,
        )

        async def fake_call_api(
            messages: list[dict[str, str]], temperature: float, max_tokens: int, **kwargs: Any
        ) -> dict[str, str]:
            return {"content": "plan23 persisted scan"}

        # Keep the real write path (_do_complete -> cache.set); only the wire
        # call and response parsing are stood down so the test stays offline.
        monkeypatch.setattr(client, "_call_api", fake_call_api)
        monkeypatch.setattr(client, "_extract_content", lambda raw: raw["content"])
        monkeypatch.setattr(client, "_extract_usage", lambda raw: (1, 1, 2))

        response = asyncio.run(client.complete(_messages(), temperature=0.2, max_tokens=32))
        assert response.content == "plan23 persisted scan"

        cache_key = client.build_cache_key(_messages(), 0.2, 32)
        payload = cache.get(cache_key)
        assert payload is not None, "caching write path did not persist the response"
        # Desired v2 contract (plan §5.4 / §7 item 13): every persisted entry
        # records a versioned, secret-free provenance dict. The key names below
        # are the contract this test pins; PR 6 implements the payload to match.
        provenance: dict[str, Any] = payload["cache_provenance"]
        assert provenance["cache_key_schema_version"] == 2
        assert provenance["cache_identity"]["provider"] == "deepseek"

        # Off-disk scan of every persisted byte: neither the API key nor the
        # endpoint's userinfo credentials may appear in the cache directory.
        persisted_blob = b"".join(
            p.read_bytes() for p in sorted((tmp_path / "cache").rglob("*")) if p.is_file()
        )
        assert b"sk-plan23-persisted-scan" not in persisted_blob
        assert b"scan-pass" not in persisted_blob


class TestRegressionPinIdenticalRequestsShareKeys:
    def test_identical_requests_still_share_keys(self) -> None:
        """Regression pin (expected to pass today AND after the v2 migration).

        Two separately constructed identical clients (same provider, model,
        base_url, and behavior flags) must produce equal cache keys for equal
        requests, or the cache provides no reuse at all.
        """
        common: dict[str, Any] = {
            "base_url": "https://gw.example.internal/v1",
            "thinking_enabled": True,
            "reasoning_effort": "low",
        }
        first = _deepseek(**common)
        second = _deepseek(**common)
        messages = _messages()
        assert first.build_cache_key(messages, 0.2, 100) == second.build_cache_key(
            messages, 0.2, 100
        )
        # Determinism: repeated key builds on one client are stable.
        assert first.build_cache_key(messages, 0.2, 100) == first.build_cache_key(
            messages, 0.2, 100
        )
