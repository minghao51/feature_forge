"""Tests for strict structured outputs (ADR 0011)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient
from feature_forge.llm.structured import build_strict_schema, parse_json_payload


class Inner(BaseModel):
    label: str


class ProbeResult(BaseModel):
    ok: bool
    answer: str
    inner: Inner
    notes: list[str]


class FakeProvider(LLMClient):
    """Scripted provider: pops queued raw contents / exceptions per call."""

    def __init__(self, script: list[str | Exception]) -> None:
        super().__init__(model="fake", api_key="sk-test")
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def _json_mode_kwargs(self) -> dict[str, Any]:
        return {"response_format": {"type": "json_object"}}

    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        item = self.script.pop(0) if self.script else "{}"
        if isinstance(item, Exception):
            raise item
        return item  # raw content passthrough via _extract_content

    def _extract_content(self, raw_response: str) -> str:
        return raw_response

    def _extract_usage(self, raw_response: str) -> tuple[int, int, int]:
        return (1, 1, 2)


VALID = '{"ok": true, "answer": "hy3", "inner": {"label": "x"}, "notes": ["a"]}'


class TestBuildStrictSchema:
    def test_strict_subset(self) -> None:
        wrapper = build_strict_schema(ProbeResult)
        assert wrapper["strict"] is True
        assert wrapper["name"] == "proberesult"
        schema = wrapper["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"ok", "answer", "inner", "notes"}
        inner = schema["$defs"]["Inner"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["label"]

    def test_custom_name(self) -> None:
        assert build_strict_schema(ProbeResult, "my_schema")["name"] == "my_schema"

    def test_defaults_dropped_and_required(self) -> None:
        class WithDefault(BaseModel):
            a: int
            b: str = "x"

        schema = build_strict_schema(WithDefault)["schema"]
        assert schema["required"] == ["a", "b"]
        assert "default" not in schema["properties"]["b"]


class TestParseJsonPayload:
    def test_plain(self) -> None:
        assert parse_json_payload('{"a": 1}') == {"a": 1}

    def test_fenced(self) -> None:
        assert parse_json_payload('```json\n{"a": 1}\n```') == {"a": 1}

    def test_think_block(self) -> None:
        content = '<think>reasoning about the answer</think>\n{"a": 1}'
        assert parse_json_payload(content) == {"a": 1}

    def test_prose_wrapped(self) -> None:
        content = 'Here is the result you asked for: {"a": 1, "b": [1, 2]} — done.'
        assert parse_json_payload(content) == {"a": 1, "b": [1, 2]}

    def test_braces_inside_strings(self) -> None:
        content = '{"a": "literal } brace"}'
        assert parse_json_payload(content) == {"a": "literal } brace"}

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_json_payload("   ")

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no valid JSON"):
            parse_json_payload("no json here at all")


@pytest.mark.asyncio
class TestCompleteStructured:
    async def test_happy_path_sends_json_schema(self) -> None:
        provider = FakeProvider([VALID])
        result = await provider.complete_structured(
            [{"role": "user", "content": "go"}], ProbeResult
        )
        assert isinstance(result, ProbeResult)
        assert result.answer == "hy3"
        kwargs = provider.calls[0]["kwargs"]
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["strict"] is True
        # schema instruction injected into the system slot
        assert "valid JSON" in provider.calls[0]["messages"][0]["content"]

    async def test_fenced_reply_validates(self) -> None:
        provider = FakeProvider(["```json\n" + VALID + "\n```"])
        result = await provider.complete_structured([{"role": "user", "content": "x"}], ProbeResult)
        assert result.ok is True

    async def test_repair_round_recovers(self) -> None:
        bad = '{"ok": true, "answer": 42, "inner": {"label": "x"}, "notes": []}'
        provider = FakeProvider([bad, VALID])
        result = await provider.complete_structured([{"role": "user", "content": "x"}], ProbeResult)
        assert result.answer == "hy3"
        assert len(provider.calls) == 2
        repair_messages = provider.calls[1]["messages"]
        assert repair_messages[-2]["role"] == "assistant"
        assert "violated" in repair_messages[-1]["content"]

    async def test_repair_failure_raises(self) -> None:
        bad = '{"ok": "not-a-bool"}'
        provider = FakeProvider([bad, bad])
        with pytest.raises(LLMError, match="failed validation"):
            await provider.complete_structured([{"role": "user", "content": "x"}], ProbeResult)
        assert len(provider.calls) == 2

    async def test_400_degrades_to_prompt_mode_and_memoizes(self) -> None:
        provider = FakeProvider([LLMError("fake API error: Error code: 400 - rejected"), VALID])
        result = await provider.complete_structured([{"role": "user", "content": "x"}], ProbeResult)
        assert result.ok is True
        assert "response_format" not in provider.calls[1]["kwargs"]  # retried without
        assert provider._structured_mode_degraded is True
        # memoized: subsequent structured calls skip response_format entirely
        provider.script = [VALID]
        await provider.complete_structured([{"role": "user", "content": "y"}], ProbeResult)
        assert "response_format" not in provider.calls[-1]["kwargs"]

    async def test_non_400_error_propagates(self) -> None:
        provider = FakeProvider([LLMError("fake API error: Error code: 429 - rate limited")])
        with pytest.raises(LLMError, match="429"):
            await provider.complete_structured([{"role": "user", "content": "x"}], ProbeResult)


@pytest.mark.asyncio
class TestCompleteJsonNegotiation:
    async def test_json_object_400_degrades(self) -> None:
        provider = FakeProvider(
            [LLMError("fake API error: Error code: 400 - nope"), '{"agents": ["unary"]}']
        )
        result = await provider.complete_json(
            [{"role": "user", "content": "pick"}], '{"agents": ["unary"]}'
        )
        assert result == {"agents": ["unary"]}
        assert "response_format" in provider.calls[0]["kwargs"]
        assert "response_format" not in provider.calls[1]["kwargs"]
        assert provider._json_mode_degraded is True

    async def test_json_object_used_when_supported(self) -> None:
        provider = FakeProvider(['{"agents": ["unary"]}'])
        await provider.complete_json([{"role": "user", "content": "x"}], "{}")
        assert provider.calls[0]["kwargs"] == {"response_format": {"type": "json_object"}}

    async def test_fenced_json_parsed(self) -> None:
        provider = FakeProvider(['```json\n{"ok": true}\n```'])
        result = await provider.complete_json([{"role": "user", "content": "x"}], "{}")
        assert result == {"ok": True}
