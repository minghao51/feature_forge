#!/usr/bin/env python
"""Probe structured-output support of the configured LLM backend.

Diagnostic for the 2026-09-05 hy3 `json_mode` 400s: sends three tiny
completions (baseline / json_object / json_schema-strict) through the
standard LLMClient (no cache, no retries) and reports which response_format
modes the gateway accepts. Also dumps a content preview so fence/thinking
markup can be inspected for parser design.

Usage: uv run python scripts/probe_structured_output.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel, TypeAdapter

from feature_forge.config import RetryConfig, get_settings
from feature_forge.exceptions import LLMError
from feature_forge.llm.factory import create_llm_client


class ProbeResult(BaseModel):
    ok: bool
    answer: str


async def main() -> None:
    settings = get_settings()
    llm_cfg = settings.llm.model_copy(update={"cache_responses": False})
    client = create_llm_client(llm_cfg, retry_config=RetryConfig(max_retries=0))
    messages = [{"role": "user", "content": 'Return a JSON object: {"ok": true, "answer": "hy3"}'}]

    schema = TypeAdapter(ProbeResult).json_schema()
    schema["additionalProperties"] = False
    schema["required"] = sorted(schema["properties"])  # strict mode: all required

    max_tokens = int(os.environ.get("PROBE_MAX_TOKENS", "1024"))
    modes: dict[str, dict[str, Any] | None] = {
        "baseline (no response_format)": None,
        "json_schema (strict, proper)": {
            "type": "json_schema",
            "json_schema": {"name": "probe_result", "strict": True, "schema": schema},
        },
    }

    for label, fmt in modes.items():
        try:
            kwargs = {} if fmt is None else {"response_format": fmt}
            resp = await client.complete(messages, temperature=0.0, max_tokens=max_tokens, **kwargs)
            print(
                f"[OK]   {label} (tokens: {resp.total_tokens})\n"
                f"       content={resp.content[:400]!r}"
            )
        except LLMError as exc:
            print(f"[FAIL] {label}\n       {str(exc)[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
