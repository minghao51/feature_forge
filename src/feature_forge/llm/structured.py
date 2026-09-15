"""Structured outputs: pydantic-first JSON handling for LLM responses.

Empirical baseline (OpenCode Go gateway, ``hy3`` = Hunyuan 3, probed
2026-09-05 with ``scripts/probe_structured_output.py``):

=================  ===========================  =============================
mode               gateway                      notes
=================  ===========================  =============================
``json_object``    400 (rejected)               what providers sent before
``json_schema``    200, schema-conformant       strict, pydantic-generated
prompt-only        200, follows injected        reliable degraded fallback
                   schema instruction
=================  ===========================  =============================

Design:

- ``LLMClient.complete_structured()`` sends an OpenAI strict
  ``response_format: json_schema`` built from a pydantic model, then
  validates the reply with the same model and performs exactly one
  repair round on ``ValidationError``.
- Gateways that reject ``response_format`` degrade per-client to
  prompt-only mode (schema instruction + robust parsing), memoized for
  the client's lifetime.
- ``parse_json_payload()`` tolerates reasoning-model quirks: ``<think>``
  blocks, markdown fences, and prose-wrapped JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def build_strict_schema(model: type[BaseModel], name: str | None = None) -> dict[str, Any]:
    """Build the ``json_schema`` wrapper for OpenAI strict structured outputs.

    Produces ``{"name": ..., "strict": True, "schema": ...}`` where the
    schema is the pydantic model's JSON schema normalized to the strict
    subset: every object declares ``additionalProperties: false`` and
    lists **all** its properties in ``required`` (``default`` values are
    dropped so the model must emit every key; pydantic validation still
    accepts the resulting objects unchanged).

    Nested models are inlined via ``$defs``/``$ref``, which strict mode
    supports.
    """
    schema: dict[str, Any] = json.loads(json.dumps(model.model_json_schema()))
    _strictify(schema)
    return {"name": name or model.__name__.lower(), "strict": True, "schema": schema}


def _strictify(node: Any) -> None:
    """Recursively normalize a JSON-schema node to the strict subset."""
    if isinstance(node, dict):
        props = node.get("properties")
        if node.get("type") == "object" and isinstance(props, dict):
            node["additionalProperties"] = False
            node["required"] = sorted(props)
            for value in props.values():
                value.pop("default", None)
                _strictify(value)
        for key in ("$defs", "definitions"):
            for sub in node.get(key, {}).values():
                _strictify(sub)
        for key in ("items", "prefixItems"):
            _strictify(node.get(key))
        for combinator in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(combinator, []):
                _strictify(sub)
    elif isinstance(node, list):
        for sub in node:
            _strictify(sub)


def parse_json_payload(content: str) -> Any:
    """Parse a JSON payload from raw LLM content, tolerating noise.

    Tries, in order: the stripped text as-is; markdown-fence bodies;
    ``<think>``-stripped text; the first balanced ``{...}``/``[...]``
    substring. Raises ``ValueError`` when nothing parses.
    """
    text = content.strip()
    if not text:
        raise ValueError("empty response content")

    candidates: list[str] = [text]
    candidates += [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    think_stripped = _THINK_RE.sub("", text).strip()
    if think_stripped and think_stripped != text:
        candidates.append(think_stripped)
        candidates += [m.group(1).strip() for m in _FENCE_RE.finditer(think_stripped)]
    balanced = _first_balanced(text) or _first_balanced(think_stripped)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no valid JSON found in: {content[:200]!r}")


def _first_balanced(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` substring, if any."""
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        close = "}" if ch == "{" else "]"
        depth, in_string, escaped = 0, False, False
        for pos in range(start, len(text)):
            c = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
            elif c == '"':
                in_string = True
            elif c == ch:
                depth += 1
            elif c == close:
                depth -= 1
                if depth == 0:
                    return text[start : pos + 1]
        return None  # unbalanced from the first opener — no later opener can fix it
    return None


def build_repair_messages(
    messages: list[dict[str, str]], content: str, exc: Exception
) -> list[dict[str, str]]:
    """Build the one-shot repair message list for a failed structured reply."""
    if isinstance(exc, ValidationError):
        detail = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
    else:
        detail = f"reply was not parseable as JSON: {exc}"
    repair_instruction = (
        "Your previous reply violated the required output contract:\n"
        f"{detail}\n"
        "Return ONLY the corrected JSON object conforming to the schema above. "
        "No explanation, no markdown fences."
    )
    return [
        *messages,
        {"role": "assistant", "content": content},
        {"role": "user", "content": repair_instruction},
    ]


__all__ = [
    "build_repair_messages",
    "build_strict_schema",
    "parse_json_payload",
]
