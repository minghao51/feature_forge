"""Shared utilities used across feature_forge."""

from __future__ import annotations


def strip_markdown_fences(code: str) -> str:
    """Remove leading/trailing markdown code fences from a string.

    Handles both `````python`` and bare ``````` fences.
    """
    if code.startswith("```"):
        code = code.removeprefix("```python").removeprefix("```")
        code = code.removesuffix("```").strip()
    return code
