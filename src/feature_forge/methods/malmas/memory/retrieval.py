from __future__ import annotations

from typing import Any


def retrieve_top_k(
    entries: list[dict[str, Any]],
    current_columns: set[str],
    current_round: int,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Score and rank memory entries by relevance to current context.

    Each entry gets a relevance score: 50% column overlap, 30% recency, 20% gain.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for e in entries:
        cols = set(e.get("base_columns", []))
        union = cols | current_columns
        overlap = len(cols & current_columns) / max(len(union), 1)
        round_idx = int(e.get("round_idx", current_round))
        recency_denominator = max(1, 1 + current_round - round_idx)
        recency = 1 / recency_denominator
        gain = max(e.get("value", 0), 0)
        score = 0.5 * overlap + 0.3 * recency + 0.2 * gain
        scored.append((score, e))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]
