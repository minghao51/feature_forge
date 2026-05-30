# SAGE-Inspired Memory Evolution Plan

**Version:** 0.2.0
**Date:** 2026-05-17
**Status:** Updated after honest assessment
**Inspired by:** [SAGE: Self-Evolving Agentic Graph-Memory Engine](https://arxiv.org/abs/2605.12061v1) (Peking University / BIT)
**Reference codebase:** https://anonymous.4open.science/r/Unified-Representation-A9D9/

---

## 1. Executive Summary

Feature Forge uses a flat, JSON-based 3-tier memory system (procedural, feedback, conceptual). The SAGE paper was investigated for inspiration, but after analysis, most of its ideas (graph infrastructure, structurally gated propagation, GFM pretraining) are **overkill** for the current scale: ~3-5 rounds, 6 agents, ~50-300 features total across runs. A flat list is functionally equivalent to a graph at this scale.

**What's worth taking from SAGE:** Only the idea of being selective about what you retrieve. The rest is noise.

---

## 2. Current State Analysis

### 2.1 Current Memory Architecture

```
AgentMemory (per-agent JSON file)
├── procedural: [{base_columns, transform, feature_name, type, description, round_idx}]
├── unused_procedural: [{base_columns, transform, feature_name, type, description, round_idx}]
├── feedback: [{feature_name, metric, value, effective, round_idx, base_columns, type}]
├── conceptual: [str]  (LLM-summarized rules)
├── global_summary: [str]
├── stats: {effective_transforms, effective_fields, effective_types}
└── conceptual_summary: str
```

**Write path** (`iterative.py:338-359`):
1. After each round, `_post_round()` iterates agents
2. For each feature, records procedural + feedback + (if ineffective) unused_procedural
3. Saves to JSON via `MemoryPersistence`
4. Updates router performance scores

**Read path** (`iterative.py:288-300`):
1. `_build_agent_context()` calls `memory.generate_prompt_section(use_feedback=True)`
2. Gets positive/negative feature lists
3. Injects memory context string into agent prompt as flat text

### 2.2 Limitations

| # | Limitation | Impact | Worth fixing? |
|---|---|---|---|
| L1 | **No relationships between features** — flat lists, no graph | Agents can't reason over feature dependencies | **No** — at this scale, the LLM handles cross-feature reasoning fine in text |
| L2 | **No retrieval ranking** — all memory dumped into prompt | Prompt bloat; irrelevant history competes with relevant | **Yes** — cheap fix, noticeable improvement |
| L3 | **No feedback loop** — memory records but doesn't improve | Same mistakes repeated | **Defer** — valuable in theory, but ~100 entries is small enough that LLM sees the pattern |
| L4 | **No cross-agent sharing** — isolated JSON per agent | Agents can't learn from each other | **No** — 6 agents with <50 features each makes this redundant |
| L5 | **No cross-dataset transfer** — memory per dataset | Every new dataset starts from scratch | **No** — research-level problem, not practical here |
| L6 | **No deducibility signal** — can't distinguish useful from noise | Noisy conceptual summaries | **Low priority** — LLM summarization already filters noise decently |

---

## 3. Reality Check

### Is there a graph in the current setup?

**No.** It's zero graph. `AgentMemory` is a list of dicts per agent, stored as flat JSON. No edges, no relationships, no cross-agent structure.

### Is graph infrastructure overkill?

**Yes, clearly.** Here's why:

| Approach | Best for | Our scale | Verdict |
|---|---|---|---|
| NetworkX graph | >10K nodes, complex traversals | ~50-300 entries total | ❌ 500+ lines of infrastructure for a sorted list |
| Structurally gated propagation | Millions of edges, learned attention | ~0 edges, no topology to learn | ❌ Months of work, zero benefit |
| GFM pretraining | Cross-graph transfer at scale | 1 graph per dataset run | ❌ Research-level, not applicable |
| Top-K weighted scoring | Small lists with relevance signals | Exactly our scale | ✅ ~30 lines, immediate value |

**For <500 items, a list with a scoring function is equivalent to any graph traversal.** The only question is: do you dump everything into the prompt, or do you pick the best K?

---

## 4. Single Improvement: Top-K Relevance Retrieval

This is the only idea from SAGE worth implementing. Everything else in the original plan is removed as overengineering.

### What it does

Instead of feeding all memory into the agent prompt, score each entry by relevance to the current context and keep only the top-K.

### Score function

```python
def rank_memory_entries(
    entries: list[dict],
    current_columns: set[str],
    current_round: int,
    top_k: int = 15,
) -> list[dict]:
    max_round = current_round
    for e in entries:
        cols = set(e.get("base_columns", []))
        overlap = len(cols & current_columns) / max(len(cols | current_columns), 1)
        recency = 1 / (1 + max_round - e.get("round_idx", max_round))
        gain = max(e.get("value", 0), 0)

        e["_relevance"] = 0.5 * overlap + 0.3 * recency + 0.2 * gain

    entries.sort(key=lambda e: e["_relevance"], reverse=True)
    return entries[:top_k]
```

**Dominant signal is column overlap:** if this dataset has `[age, fare]`, a memory entry about `age` is more relevant than one about `sibsp`.

### Why this helps

| Before | After |
|---|---|
| All entries dumped (~200 lines of text) | Top-15 only (~20 lines) |
| Irrelevant history competes with relevant | Only relevant entries shown |
| LLM must infer which patterns matter | Scoring makes bias explicit |
| Prompt grows linearly with rounds | Prompt is capped at K entries |

### Why other SAGE ideas don't help here

| Idea | Why not | When to revisit |
|---|---|---|
| **Evolution feedback** | LLM handles ~100 entries fine; tracking causal links adds complexity for marginal gain at this scale | Memory >500 entries or >2K prompt tokens |
| **Graph memory** | Flat list + scoring function is equivalent for <1K items | Columns >100 or features >1000 |
| **Schema priors** | Cross-dataset transfer is a research problem | After running on 20+ datasets and seeing patterns |

---

## 5. What Not to Build (Removed from Original Plan)

The following were in v0.1 of this plan and are **removed** after honest assessment:

- ❌ **I3: Self-Evolution Feedback Loop** — marginal benefit at current scale
- ❌ **I4: Cross-Agent Memory Graph** — networkx adds 500+ lines for zero gain
- ❌ **I5: Structured Graph Retrieval** — BFS propagation over 100 nodes is silly
- ❌ **I6: Structural Feature Gating** — no graph topology to speak of
- ❌ **I7: Schema Priors** — cross-dataset transfer not needed yet
- ❌ **I8: Cross-Dataset Transfer Learning** — research-level scope creep

---

## 6. Implementation

**Effort:** ~2 hours for the scoring function + wiring. 1 afternoon.

**New file:**

`src/feature_forge/methods/malmas/memory/retrieval.py`

```python
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
        recency = 1 / (1 + current_round - e.get("round_idx", current_round))
        gain = max(e.get("value", 0), 0)
        score = 0.5 * overlap + 0.3 * recency + 0.2 * gain
        scored.append((score, e))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]
```

**Modified file:**

`src/feature_forge/methods/malmas/memory/base.py` — add a `retrieve` method:

```python
def retrieve_relevant(
    self,
    current_columns: set[str],
    current_round: int,
    top_k: int = 15,
) -> str:
    """Build prompt section from top-K relevant entries."""
    all_entries = self.procedural + self.feedback
    ranked = retrieve_top_k(all_entries, current_columns, current_round, top_k)
    # format ranked entries into prompt text
    ...
```

**Modified file:**

`src/feature_forge/methods/malmas/pipeline/iterative.py` — in `_build_agent_context()`:

```python
memory = self._get_memory(agent.name)
# Before: memory.generate_prompt_section(use_feedback=True)
# After:
memory_context = memory.retrieve_relevant(
    current_columns=set(X_train.columns),
    current_round=round_idx,
    top_k=15,
)
```

---

## 7. Success Metrics

| Metric | Before | After | How to measure |
|---|---|---|---|
| Prompt token usage (memory section) | Grows linearly with rounds | Capped at ~K entries | Count tokens in `generate_prompt_section` vs `retrieve_relevant` |
| Relevancy of injected entries | All entries equally visible | High-gain + column-match entries first | Spot-check top-3 vs bottom-3 entries |
| Feature quality impact | Baseline | Hypothesis: neutral or slightly positive | Compare gain across rounds before/after |

---

## 8. Relationship to SAGE

This is a minimal, honest adaptation of one SAGE idea:

| SAGE concept | Our adaptation |
|---|---|
| Subgraph selection via learned gating `π_e(q)` | Heuristic top-K scoring by column overlap + recency + gain |
| GFM pretrained reader | Not used — no model needed for a scored list |
| Structurally gated propagation | Not used — no graph |
| Writer-reader evolution | Not used — not needed at this scale |
| Context-schema decomposition | Not used — future concern |

---

## 9. When to Revisit

The graph/evolution ideas get valuable when:

- **Memory grows beyond ~500 entries** across multiple datasets
- **Columns exceed ~100** per dataset (graph traversal becomes faster than linear scan)
- **Running on 20+ datasets** and patterns emerge across them
- **Memory prompt exceeds 2K tokens** consistently

Until then, a sorted list is the right tool.
