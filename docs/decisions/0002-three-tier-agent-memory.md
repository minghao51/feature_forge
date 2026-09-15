# ADR 0002: Three-tier agent memory for MALMAS

- **Status:** Accepted
- **Date:** 2026-08-29
- **Related:** `docs/plan/01_architecture.md` (memory layer); ADR 0001

## Context

MALMAS gains its advantage over single-shot methods by accumulating
experience across rounds: what worked, what was measured, and what was
learned. A single flat log of past features blows the LLM context budget
and gives the model no structure to reason over; embedding-based
retrieval was rejected as premature under ADR 0001's smallest-architecture
preference. The MALMAS paper's tiered design maps cleanly onto bounded,
inspectable Python structures.

This record was written retroactively on 2026-08-29, after
implementation; see Related for the originating plan documents.

## Decision

Memory is organized as three tiers inside a per-agent `AgentMemory`
(`src/feature_forge/methods/malmas/memory/`):

1. **Per-agent isolation with three tiers.** Each of the six MALMAS agents
   owns a private `AgentMemory` holding procedural memory (successful
   transforms, plus an `unused_procedural` list for ineffective ones),
   feedback memory (per-feature evaluation records plus mechanical
   statistics), and conceptual memory (LLM-distilled rule strings and the
   latest conceptual summary). No shared mutable state between agents.
2. **Global conceptual distillation.** `ConceptualMemory` summarizes each
   agent's experience into conceptual rules (`summarize_agent`) and
   distills a cross-agent global summary once per round
   (`summarize_global`), both via the shared `LLMClient`.
3. **Heuristic relevance retrieval.** Memory reaches prompts through
   `retrieve_top_k`, which ranks entries by 50% column overlap, 30%
   recency, and 20% measured gain (top-k, default 15) — mechanical
   scoring, not embeddings.
4. **Bounded, persistent memory.** State persists as plain JSON via
   `MemoryPersistence` under `MemoryConfig.persistence_dir`; tier sizes
   are capped by `MemoryConfig.max_size` with oldest-first trimming
   applied on record and save.
5. **Ablatable.** `NoMemoryPipeline` exists specifically to measure the
   memory system's contribution, alongside the other ablation pipelines.

## Consequences

- Prompt context stays bounded and structured, and the memory/no-memory
  comparison is a first-class experiment, not an afterthought.
- Retrieval is lexical and recency-biased; it will miss semantically
   related but lexically distant transforms. Moving to embedding-based
   (RAG-style) retrieval is a boundary change that requires a superseding
   ADR per ADR 0001.
- Memory JSON is written unencrypted to disk — a known, accepted
  low-severity exposure of feature-engineering logic.
- Trimming is per-tier and oldest-first, so very long runs silently drop
  history; raising `max_size` trades context budget for retention.
