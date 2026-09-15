# ADR 0003: Dynamic agent router with selectable strategies

- **Status:** Accepted
- **Date:** 2026-08-29
- **Related:** `docs/plan/01_architecture.md` (agent system); ADR 0002

## Context

Running all six MALMAS agents in every round wastes LLM budget on agents
that cannot help the dataset at hand, while always running a fixed subset
gives up the adaptive behavior that motivates the multi-agent design.
The router decides which agents are active each round; its selection
policy must be measurable (ablatable) and configurable without code
changes, because strategy comparison is itself an experiment axis.

This record was written retroactively on 2026-08-29, after
implementation; see Related for the originating plan documents.

## Decision

Agent selection is owned by `RouterAgent`
(`src/feature_forge/methods/malmas/agents/router.py`) behind four
selectable strategies:

1. **Four strategies, one interface.** `data_driven` (dataset
   characteristics), `performance_driven` (historical per-agent gains
   fed by feedback memory, ADR 0002), `hybrid` (union of both — the
   default), and `llm` (LLM-decided selection).
2. **Declarative configuration.** Strategy, `max_agents`/`min_agents`
   bounds, and `warmup_rounds` are set via `RouterConfig`
   (pydantic, YAML/env-overridable); during warmup rounds all agents
   are active so performance signals can bootstrap.
3. **Closed feedback loop.** The pipeline reports per-agent feature
   gains after each round via `update_performance()`; the router's
   selections co-evolve with measured outcomes across rounds.
4. **Ablatable.** `NoRouterPipeline`, `SingleAgentPipeline`, and
   `NoMemoryStaticRouterPipeline` isolate the router's contribution from
   memory's (ADR 0002).

## Consequences

- Adaptive agent activation cuts LLM cost per round while keeping the
  multi-agent design intact, and strategy choice is a tunable experiment
  variable rather than a code change.
- `performance_driven` selection is only as good as the feedback memory
  feeding it; early rounds depend on warmup to avoid cold-start
  degenerate selections.
- Hybrid (union) selection can activate many agents on wide datasets —
  `max_agents` is the guardrail and defaults to unbounded.
- Each additional strategy multiplies the ablation matrix; new
  strategies must go through `RouterConfig` and stay selectable, not
  hard-wired into pipelines.
