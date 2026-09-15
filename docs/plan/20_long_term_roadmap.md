# Feature Forge — Long-Term Roadmap

> **Status:** living document · 2026-08-29 · exploratory (decision order 3).
> Accepted decisions live in `docs/decisions/`; this doc proposes direction only.
> Combines the founding vision (`docs/MALMAS_Technical_Roadmap.md`, now
> reclassified as provenance), the realized architecture
> (`docs/plan/00_index.md`, `docs/decisions/0001-0007`, `.planning/STATE.md`),
> and the deferred research ledger (`docs/deferred-design.md`).
>
> **Implementation update (2026-09-09):** Claims below that the library is
> complete describe the pre-Hamilton baseline. The active execution-runtime
> work is `docs/plan/21_hamilton_default_execution_handoff.md`; this document
> remains the roadmap for later evidence-gated research themes.

## 1. Where we are — realized foundation

`feature_forge` v0.1.0 is **library-complete**: every capability in
`.planning/STATE.md` is "Full". The founding roadmap's four asks are shipped:

| Founding ask (Apr 2026) | Realized in `feature_forge` |
|-------------------------|-----------------------------|
| Proper package + `pyproject.toml` | `src/feature_forge/` (hatchling), `pyproject.toml` |
| sklearn-compatible API | `FeatureForge(BaseEstimator, TransformerMixin)` (`api.py`) |
| `LLMClient` abstraction | `llm/base.py` ABC → 4 providers + `DiskCache` |
| Sandboxed execution | `evaluation/sandbox.py` (AST + process isolation) |

Plus: `ExperimentalPlatform` (cartesian experiment matrix), 5 methods
(`malmas`, `caafe`, `llmfe`, `malmus`, `openfe`), 6 agents + router, 3-tier
memory, entry-point plugin system, Langfuse/OTel wiring, pydantic-settings
config. Governance now includes accepted ADRs through 0016; ADR 0016 authorizes
legacy-engine removal after Hamilton qualification, while plan 22 requires ADR
0017 before fail-fast/cancellation implementation. The smallest-architecture
rule remains ADR 0001.

**Default LLM backend switched to OpenCode Go `hy3`** (2026-08-29) — an
OpenAI-compatible gateway over a curated model pool; model is overridable per
run via `FF_LLM__MODEL` (see `docs/API_KEY_SETUP.md`).

## 2. The real gap — research code → validated method

The library is built; what is missing is **evidence**. The active frontier is
`experiments/`:

| Suite | Question it answers | Gates |
|-------|---------------------|-------|
| `memory_ablations/` | Does memory help, and *where*? | D-1 (memory evolution) |
| `router_strategies/` | Which router strategy wins? | informs method-ranking |
| `agent_isolation/` | What does each agent contribute? | method-ranking |

These produce the method-ranking claims the founding roadmap promised.
Statistical power is currently capped at **2 datasets** (titanic, house_prices)
— cross-dataset claims need D-5 (dataset-registry growth) to reach 3+.

## 3. Five themes (each gated on a trigger)

> Scope discipline (ADR 0001): nothing here is built until its trigger fires
> **and** an ADR is accepted for any boundary change.

**Theme 1 — Validate & publish the science**
- Finish + write up the three ablation suites → a paper-ready leaderboard.
- Cross-dataset ranking unlocks when D-5 lands (3+ datasets).
- Uses the existing harness; no new architecture.

**Theme 2 — Reproducibility & provenance**
- Promote **D-6** (Langfuse round/agent spans), **D-7** (OTel trace⇄log
  correlation), **D-8** (tracker-native sweeps) when a debugging session or
  tuning study hits their stated triggers.
- Enforced `DiskCache` (ADR 0001 #2) stays the reproducibility backbone.

**Theme 3 — Surface, don't expand (UX, D-2)**
- `forge` CLI, config wizard, real-time progress, memory inspector, quick-CSV,
  actionable errors (plan 19: U1–U4, U6, U8, U9).
- Highest *adoption* leverage; gated on a recorded onboarding-friction event.
- Deliberately **not** a web UI yet (D-2; CLI covers ~90%).
- Requires a new ADR using the next available number — a new public CLI surface
  is a boundary change.

**Theme 4 — Memory & method evolution (D-1)**
- Keep 3-tier + top-K retrieval (already shipped in `memory/retrieval.py`).
- Graph/evolution machinery only on triggers: memory >500 entries, >100
  columns, 20+ datasets, or memory prompt >2K tokens.

**Theme 5 — Scale the experiment surface (D-4 / D-5)**
- Multi-table ingestion + dataset-registry growth only when an ablation needs
  3+ datasets or the large Phase-1 datasets (Porto Seguro / Santander).

## 4. Deliberately NOT doing (scope discipline)

| Idea | Why not (yet) | Revival |
|------|---------------|---------|
| PydanticAI migration | Portability solved via OpenCode Go / OpenAI-compatible pool; rewrapping risks experiment comparability | decided — see ADR bindings |
| Web UI | D-2; CLI covers 90% at lower cost | D-2 trigger |
| SAGE graph / evolution | D-1; flat list ≡ graph at current scale | D-1 triggers |
| MLflow parity | D-3; WandB covers needs | D-3 trigger |
| OTel export | D-7; single-process runs, Langfuse sufficient | D-7 trigger |
| New LLM-provider rewrapping | OpenCode Go pool covers model variety | — |

## 5. Governance & decision ledger

- **Every theme's implementation needs an ADR before code** (AGENTS.md rule 5).
  Assign the next available number when a trigger fires; ADRs through 0016 are
  accepted, and 0017 is reserved for plan 22's fail-fast/cancellation contract.
- Revival ledger = `docs/deferred-design.md` (D-1…D-8). Promote by editing the
  entry to "promoted → ADR 00XX / shipped in PR #".
- Already archived (superseded/completed): `docs/archive/01_consolidate_notebooks.md`,
  `docs/archive/02_llm_code_parsing_research.md`.

## 6. Suggested sequencing (12-month shape, conditional)

- **Q1** — finish ablations (Theme 1) → first method-ranking claims.
- **Q2** — D-5 dataset growth *if* claims need power; D-2 UX *if* friction
  recorded.
- **Q3** — Theme 2 provenance depth *if* a debugging session needs agent-level
  cost/latency attribution.
- **Q4** — Theme 4/5 *only* if their triggers fire.

Each step is conditional; nothing here is committed without an ADR and a
fired trigger. The founding ambition (a production-grade, validated
multi-agent feature-engineering library) is unchanged — this roadmap just
replaces its 4-week plan with evidence-gated, architecture-preserving increments.
