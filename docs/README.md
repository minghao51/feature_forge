# Documentation Map

Entry point for navigating Feature Forge documentation, referenced by
`AGENTS.md` ("Start from the documentation map"). Authority levels follow
the decision order in `AGENTS.md`.

## Authoritative (decision order 1–2)

| Location | Role |
|----------|------|
| `README.md` | Package contract: features, install, quick start |
| `docs/plan/00_index.md` | Active implementation plan index (decision order 1, with `README.md`, per `AGENTS.md`) |
| `docs/decisions/` | ADRs — accepted decisions (see its `README.md`) |

## Exploratory (decision order 3)

### Implementation plan documents (`docs/plan/`)

Descriptions follow the plan-structure table in `docs/plan/00_index.md`
(authoritative, listed above); status flags are taken from each document's
header. Plan docs 01–21 are sub-ADR context: `AGENTS.md` enumerates only
`00_index.md` among them explicitly; where they conflict with an accepted
ADR, the ADR wins. (The earlier `01`/`02` number collision —
`01_consolidate_notebooks.md` and `02_llm_code_parsing_research.md` — was
resolved on 2026-08-29 by archiving both to `docs/archive/`.)

| Document | Description |
|----------|-------------|
| `docs/plan/01_architecture.md` | High-level architecture and design philosophy (every method a first-class experiment unit) |
| `docs/plan/02_directory_structure.md` | Complete target directory layout |
| `docs/plan/03_key_design_decisions.md` | Configuration (pydantic-settings), caching, sandboxing, plugin system |
| `docs/plan/04_implementation_phases.md` | 13-phase implementation roadmap |
| `docs/plan/05_dependencies.md` | `pyproject.toml` specification and dependency choices |
| `docs/plan/06_data_strategy.md` | Kaggle-first data ingestion strategy |
| `docs/plan/07_observability.md` | structlog + Langfuse + OpenTelemetry observability design |
| `docs/plan/08_experiment_tracking.md` | WandB + MLflow experiment-tracking abstraction |
| `docs/plan/09_baseline_selection.md` | Why MALMAS + OpenFE + CAAFE + LLM-FE were selected as the method set |
| `docs/plan/10_experimental_platform_refactor.md` | Hybrid plugin platform refactor: `ExperimentalPlatform` API + entry point hooks for baselines/datasets/models/metrics (phases 1–3 done; tests and export CLI outstanding per index) |
| `docs/plan/11_platform_refactor_review.md` | Post-implementation review of the platform refactor, gaps identified, fixes applied — header status: completed |
| `docs/plan/12_code_generation_improvements.md` | Code generation reliability improvements and schema enforcement |
| `docs/plan/13_methods_restructure.md` | Restructure methods into the plugin-based `methods/` architecture — header status: completed, with an implementation note on deviations |
| `docs/plan/14_parallelize_code_generation.md` | Parallelize per-agent code generation across agents — header status: draft |
| `docs/plan/15_notebook_updates_handoff.md` | Notebook updates and documentation handoff after the methods restructure — **session handoff doc** ("Ready for Implementation"), tied to `13_methods_restructure.md` |
| `docs/plan/16_prompt_colocation_pydantic.md` | YAML prompt colocation + Pydantic model migration for prompt rendering |
| `docs/plan/17_code_simplification.md` | Full `src/` code simplification pass: dedup sandbox init, iterative helpers, parse guards, router conditions, registry discovery |
| `docs/plan/18_sage_inspired_memory_evolution.md` | SAGE-inspired memory evolution: graph memory, structured retrieval, self-evolution loop, cross-dataset transfer — header status: updated after honest assessment |
| `docs/plan/19_ux_improvements.md` | UX improvements: CLI tool, config wizard, progress display, dashboard, feature explorer |
| `docs/plan/20_long_term_roadmap.md` | Combined long-term plan: founding vision → realized state → validation frontier + 5 themes, each gated on a `docs/deferred-design.md` trigger. |
| `docs/plan/21_hamilton_default_execution_handoff.md` | Implemented Hamilton-default execution handoff: PRs 1–6, technical removal gates, and the ADR 0016 legacy-removal change complete; `experiments/legacy_removal/2026-09-14/` remains the historical parity record. |
| `docs/plan/22_fail_fast_cancellation_contract.md` | Implemented plan for continue/fail-fast scheduling and cooperative case-boundary cancellation under accepted ADR 0017. |
| `docs/plan/23_evaluation_integrity_security_hardening.md` | Proposed audit-remediation plan for independent evaluation, Platinum policy enforcement, sandbox containment, bounded workers, and LLM cache identity v2. |

Plans 18–23 are published as nav entries under *Implementation Plan*. Plan 21
remains the completed execution-runtime specification; plan 22 is the active
scheduler handoff. Plans 18–20 retain their exploratory research and roadmap
roles.

### Other exploratory documents (`docs/`)

| Location | Role |
|----------|------|
| `docs/methods.md` | Method descriptions and comparisons (also published) |
| `docs/MALMAS_Technical_Roadmap.md` | Founding vision (Apr 2026), reclassified as provenance — see its in-file header. Realized architecture lives in `docs/plan/` + `docs/decisions/`; forward plan in `docs/plan/20_long_term_roadmap.md`. |
| `docs/deferred-design.md` | Deferred research parking lot with revival triggers |

## Archive (decision order: none)

| Location | Role |
|----------|------|
| `docs/archive/` | Superseded/deprecated docs, kept verbatim; never guidance |
| `docs/archive/01_consolidate_notebooks.md` | Archived 2026-08-29 from `docs/plan/01_consolidate_notebooks.md` — completed one-off notebook-consolidation plan (work shipped). |
| `docs/archive/02_llm_code_parsing_research.md` | Archived 2026-08-29 from `docs/plan/02_llm_code_parsing_research.md` — LLM code-parsing options analysis; approach shipped via plan 12/17. |

## Agent-facing and logs (repo root)

| Location | Role |
|----------|------|
| `AGENTS.md` | Agent behavior rules and decision order |
| `REPORT_LOG.md` | Material findings log + AI-assistance disclosure |
| `.planning/OVERVIEW.md` | Architecture quick context |
| `.planning/STATE.md` | Current status and risks |
| `.planning/STYLE.md` | Coding conventions and workflow |

## Published site (mkdocs)

User-facing pages are built by mkdocs; the nav in `mkdocs.yml` defines
what is published. This map, `docs/decisions/`, `docs/deferred-design.md`,
and `docs/archive/` are internal working documents and are intentionally not
in the nav.

| Nav page (file) | Content |
|-----------------|---------|
| Home (`docs/index.md`) | Landing page: project overview, key features, badges, links |
| Quick Start (`docs/quick_start.md`) | Installation (`uv`/pip) and first steps with the sklearn-compatible API |
| API Key Setup (`docs/API_KEY_SETUP.md`) | Setting the DeepSeek API key via env vars, dotenvx-encrypted `.env`, or `settings.yaml` |
| Methods (`docs/methods.md`) | Every method and pipeline (MALMAS core, OpenFE, CAAFE, LLM-FE, Malmus), agent/memory/router internals, full reference list |
| API Reference (`docs/api_reference.md`) | Core classes (`FeatureForge`, platform, config) with parameters and examples |
| Migration Guide (`docs/migration_guide.md`) | Porting from the MALMAS research codebase: config, imports, API mapping |
| Operations (`docs/operations.md`) | Hamilton cache inspection/retention, artifact verification, and recovery guidance |
| Sandbox Isolation Spike (`docs/spikes/2026-09-15-sandbox-os-isolation.md`) | Time-boxed OS-isolation spike record for plan 23 §5.2 / ADR 0019: probe results (Landlock ABI 7, userns, seccomp, bwrap), live sentinel-denial demo, and the strict-mode containment recommendation |
| Generated Stage DAGs (`docs/generated/stage_dags.md`) | Freshness-tested Bronze/Silver/Gold/Platinum DAGs and durable stage boundaries |
| Implementation Plan (`docs/plan/`) | `00_index.md`, `MALMAS_Technical_Roadmap.md` (founding vision, reclassified as provenance), plus `01_architecture.md` through `23_evaluation_integrity_security_hardening.md` published as separate nav entries |
| Notebooks (`docs/notebooks/`) | `index.md` overview + five rendered tutorials: `01_getting_started.ipynb` (offline intro), `02_pipeline_deep_dive.ipynb` (MALMAS agents/router/memory), `03_benchmarks_and_artifacts.ipynb`, `04_custom_method.ipynb` (writing a custom method), `05_methods_deep_dive.ipynb` (CAAFE, LLM-FE, OpenFE, Malmus). The `index.md` table itself only describes 01–03 |

The completed Hamilton implementation handoff is
`docs/plan/21_hamilton_default_execution_handoff.md`; the active scheduler plan
is `docs/plan/22_fail_fast_cancellation_contract.md`; the active audit remediation
is `docs/plan/23_evaluation_integrity_security_hardening.md`, and later research themes
live in `docs/plan/20_long_term_roadmap.md`. The founding
`MALMAS_Technical_Roadmap.md` is nested under *Implementation Plan* as provenance.

## Maintenance rules

- Superseded docs move to `docs/archive/` (verbatim + archival header),
  they are never deleted or left in place as stale guidance.
- Deferred ideas go to `docs/deferred-design.md` with a revival trigger,
  not into new plan documents.
- Accepted decisions go to `docs/decisions/` as ADRs.
