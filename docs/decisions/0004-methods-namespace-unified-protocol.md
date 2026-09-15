# ADR 0004: Methods namespace, unified protocol, and sklearn-compatible public API

- **Status:** Accepted
- **Date:** 2026-08-29
- **Related:** `docs/plan/13_methods_restructure.md`;
  `docs/plan/09_baseline_selection.md`; ADR 0001

## Context

Before the restructure, MALMAS components were scattered across
`agents/`, `pipeline/`, `memory/`, and `prompts/`, while CAAFE, LLM-FE,
OpenFE, and Malmus lived in `baselines/` — a name that undersold them as
secondary despite being the comparison targets selected in
`docs/plan/09_baseline_selection.md`. MALMAS (`FeatureForge`) and the
baselines also had separate APIs, so they were not interchangeable in
experiments. Adding a method forced the contributor to decide "is it an
agent or a baseline?" — the wrong question.

This record was written retroactively on 2026-08-29, after
implementation; see Related for the originating plan documents.

## Decision

1. **One namespace of peers.** Every feature-engineering method lives as
   a self-contained sub-package under `src/feature_forge/methods/`
   (`malmas`, `caafe`, `llmfe`, `openfe`, `malmus`). MALMAS-internal
   components nest under `methods/malmas/` (`agents/`, `pipeline/`,
   `memory/`, `prompts/`).
2. **One protocol.** `MethodProtocol` in `methods/base.py`
   (`fit` / `transform` / `fit_transform` / `generated_scripts` /
   `feature_metadata` / `get_artifacts`) is the contract for every
   method; `MethodRegistry` discovers via the
   `feature_forge.methods` entry-point group with built-in fallbacks
   (agents: `feature_forge.methods.malmas.agents`). This refines ADR
   0001 rule 1 — the registry is the only extension point, and all
   methods are interchangeable in `ExperimentalPlatform`.
3. **Sklearn-compatible public API.** The package's public entry point is
   the `FeatureForge` estimator in `api.py`
   (`BaseEstimator` + `TransformerMixin` + `ArtifactExporter`), so
   MALMAS composes with sklearn `Pipeline` and model selection.
   `MALMASMethod` is a thin adapter that wraps `FeatureForge` to satisfy
   `MethodProtocol`; baselines implement the protocol directly.
4. **Clean break.** The old top-level `agents/`, `baselines/`,
   `pipeline/`, `memory/`, and `prompts/` directories were removed with
   no re-exports; terminology follows the namespace
   (`BaseMethod`, `*Method`, not `Baseline`).

## Consequences

- "Agent vs baseline" is no longer a contributor decision — everything
  is a method; MALMAS is just the one with internal agents.
- The sklearn contract keeps `fit`/`transform` semantics honest (e.g.
  `transform` reproduces exactly the features selected during `fit`) and
  makes methods droppable into standard sklearn workflows.
- The clean break broke every import path once, deliberately; migration
  is documented in `docs/migration_guide.md` and external plugins must
  target the new entry-point groups.
- Two layers (`FeatureForge` estimator + `MALMASMethod` adapter) must be
  kept API-aligned when either evolves.
