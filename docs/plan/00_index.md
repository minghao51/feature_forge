# Feature Forge Implementation Plan

**Version:** 0.1.0
**Date:** 2026-09-09
**Status:** Active

## Overview

This document outlines the comprehensive implementation plan for `feature_forge`, a modular experimentation platform for LLM-based multi-agent automated feature engineering. It is designed to systematically break down, compare, and optimize feature engineering methods — starting with the MALMAS architecture and its competitive baselines.

## Goals

1. **Reproduce and optimize** the MALMAS paper results on standard tabular datasets
2. **Enable isolated experimentation** of every component (agents, memory, router, baselines)
3. **Provide sklearn-compatible APIs** for drop-in adoption
4. **Track experiments reproducibly** — optional WandB/MLflow backends plus
   local structured logs and artifacts; external tracking defaults off
5. **Support dynamic data ingestion** from Kaggle (starting simple, scaling to multi-table)

## Plan Structure

| Document | Purpose |
|----------|---------|
| `01_architecture.md` | High-level architecture and design philosophy |
| `02_directory_structure.md` | Complete directory layout |
| `03_key_design_decisions.md` | Configuration, caching, sandboxing, plugin system |
| `04_implementation_phases.md` | 13-phase implementation roadmap |
| `05_dependencies.md` | `pyproject.toml` specification |
| `06_data_strategy.md` | Kaggle-focused data ingestion strategy |
| `07_observability.md` | structlog + Langfuse + OpenTelemetry |
| `08_experiment_tracking.md` | WandB + MLflow abstraction |
| `09_baseline_selection.md` | Why MALMAS + OpenFE + CAAFE + LLM-FE |
| `10_experimental_platform_refactor.md` | Hybrid plugin platform refactor: `ExperimentalPlatform` API + entry point hooks for baselines/datasets/models/metrics |
| `11_platform_refactor_review.md` | Post-implementation review, gaps identified, and fixes applied |
| `12_code_generation_improvements.md` | Code generation improvements and schema enforcement |
| `13_methods_restructure.md` | Restructure methods into plugin-based architecture |
| `14_parallelize_code_generation.md` | Parallel code generation across agents |
| `15_notebook_updates_handoff.md` | Notebook updates and documentation handoff |
| `16_prompt_colocation_pydantic.md` | YAML prompt colocation + Pydantic model migration |
| `17_code_simplification.md` | Full `src/` code simplification — dedup sandbox init, iterative helpers, parse guards, router conditions, registry discovery |
| `18_sage_inspired_memory_evolution.md` | SAGE-inspired memory evolution: graph memory, structured retrieval, self-evolution loop, cross-dataset transfer |
| `19_ux_improvements.md` | UX improvements: CLI tool, config wizard, progress display, dashboard, feature explorer |
| `20_long_term_roadmap.md` | Evidence-gated living roadmap for later research themes |
| `21_hamilton_default_execution_handoff.md` | Agent-ready implementation specification for Hamilton-default case execution, medallion evidence, and default-on DAG caching |
| `22_fail_fast_cancellation_contract.md` | Implemented plan (PRs 1–4) for continue/fail-fast scheduling, cooperative case-boundary cancellation, typed cancelled results, and lifecycle evidence |
| `23_evaluation_integrity_security_hardening.md` | Proposed remediation plan for independent evaluation, enforceable Platinum selection, uncertainty correctness, sandbox containment, bounded worker lifecycle, and LLM cache identity v2 |

## Research Basis

This plan is informed by:
- **Google AI Search** (May 2026): LLM-based AFE architecture best practices, WandB vs MLflow comparison, Langfuse multi-agent observability, structlog best practices
- **Context7 Documentation**: wandb, mlflow, langfuse-python, structlog official docs
- **MALMAS Technical Roadmap** (`docs/MALMAS_Technical_Roadmap.md`): Current state assessment and refactoring recommendations
- **MALMAS Codebase Analysis** (`@/Users/minghao/Desktop/personal/MALMAS`): Deep dive into existing methods, agents, baselines
- **python-project-structure skill**: pydantic-settings, YAML config, dotenvx secrets
- **python-tooling skill**: uv, ruff, pytest, pre-commit, CI/CD

## Quick Start Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dataset source | **Kaggle** | Real-world datasets, clear path to multi-table complexity |
| Experiment config | **Python-first**, YAML supported | Flexibility for researchers, declarative option for reproducibility |
| LLM caching | **Enforced default ON** | Prevent accidental API costs; explicit opt-out only |
| Tracking backend | **None by default**, WandB/MLflow optional | Safe local execution; external tracking is an explicit experiment choice per ADR 0006 |
| Observability | **Langfuse cloud** | Zero infra overhead, hierarchical tracing, prompt management |
| Logging | **structlog** | 2x faster than stdlib, JSON in prod, pretty in dev, OTel integration |
| Baselines | **OpenFE + CAAFE + LLM-FE** | Top 3 non-MALMAS methods per 2026 rankings |
| Package manager | **uv** | Modern, fast, deterministic with `uv.lock` |
| Layout | **src/** | Tests run against installed package |

---

## Current Phase

**Hamilton-default execution is implemented.** PRs 1–6 are complete in the
working tree: verified stage execution, default-on cache/telemetry operations,
MALMAS correctness fixes, generated/user documentation, and a core-only package
contract are present. Technical legacy-removal gates 1–5 have reproducible
evidence, and the ADR 0016 removal change has landed: Hamilton is the sole
execution engine, stale legacy configuration fails fast with migration
guidance, and the parity evidence is preserved under
`experiments/legacy_removal/2026-09-14/`.

**Plan 22 (fail-fast and cooperative cancellation, ADR 0017) is implemented —
PRs 1–4 are landed and the final validation sweep is complete.** The typed contracts,
sequential fail-fast scheduling, bounded process submission, and the operator
documentation/completion audit are in place; see `docs/operations.md`
("Execution failure policy and cancellation"). The case-retry gap
(`ResourceConfig.max_attempts`) remains explicitly dormant.

## Next Steps

1. ✅ PR 1: authoritative ADRs, configuration, and dependency contract.
2. ✅ PR 2A–2C foundations: contracts, storage, identity, and resume.
3. ✅ PR 3A–3E: stage DAGs, Platinum decomposition, and cache qualification.
4. ✅ PR 4: first-party `HamiltonLayerExecutor` and atomic default switch.
5. ✅ PR 5: operations, telemetry, CLI, and user documentation.
6. ✅ PR 6: legacy correctness and dependency simplification.
7. ✅ Collect representative experiment and technical legacy-removal evidence.
8. ✅ Gate 6 maintainer decision: ADR 0016 accepts legacy-engine removal.
9. ✅ Remove `engine=legacy` in a dedicated change — landed 2026-09-14 per
   ADR 0016; Hamilton is the sole engine and stale legacy configuration fails
   with migration guidance.
10. ✅ Implement `22_fail_fast_cancellation_contract.md` — PRs 1–4 landed
    2026-09-14 (ADR 0017 accepted; contracts, sequential fail-fast, bounded
    process scheduling, docs + completion audit), pending only final
    validation. The dormant `ResourceConfig.max_attempts` retry fields remain
    an explicit open gap, not part of this plan.
11. ✅ Final validation sweep for the plan 22 slice complete (2026-09-14):
    full suite 1000 passed / 9 expected optional-XGBoost skips on Python 3.11,
    3.12, and 3.13 under single-thread BLAS/OpenMP limits (both mp contexts
    exercised in-suite); mypy src+tests, ruff, format, hygiene, docs
    references, stage-DAG freshness, mkdocs strict, `uv lock --check`, and
    `git diff --check` all green. Then use
    `20_long_term_roadmap.md` for later research themes.
12. In progress: `23_evaluation_integrity_security_hardening.md` — PR 1 landed
    2026-09-15 (40 characterization tests for all nine confirmed findings —
    37 `xfail(strict=True)` + 3 pins; OS-isolation spike proving Landlock
    ABI 7 containment on this Linux host (WSL2, kernel 6.18); ADRs 0018–0020
    drafted and **accepted by the maintainer on 2026-09-15**). PR 2
    (partition-aware discovery) landed 2026-09-16 per ADR 0018 decisions
    1–3: typed `protocol`/`evaluation_holdout_fraction` settings, fail-closed
    holdout partitioning, executor-threaded protocol, discovery-only method
    fitting, the row-local scope contract, and protocol-aware Silver/Gold/
    Platinum fingerprints. PR 3 (fold-local preprocessing and selection)
    landed 2026-09-16 per ADR 0018 decisions 4–6: fold-local
    preprocessing with the documented -1 unknown-category sentinel and
    retained specifications, discovery-fold candidate/selection evidence,
    greedy forward selection enforcing every selection-policy field,
    evaluation-fold two-arm reporting with rejected winners mirroring
    baseline, and PlatinumRequest partition/profile validation. PR 4
    (Platinum v2 evidence and uncertainty) landed 2026-09-18 per ADR 0018
    decisions 7–8: directional Student-t intervals (scipy direct dep), the
    12-artifact evidence schema v2 (evidence.json marker, discovery fold
    metrics, selection steps, preprocessing identity) with offline
    reconstruction and tamper detection in the loader, Platinum-only
    `PLATINUM_IDENTITY_SCHEMA_VERSION="2"` reuse rejection (v1 stays
    readable, never reusable), and directional gain/bounds + protocol
    labeling in results and tracker output. PR 5 (sandbox containment and
    bounded worker lifecycle) landed 2026-09-18 per ADR 0019: expanded AST
    I/O policy (direct + aliased NumPy/Pandas file APIs via normalized-path
    and terminal-name matching), raw-ctypes Landlock strict enforcement
    with exact input/output inode grants and an empty-allow TCP policy
    (seccomp fallback below ABI 4, fail-closed when unavailable), an
    explicit `degraded_development` profile with provenance, and a bounded
    sync/async worker lifecycle (event-loop-owned spawn, one monotonic
    deadline, process-group kill, leak-free cleanup). **Next: PR 6 (LLM
    cache identity v2, ADR 0020), then PR 7** in isolated
    changes per the plan sequence.
    Evaluation correctness precedes new benchmark claims; strict sandbox
    containment precedes restoring unqualified production-readiness claims.
