# Feature Forge Implementation Plan

**Version:** 0.1.0  
**Date:** May 2026  
**Status:** Draft  

## Overview

This document outlines the comprehensive implementation plan for `feature_forge`, a modular experimentation platform for LLM-based multi-agent automated feature engineering. It is designed to systematically break down, compare, and optimize feature engineering methods — starting with the MALMAS architecture and its competitive baselines.

## Goals

1. **Reproduce and optimize** the MALMAS paper results on standard tabular datasets
2. **Enable isolated experimentation** of every component (agents, memory, router, baselines)
3. **Provide sklearn-compatible APIs** for drop-in adoption
4. **Track everything** — experiments, LLM costs, feature quality, with WandB and Langfuse
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
| Tracking backend | **WandB default**, MLflow optional | Superior visualization, free academic tier, W&B Weave for LLM |
| Observability | **Langfuse cloud** | Zero infra overhead, hierarchical tracing, prompt management |
| Logging | **structlog** | 2x faster than stdlib, JSON in prod, pretty in dev, OTel integration |
| Baselines | **OpenFE + CAAFE + LLM-FE** | Top 3 non-MALMAS methods per 2026 rankings |
| Package manager | **uv** | Modern, fast, deterministic with `uv.lock` |
| Layout | **src/** | Tests run against installed package |

---

## Next Steps

1. Review all plan documents in `docs/plan/`
2. Approve or modify Phase 1 scope
3. Begin implementation with `uv init` and directory scaffolding
