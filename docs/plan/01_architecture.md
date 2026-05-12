# Architecture & Design Philosophy

## Core Philosophy

`feature_forge` treats **every method as a first-class, independently runnable, composable experiment unit**. We move from "run the pipeline" to "design an experiment matrix."

## Design Principles

### 1. Plugin-Ready Everything
Every agent and baseline is discoverable via Python entry points. This makes the core repo lightweight while allowing research groups to publish extensions as independent pip packages.

```toml
# Downstream package's pyproject.toml
[project.entry-points."feature_forge.agents"]
my_domain_agent = "my_package:DomainAgent"
```

### 2. Experiment-First
Instead of running one pipeline, you define an experiment matrix:

```python
from feature_forge.experiment.matrix import ExperimentMatrix

matrix = (
    ExperimentMatrix()
    .datasets(["titanic", "house-prices"])
    .methods({"malmas_full": ["unary", "cross", "aggregation", "temporal"],
              "malmas_no_memory": [...],
              "openfe": ["openfe"]})
    .seeds([0, 1, 2])
    .models(["xgboost", "lightgbm"])
    .rounds([1, 2, 4])
)
```

### 3. Immutable Configuration
No mutable global state. All configuration is instance-based, validated at startup, and overridable via env vars.

### 4. Security by Default
- Sandboxed code execution (AST validation, restricted builtins)
- LLM response caching enforced by default
- No raw `exec()` without sandbox

### 5. Observable Everything
- Every agent call traced via Langfuse
- Every pipeline step logged via structlog
- Every experiment tracked via WandB
- Costs transparent: token usage → USD per agent per round

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  EXPERIMENT LAYER                                           │
│  - ExperimentMatrix (Cartesian product definitions)         │
│  - ExperimentRunner (execution engine)                      │
│  - ExperimentTracker (WandB/MLflow abstraction)             │
│  - Reporter (auto-generated markdown/HTML reports)          │
├─────────────────────────────────────────────────────────────┤
│  PIPELINE LAYER                                             │
│  - FeatureForge (sklearn-compatible API)           │
│  - CorePipeline (single-round execution)                    │
│  - IterativePipeline (N-round with memory + router)         │
│  - AblationPipelines (no-memory, no-router, single-agent)   │
├─────────────────────────────────────────────────────────────┤
│  AGENT & BASELINE LAYER                                     │
│  - Agent ABC + Registry (entry-point discovery)             │
│  - 6 MALMAS agents (unary, cross, aggregation, ...)         │
│  - RouterAgent (data-driven, performance-driven, hybrid)    │
│  - Baseline ABC + Registry                                  │
│  - OpenFE, CAAFE, LLM-FE baselines                          │
├─────────────────────────────────────────────────────────────┤
│  MEMORY LAYER                                               │
│  - ProceduralMemory (successful transforms)                 │
│  - FeedbackMemory (feature gains/losses)                    │
│  - ConceptualMemory (LLM-summarized rules)                  │
│  - Persistence (JSON/dill serializers)                      │
├─────────────────────────────────────────────────────────────┤
│  LLM LAYER                                                  │
│  - LLMClient ABC (unified interface)                        │
│  - Provider implementations (OpenAI, DeepSeek, Anthropic)   │
│  - DiskCache (enforced default, SHA-256 keyed)              │
│  - LangfuseWrapper (auto-tracing + cost tracking)           │
├─────────────────────────────────────────────────────────────┤
│  EVALUATION LAYER                                           │
│  - Metrics (AUC, ACC, NRMSE, custom)                        │
│  - CV (k-fold cross-validation)                             │
│  - ModelFactory (XGB, LGB, CatBoost, RF, MLP)              │
│  - Sandbox (AST-validated, restricted-builtin execution)    │
├─────────────────────────────────────────────────────────────┤
│  DATA LAYER                                                 │
│  - Dataset ABC + Registry                                   │
│  - KaggleFetcher (primary source)                           │
│  - OpenMLFetcher (secondary)                                │
│  - Sample datasets (for quick testing)                      │
├─────────────────────────────────────────────────────────────┤
│  OBSERVABILITY LAYER                                        │
│  - structlog (JSON in prod, pretty in dev)                  │
│  - OpenTelemetry processor (trace_id/span_id in logs)       │
│  - Langfuse tracer (@observe decorators)                    │
└─────────────────────────────────────────────────────────────┘
```

## Agent System Architecture

```
┌─────────────────┐     uses      ┌─────────────────┐
│  Experiment     │──────────────▶│  Iterative      │
│  Runner         │               │  Pipeline       │
└─────────────────┘               └────────┬────────┘
                                           │
                              ┌────────────┼────────────┐
                              ▼            ▼            ▼
                         ┌────────┐   ┌────────┐   ┌────────┐
                         │ Router │   │ Memory │   │ Eval   │
                         │ Agent  │   │ System │   │ Engine │
                         └───┬────┘   └────────┘   └────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Agent 1  │  │ Agent 2  │  │ Agent N  │
        │ (unary)  │  │ (cross)  │  │ (...)    │
        │ Memory   │  │ Memory   │  │ Memory   │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    ┌──────────────┐
                    │ LLM Client   │
                    │ + Cache      │
                    │ + Langfuse   │
                    └──────────────┘
```

## Data Flow

```
Raw Dataset (Kaggle)
    │
    ▼
Dataset Loader → df_train, df_test, target, metadata
    │
    ▼
Feature Engineering Pipeline
    │
    ├─→ Router selects active agents
    │   ├─→ Each agent: prompt → LLM plan → LLM code → sandbox execution
    │   ├─→ Evaluate each feature via 5-fold CV
    │   ├─→ Update agent memory (procedural, feedback, conceptual)
    │   └─→ Persist top features to df_train/df_test
    │
    ├─→ Global conceptual summary
    │
    └─→ Next round (if Nround > 1)
    │
    ▼
Final Evaluation
    ├─→ Baseline model score (original features)
    ├─→ MALMAS score (original + generated features)
    └─→ Baseline methods scores (OpenFE, CAAFE, LLM-FE)
    │
    ▼
Experiment Tracker (WandB)
    ├─→ Log all metrics, parameters, artifacts
    ├─→ Log LLM costs per agent per round
    └─→ Generate comparison visualizations
```

## Security Model

| Layer | Mechanism |
|-------|-----------|
| Code Execution | AST parsing + forbidden names whitelist + restricted builtins |
| LLM Calls | DiskCache enforced, no uncached execution by default |
| Secrets | dotenvx encrypted `.env`, never committed |
| Imports | No dynamic imports in sandboxed code |
| File System | No `open()`, no file operations in sandbox |

## Concurrency Model

- **Agent-level parallelism**: `asyncio.gather()` for selected agents per round
- **Experiment-level parallelism**: `ProcessPoolExecutor` for independent experiment combinations
- **LLM calls**: Async with semaphore-based rate limiting
- **Memory access**: Per-agent memory is isolated (no shared state)
