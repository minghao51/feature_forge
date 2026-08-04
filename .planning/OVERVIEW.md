# Feature Forge — Overview

## Architecture

**Pattern:** Layered Python library — plugin-based method registry with experiment harness

```
┌─────────────────────────────────────────────────────────┐
│  ExperimentalPlatform / FeatureForge (sklearn API)      │
├─────────────────────────────────────────────────────────┤
│  Experiment Layer  │ matrix → runner → case_executor    │
├─────────────────────────────────────────────────────────┤
│  Methods Layer     │ MethodRegistry → BaseMethod        │
│                    │ malmas / caafe / llmfe / openfe /  │
│                    │ malmus                              │
├──────────────┬─────┴─────────────────────────────────────┤
│  Agent Layer │ 6 agents + router (MALMAS only)          │
├──────────────┴───────────────────────────────────────────┤
│  Memory Layer    │ procedural / feedback                 │
├──────────────────────────────────────────────────────────┤
│  LLM Layer       │ LLMClient → providers → DiskCache    │
├──────────────────────────────────────────────────────────┤
│  Evaluation      │ CVEvaluator → ModelFactory → Sandbox │
├──────────────────────────────────────────────────────────┤
│  Data            │ DatasetRegistry → ingestion          │
├──────────────────────────────────────────────────────────┤
│  Observability   │ structlog + Langfuse + OpenTelemetry │
├──────────────────────────────────────────────────────────┤
│  Config          │ pydantic-settings (YAML + env + .env)│
└──────────────────────────────────────────────────────────┘
```

### Core Library — Python (Python 3.11+)

Layered: **Plugin-registry pattern with entry points for methods, agents, metrics, models, and datasets**

| Layer | Location | Pattern |
|-------|----------|---------|
| Public API (sklearn) | `src/feature_forge/api.py` | `FeatureForge(BaseEstimator, TransformerMixin)` — fit/transform with pipeline dispatch |
| Platform API | `src/feature_forge/platform.py` | `ExperimentalPlatform` facade — method comparison with cartesian matrix |
| Experiment harness | `src/feature_forge/experiment/` | `ExperimentalPlatform` → `ExperimentCaseExecutor` — sequential or process-pool; tracker via `create_tracker_from_config` |
| Durable contracts | `src/feature_forge/contracts/` | Versioned Bronze/Silver/Gold/Platinum, orchestration, catalog, and verification models |
| Hamilton dataflows | `src/feature_forge/dataflows/` | Optional inner case DAG; legacy outer control plane remains compatible |
| Durable storage | `src/feature_forge/storage/` | Atomic manifest packages, verified resume, and rebuildable DuckDB catalog |
| Verification CLI | `src/feature_forge/cli.py`, `verification/` | Read-only verification, explicit catalog rebuild, side-effect-free planning |
| Methods (plugin registry) | `src/feature_forge/methods/` | `BaseMethod` + `MethodRegistry` (entry-point discovery) — 5 methods |
| MALMAS agents | `src/feature_forge/methods/malmas/agents/` | 6 specialized agents + `RouterAgent` — entry-point pluggable |
| MALMAS pipeline | `src/feature_forge/methods/malmas/pipeline/` | `CorePipeline` (single-round) → `IterativePipeline` (multi-round with router/memory) |
| MALMAS memory | `src/feature_forge/methods/malmas/memory/` | 2-tier: procedural, feedback — with persistence |
| MALMAS prompts | `src/feature_forge/methods/malmas/prompts/` | YAML templates per agent type |
| LLM abstraction | `src/feature_forge/llm/` | `LLMClient` (ABC) → 4 providers (deepseek, openai, anthropic, litellm) + DiskCache |
| Evaluation | `src/feature_forge/evaluation/` | `CVEvaluator` (k-fold) + `SandboxedExecutor` (AST validation + subprocess worker) |
| Data loading | `src/feature_forge/data/` | `DatasetRegistry` (entry-point) + `ingestion` — titanic, house_prices built-in |
| Observability | `src/feature_forge/observability/` | structlog config + Langfuse tracer |
| Artifacts | `src/feature_forge/artifacts/` | `ArtifactExporter` base + `DataFrameStorage` (memory/disk/hybrid) |
| Configuration | `src/feature_forge/config.py` | `Settings(BaseSettings)` — YAML → env (`FF_*`) → .env (dotenvx) |
| Types | `src/feature_forge/types.py` | `FeatureSpec`, `DatasetName`, `MetricName`, newtypes |

**Entry points**:
- `from feature_forge import ExperimentalPlatform` → `src/feature_forge/__init__.py:10`
- `from feature_forge.api import FeatureForge` → `src/feature_forge/api.py:40`
- `uv run pytest` → `tests/` via `pyproject.toml:213`

## Key Data Flows

**Sklearn flow**: `FeatureForge.fit(X, y)` → `IterativePipeline.run()` → N rounds of [Router selects agents → agents generate FeatureSpecs → CorePipeline generates/executes code → CVEvaluator scores] → `FeatureForge.transform(X)` applies code via `SandboxedExecutor`

**Experiment flow**: `ExperimentalPlatform.run()` → cartesian `datasets × methods × models × seeds` → `ExperimentCaseExecutor.execute()` per case → `MethodRegistry` resolves method → `BaseMethod.fit_transform()` → results collected → `Reporter.to_markdown()`

**Medallion flow (opt-in)**: outer scheduler/context/resource plan → Hamilton inner case DAG → Bronze source evidence → Silver canonical rows/folds → Gold replayable features → Platinum fold evidence → verified manifests → derived catalog

**LLM call flow**: `LLMClient.chat()` → provider-specific SDK (openai/anthropic/deepseek/litellm) → response cached by SHA-256 key in `DiskCache` → optional Langfuse trace

**Agent generation flow**: Agent reads YAML prompt + memory context → `LLMClient.generate()` → structured JSON parsed to `FeatureSpec` list → `CodeGenerator` produces Python code → `SandboxedExecutor.execute()` runs in subprocess with AST validation

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Core runtime |
| Package manager | uv | Dependency management, virtual environments |
| Build system | hatchling | PEP 517 build backend |
| Config | pydantic-settings + PyYAML | Layered config: YAML → env → .env |
| Secrets | dotenvx | Encrypted `.env` file management |
| LLM clients | openai, anthropic | SDK for LLM API calls |
| LLM routing | litellm (optional) | Multi-provider LLM proxy |
| ML models | scikit-learn, xgboost | Baseline and evaluation models |
| ML models (opt) | lightgbm, catboost | Additional evaluation models |
| Data | pandas, numpy, pyarrow | DataFrame manipulation |
| AutoFE baselines | openfe, caafe | Comparison feature engineering methods |
| Execution sandbox | multiprocessing + ast | Sandboxed LLM code execution |
| Logging | structlog | Structured JSON/console logging |
| Observability | Langfuse, OpenTelemetry | LLM call tracing and monitoring |
| Experiment tracking | none (safe default), wandb/mlflow (opt) | Scalar metrics and durable artifact references |
| Caching | diskcache | LLM response cache (SHA-256 keys) |
| Testing | pytest, hypothesis | Unit/integration/property testing |
| Linting | ruff | Linting + formatting |
| Type checking | mypy (strict) | Static type analysis |
| CI | GitHub Actions + pre-commit | Automated quality gates |
| Docs | MkDocs Material + mkdocstrings | Documentation site |
| Notebooks | marimo, jupyter | Interactive experimentation |
| Viz | matplotlib, seaborn | Feature and result visualization |

## Infrastructure

```
uv sync --group dev          # install core + development dependencies
uv sync --extra pipeline --group dev  # add Hamilton and DuckDB
uv run pytest                # test suite
uv run ruff check src        # lint gate
uv run mypy src              # type gate
make docs-check              # generated freshness + strict MkDocs
```

No Docker. No external services required at runtime (LLM APIs called via HTTP).

## Integrations

| Service | SDK | Purpose | Status |
|---------|-----|---------|--------|
| DeepSeek API | `openai` (compatible) | Default LLM provider | Active |
| OpenAI API | `openai` | LLM provider | Active |
| Anthropic API | `anthropic` | LLM provider | Active |
| LiteLLM | `litellm` (optional) | Multi-provider LLM proxy | Optional |
| Weights & Biases | `wandb` (optional) | Experiment tracking (default) | Active |
| MLflow | `mlflow` (optional) | Experiment tracking | Optional |
| Langfuse | `langfuse` | LLM observability/tracing | Active |
| OpenTelemetry | `opentelemetry-api` | Distributed tracing | Active |
| Kaggle | `kaggle` (optional) | Dataset download | Optional |

### Auth Flow

No user authentication. All auth is API-key-based for external services: LLM provider keys (`FF_LLM__API_KEY` or provider-specific env vars), WandB key (`WANDB_API_KEY`), and Langfuse keys (`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`). Secrets are managed via dotenvx-encrypted `.env` file; non-sensitive defaults live in `config/settings.yaml`.

## Environment Variables

| Variable | Context | Purpose |
|----------|---------|---------|
| `FF_TASK` | core | `classification` or `regression` |
| `FF_METRIC` | core | Evaluation metric: auc, acc, f1, rmse, mae, r2, nrmse |
| `FF_N_ROUNDS` | core | Number of pipeline iterations |
| `FF_RANDOM_STATE` | core | Global random seed |
| `FF_LLM__MODEL` | llm | Model identifier (e.g. `deepseek-chat`) |
| `FF_LLM__API_KEY` | llm | Single API key for LLM provider |
| `FF_LLM__BASE_URL` | llm | Override API endpoint URL |
| `FF_LLM__PROVIDER` | llm | `auto`, `deepseek`, `openai`, `anthropic`, `litellm` |
| `FF_LLM__TEMPERATURE` | llm | Sampling temperature |
| `FF_LLM__MAX_TOKENS` | llm | Max response tokens |
| `FF_TRACKER__BACKEND` | tracker | `wandb`, `mlflow`, or `none` (default: `none`) |
| `FF_TRACKER__PROJECT` | tracker | Tracker project name |
| `FF_ROUTER__STRATEGY` | router | `data_driven`, `performance_driven`, `hybrid`, `llm` |
| `FF_MEMORY__MAX_SIZE` | memory | Max entries per memory type |
| `FF_EVALUATION__CV_FOLDS` | evaluation | Cross-validation folds |
| `FF_EVALUATION__EVALUATION_HOLDOUT_FRACTION` | evaluation | Discovery/evaluation split fraction (`0.0` disables; default `0.25`) |
| `FF_EVALUATION__SANDBOX_TIMEOUT_SECONDS` | evaluation | Sandbox worker timeout |
| `WANDB_API_KEY` | wandb | WandB authentication |
| `LANGFUSE_PUBLIC_KEY` | langfuse | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | langfuse | Langfuse secret key |
| `LANGFUSE_HOST` | langfuse | Langfuse endpoint URL |
