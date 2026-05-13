# Feature Forge — Overview

> Modular experimentation platform for LLM-based multi-agent automated feature engineering on tabular data. Refactoring of the MALMAS research codebase into a composable, sklearn-compatible Python package with 6 specialized agents, 3-tier memory, and a dynamic router.

## Architecture

**Pattern:** Layered library with async internals and sklearn-compatible sync API. Single Python package (`feature_forge`) with no server component.

```
User Code (sklearn Pipeline / Experiment Matrix)
        │
        ▼
   API Layer          FeatureForge (BaseEstimator + TransformerMixin)
        │
        ▼
   Pipeline Layer     IterativePipeline → CorePipeline (per-round)
        │
   ┌────┼────────────────────────┐
   ▼    ▼            ▼           ▼
 Agent  Router    Memory     CodeGenerator
 Layer   Layer    Layer          │
   │                              ▼
   └── LLM Layer ──────►  Sandbox (AST + process isolation)
        │
   Evaluation Layer   CVEvaluator → ModelFactory → Metrics
        │
   Experiment Layer   Matrix → Runner → Tracker (WandB/MLflow)
```

### Entry Points

| Entry Point | Location | Description |
|-------------|----------|-------------|
| Sklearn API | `src/feature_forge/api.py:40` (`FeatureForge`) | `fit(X, y)` / `transform(X)` / `fit_transform(X, y)` |
| Experiment Matrix | `src/feature_forge/experiment/matrix.py:9` (`ExperimentMatrix`) | Cartesian product of datasets × methods × seeds × models × rounds |
| Experiment Runner | `src/feature_forge/experiment/runner.py` | Orchestrates matrix runs with tracking |
| CLI / scripts | `scripts/` | Notebook rendering, hygiene checks, doc generation |

### Pipeline Modes

Defined in `src/feature_forge/api.py:89-93`:

| Mode | Pipeline Class | Description |
|------|---------------|-------------|
| `full` | `IterativePipeline` | Router + memory + all agents |
| `no_memory` | `NoMemoryPipeline` | Router but no memory |
| `no_router` | `NoRouterPipeline` | Memory but all agents every round |
| Agent name (e.g. `unary`) | `SingleAgentPipeline` | Single agent, no router/memory |

## Key Data Flows

**Feature generation (per round):**
1. Router selects agents → agents call LLM with prompts + memory context → JSON feature specs
2. CodeGenerator calls LLM to produce pandas code from specs → AST-validated sandbox execution
3. CVEvaluator scores each feature against baseline → top-k effective features selected
4. Memory updated (procedural, feedback, conceptual) → router performance updated

**Experiment flow:**
`ExperimentMatrix.generate()` → list of config dicts → `ExperimentRunner.run()` → for each config: load dataset → create FeatureForge → fit/transform → evaluate → track results → `Reporter` summarizes

**Sync/async bridge:**
`FeatureForge.fit()` (sync) → `utils.py:78` `run_coro_sync()` → `IterativePipeline.run()` (async) → `CorePipeline.run()` (async) → agent `.generate()` (async)

## Tech Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | >=3.11 | Runtime |
| Package manager | uv | (latest) | Dependency management, virtualenvs |
| Build system | hatchling | (latest) | PEP 517 build backend |
| Config | pydantic + pydantic-settings | >=2.0 | Immutable validated settings, YAML + env loading |
| LLM (core) | openai | >=1.0 | OpenAI-compatible API client |
| LLM (alt) | anthropic | >=0.40 | Anthropic Claude client |
| Data | pandas + numpy | >=2.0 / >=1.24 | Tabular data manipulation |
| ML | scikit-learn | >=1.3 | CV, metrics, Pipeline compatibility |
| ML | xgboost | >=2.0 | Default evaluation model |
| Viz | matplotlib | >=3.7 | Plotting |
| Logging | structlog | >=24.0 | Structured logging |
| Retry | tenacity | >=8.0 | LLM API retry logic |
| CLI output | rich + tqdm | >=13.0 / >=4.65 | Terminal formatting, progress bars |
| Config format | PyYAML | >=6.0 | YAML settings files |
| **Optional: Baselines** | | | |
| Baseline | openfe | (any) | OpenFE baseline method |
| Baseline | caafe | (any) | CAAFE baseline method |
| Baseline | mlflow | >=2.10 | MLflow experiment tracking |
| Baseline | litellm | >=1.40 | Multi-provider LLM routing |
| **Optional: Observability** | | | |
| Tracking | wandb | >=0.16 | Weight & Biases experiment tracking |
| Observability | langfuse | >=2.0 | LLM observability / tracing |
| Observability | opentelemetry-api | >=1.20 | Distributed tracing |
| Cache | diskcache | >=5.6 | SQLite-backed LLM response cache |
| Serialization | pyarrow | >=23.0.1 | Parquet I/O for sandbox IPC |
| **Dev** | | | |
| Test | pytest | >=8.0 | Testing framework |
| Lint | ruff | >=0.3 | Linter + formatter |
| Type check | mypy | >=1.0 | Static type checking (strict mode) |
| Docs | mkdocs-material + mkdocstrings + mike | >=9.5 / >=0.25 / >=2.0 | Documentation site |
| Notebooks | marimo + jupyter | >=0.5 / >=7.0 | Interactive notebooks |
| Pre-commit | pre-commit | >=3.6 | Git hooks |
| Security | pip-audit | >=0.8 | Dependency vulnerability scanning |

## Infrastructure

No Docker or server deployment. Pure Python library.

```
uv sync                    # Install all dependencies
uv run pytest              # Run tests
uv run ruff check src tests # Lint
uv run mypy src            # Type check
make docs                  # Build MkDocs site
make docs-serve            # Live docs preview
```

### CI (`.github/workflows/`)

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yml` | push/PR to main/develop | lint, typecheck, test (3.11/3.12/3.13), security (pip-audit), hygiene |
| `docs.yml` | push to main | Build + deploy MkDocs to GitHub Pages |
| `benchmark.yml` | (manual/schedule) | Run experiment benchmarks |

## Integrations

| Service | SDK / Package | Purpose | Status |
|---------|--------------|---------|--------|
| OpenAI API | `openai` | LLM completions (GPT models) | Active |
| DeepSeek API | `openai` (compatible) | LLM completions (default provider) | Active |
| Anthropic API | `anthropic` | LLM completions (Claude models) | Active |
| LiteLLM | `litellm` | Multi-provider LLM routing fallback | Active (optional) |
| WandB | `wandb` | Experiment tracking (default backend) | Active (optional) |
| MLflow | `mlflow` | Experiment tracking (alternative backend) | Active (optional) |
| Langfuse | `langfuse` | LLM observability / tracing | Active (optional) |
| OpenTelemetry | `opentelemetry-api` | Distributed tracing | Stub (optional) |
| Kaggle | `kaggle` (via `data/ingestion.py`) | Dataset downloading | Active (optional) |
| DiskCache | `diskcache` | LLM response caching (SHA-256 keys, SQLite-backed) | Active (optional) |
| OpenFE | `openfe` | Baseline feature engineering method | Active (optional) |
| CAAFE | `caafe` | Baseline feature engineering method | Active (optional) |

### Agent Registry (Entry Points)

Agents are discovered via `importlib.metadata` entry points (`pyproject.toml:84-90`):

| Agent Name | Class | Module |
|-----------|-------|--------|
| `unary` | `UnaryFeatureAgent` | `agents/unary.py` |
| `cross_compositional` | `CrossCompositionalAgent` | `agents/cross_compositional.py` |
| `aggregation` | `AggregationConstructAgent` | `agents/aggregation.py` |
| `temporal` | `TemporalFeatureAgent` | `agents/temporal.py` |
| `local_transform` | `LocalTransformAgent` | `agents/local_transform.py` |
| `local_pattern` | `LocalPatternAgent` | `agents/local_pattern.py` |

### Baselines Registry (Entry Points)

Baselines via `pyproject.toml:93-96`: `openfe`, `caafe`, `llmfe`, `malmus`

## Environment Variables

Config priority: constructor args > env vars > `.env` (dotenvx) > `config/settings.yaml`. All env vars use `FF_` prefix with `__` nested delimiter (`src/feature_forge/config.py:233-236`).

| Variable | Purpose |
|----------|---------|
| `FF_TASK` | Task type: `classification` or `regression` |
| `FF_METRIC` | Evaluation metric: `auc`, `acc`, `f1`, `rmse`, `mae`, `r2` |
| `FF_N_ROUNDS` | Number of pipeline iterations (default: 4) |
| `FF_RANDOM_STATE` | Global random seed (default: 42) |
| `FF_VERBOSE` | Verbosity level 0-2 (default: 1) |
| `FF_LLM__MODEL` | LLM model identifier (default: `deepseek-chat`) |
| `FF_LLM__PROVIDER` | Provider: `auto`, `deepseek`, `openai`, `anthropic`, `litellm` |
| `FF_LLM__API_KEY` | LLM API key (secret — use `.env`) |
| `FF_LLM__BASE_URL` | Override API endpoint URL |
| `FF_LLM__TEMPERATURE` | Sampling temperature 0.0–2.0 (default: 0.2) |
| `FF_LLM__MAX_TOKENS` | Max tokens per response (default: 4096) |
| `FF_LLM__CACHE_RESPONSES` | Enable DiskCache for LLM responses (default: true) |
| `FF_LLM__MAX_CONCURRENT_CALLS` | Semaphore limit for parallel LLM calls (default: 3) |
| `FF_TRACKER__BACKEND` | Tracker: `wandb`, `mlflow`, or `none` |
| `FF_TRACKER__PROJECT` | Tracker project name |
| `FF_TRACKER__ENTITY` | WandB entity/team name |
| `FF_ROUTER__STRATEGY` | Agent selection: `data_driven`, `performance_driven`, `hybrid`, `llm` |
| `FF_ROUTER__MIN_AGENTS` | Minimum agents per round (default: 1) |
| `FF_ROUTER__MAX_AGENTS` | Maximum agents per round (default: all) |
| `FF_MEMORY__MAX_SIZE` | Max entries per memory type (default: 100) |
| `FF_MEMORY__PERSISTENCE_DIR` | Memory JSON storage directory |
| `FF_EVALUATION__CV_FOLDS` | Cross-validation folds (default: 5) |
| `FF_EVALUATION__TEST_SIZE` | Test split fraction (default: 0.4) |
| `FF_EVALUATION__SANDBOX_TIMEOUT_SECONDS` | Sandbox worker timeout (default: 5.0) |
| `FF_EVALUATION__SANDBOX_MAX_MEMORY_MB` | Sandbox memory limit (default: 512) |
| `FF_LOG_LEVEL` | structlog level: `debug`, `info`, `warning`, `error` |
| `FF_RETRY__MAX_RETRIES` | LLM call retry count (default: 3) |
| `FF_RETRY__BACKOFF_BASE` | Exponential backoff base seconds (default: 1.0) |
| `OPENAI_API_KEY` | Provider-specific override for OpenAI |
| `ANTHROPIC_API_KEY` | Provider-specific override for Anthropic |
| `DEEPSEEK_API_KEY` | Provider-specific override for DeepSeek |
| `WANDB_API_KEY` | WandB authentication key (secret) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse observability key (secret) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key (secret) |
| `LANGFUSE_HOST` | Langfuse server URL (default: `https://cloud.langfuse.com`) |

### Secrets Management

Secrets are encrypted via **dotenvx**: `.env` is committed encrypted, `.env.keys` is gitignored. Decryption happens at runtime via `dotenvx run --` or the pydantic-settings dotenv source.
