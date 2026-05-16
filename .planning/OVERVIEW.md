# Feature Forge — Overview

> Modular experimentation platform for LLM-based multi-agent automated feature engineering on tabular data.

## Architecture

**Pattern:** Single Python package, layered architecture with plugin-style entry-point registries. No server component — CLI/library only.

```
                    ┌──────────────────────────────────┐
                    │     ExperimentalPlatform          │
                    │  (facade: datasets×methods×models) │
                    └──────────┬───────────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              ▼                ▼                    ▼
     ┌────────────┐  ┌────────────────┐   ┌──────────────┐
     │  FeatureForge│  │ ExperimentRunner│   │   Reporter   │
     │  (sklearn)  │  │   + Tracker     │   │  (markdown)  │
     └──────┬─────┘  └────────────────┘   └──────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌──────────┐ ┌──────────────────────────┐
│ Methods  │ │   MALMAS IterativePipeline│
│ Registry │ │  Router → Agents × Rounds │
└──┬───┬───┘ │  Memory → CodeGen → Eval  │
   │   │     └──────────────────────────┘
   │   │
   ▼   ▼          ┌────────────────────────────┐
┌──────┐┌───────┐ │    CorePipeline             │
│malmas││ caafe │ │  Agent → Spec → CodeGen     │
│malmus││ llmfe │ │  → Sandbox → CV Evaluate    │
│openfe│└───────┘ │  → Select (gain filter)     │
└──────┘          └────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  LLM Layer                           │
│  Factory → DeepSeek/OpenAI/Anthropic/│
│  LiteLLM + DiskCache + Langfuse      │
└──────────────────────────────────────┘
```

### feature_forge — Core Package (Python)

Layered: **Platform → API/Experiment → Methods → Agents/Pipeline → LLM/Evaluation → Data**

| Layer | Location | Pattern |
|-------|----------|---------|
| Platform | `src/feature_forge/platform.py` | Unified facade: `ExperimentalPlatform` |
| Sklearn API | `src/feature_forge/api.py` | `FeatureForge(BaseEstimator, TransformerMixin)` |
| Methods | `src/feature_forge/methods/` | Plugin registry via entry points (`pyproject.toml:111-124`) |
| Agents | `src/feature_forge/methods/malmas/agents/` | 6 specialized agents + router |
| Pipeline | `src/feature_forge/methods/malmas/pipeline/` | Core (single-round) + Iterative (multi-round) + ablations |
| Memory | `src/feature_forge/methods/malmas/memory/` | 3-tier: Procedural, Feedback, Conceptual |
| LLM | `src/feature_forge/llm/` | Factory pattern, provider-specific clients |
| Evaluation | `src/feature_forge/evaluation/` | CV evaluator, sandboxed execution, metrics, model factory |
| Data | `src/feature_forge/data/` | Dataset registry (Kaggle + local + entry points) |
| Experiment | `src/feature_forge/experiment/` | Matrix builder, runner (seq/parallel), trackers, reporter |
| Observability | `src/feature_forge/observability/` | structlog + Langfuse + OpenTelemetry |
| Config | `src/feature_forge/config.py` | Pydantic-settings: YAML → env vars → constructor args |
| Artifacts | `src/feature_forge/artifacts/` | Schema, storage, diff, comparison dashboard |

**Entry point**: `from feature_forge import ExperimentalPlatform` → `src/feature_forge/__init__.py:10`

## Key Data Flows

**Iterative feature generation**: Router selects agents → agents generate FeatureSpecs → CodeGen produces pandas code → Sandbox executes → CV evaluates gain → features above threshold are kept → memory updated.

**Experiment comparison**: `ExperimentalPlatform.run()` builds config matrix (dataset×method×model×seed) → loads data from registry → resolves method from entry points → runs fit/transform → CV evaluates baseline vs enhanced → Reporter outputs markdown table.

**Sklearn pipeline**: `FeatureForge.fit(X, y)` → creates IterativePipeline → runs N rounds → stores selected features + code → `transform(X)` replays code in sandbox on new data.

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Runtime |
| Package manager | uv | Dependency management |
| Build | hatchling | Build backend |
| Config | pydantic-settings 2.x + PyYAML | Validated settings from YAML/env/args |
| LLM clients | openai 1.x, anthropic 0.40+, litellm 1.40+ | Multi-provider LLM access |
| ML | scikit-learn 1.3+, xgboost 2.0+, pandas 2.0+, numpy 1.24+ | Models, data, evaluation |
| Logging | structlog 24.x | Structured logging |
| Observability | langfuse 2.x, opentelemetry-api 1.x | LLM tracing |
| Tracking | wandb 0.16+, mlflow 2.10+ | Experiment tracking |
| Caching | diskcache 5.6+ | LLM response cache (SHA-256 keyed) |
| Retry | tenacity 8.x | LLM API retry with backoff |
| CLI output | rich 13.x, tqdm 4.65+ | Pretty terminal output |
| Docs | mkdocs-material, mkdocstrings, mkdocs-jupyter | Documentation site |
| Testing | pytest 8.x, hypothesis, pytest-asyncio | Unit/property/metamorphic tests |
| Linting | ruff 0.3+ | Lint + format |
| Typing | mypy (strict) | Static type checking |
| Notebooks | marimo 0.5+, jupyter | Interactive exploration |

## Infrastructure

```
uv sync                              # install all deps
uv run pytest                        # run tests
uv run ruff check src                # lint
uv run mypy src                      # type check
uv run mkdocs serve                  # docs dev server :8000
make docs-serve                      # docs + notebooks
```

No Docker, no server processes. Library-only — runs as a Python package.

## Integrations

| Service | SDK | Purpose | Status |
|---------|-----|---------|--------|
| DeepSeek | `openai` (compatible) | Primary LLM provider | Active |
| OpenAI | `openai` | LLM provider | Active |
| Anthropic | `anthropic` | LLM provider | Active |
| LiteLLM | `litellm` | Catch-all LLM proxy | Active |
| Langfuse | `langfuse` | LLM observability/tracing | Active |
| WandB | `wandb` | Experiment tracking (default) | Active |
| MLflow | `mlflow` | Experiment tracking (optional) | Active |
| Kaggle | `kaggle` | Dataset fetching | Active |
| OpenFE | `openfe` | Baseline feature engineering | Active |
| CAAFE | `caafe` | Baseline feature engineering | Active |

### Auth Flow

No user auth. API keys are managed via dotenvx-encrypted `.env` file (never committed). The `FF_LLM__API_KEY` env var or provider-specific keys (`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are read by `pydantic-settings` into `LLMConfig.api_key` (`SecretStr`). The LLM factory (`src/feature_forge/llm/factory.py:41`) extracts the secret and passes it to the provider client. Tracking backends use their own keys (`WANDB_API_KEY`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`).

## Environment Variables

| Variable | Context | Purpose |
|----------|---------|---------|
| `FF_LLM__API_KEY` | LLM | Unified API key for all providers |
| `DEEPSEEK_API_KEY` | LLM | DeepSeek-specific key (overrides unified) |
| `OPENAI_API_KEY` | LLM | OpenAI-specific key |
| `ANTHROPIC_API_KEY` | LLM | Anthropic-specific key |
| `GEMINI_API_KEY` | LLM | Gemini-specific key |
| `FF_LLM__MODEL` | Config | Model name (e.g. `deepseek-chat`) |
| `FF_LLM__BASE_URL` | Config | Override API base URL |
| `FF_LLM__PROVIDER` | Config | Force provider (`auto`/`deepseek`/`openai`/`anthropic`/`litellm`) |
| `FF_TASK` | Config | Task type (`classification`/`regression`) |
| `FF_METRIC` | Config | Evaluation metric (`auc`/`acc`/`f1`/`rmse`/`mae`/`r2`/`nrmse`) |
| `FF_TRACKER__BACKEND` | Config | Tracker (`wandb`/`mlflow`/`none`) |
| `FF_TRACKER__PROJECT` | Config | Tracker project name |
| `WANDB_API_KEY` | Tracking | WandB authentication |
| `LANGFUSE_PUBLIC_KEY` | Observability | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Observability | Langfuse secret key |
| `LANGFUSE_HOST` | Observability | Langfuse host URL |
| `FF_LOG_LEVEL` | Observability | Log level (default: `warning`) |
