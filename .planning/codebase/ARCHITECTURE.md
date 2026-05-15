# Architecture

## Overall Pattern

**Layered multi-agent pipeline** with a sklearn-compatible facade. The system follows a **strategy + template method** pattern where the pipeline orchestrates specialized agents, each responsible for a different feature engineering approach. Configuration flows top-down through Pydantic settings; data flows bottom-up through DataFrames and feature specs.

Single Python package (`feature_forge`) with a `src/` layout.

## Layers (top to bottom)

### 1. API / Facade Layer

- **`src/feature_forge/api.py`** — `FeatureForge` (extends `BaseEstimator`, `TransformerMixin`, `ArtifactExporter`)
  - sklearn-compatible `fit()` / `transform()` / `fit_transform()` interface
  - Delegates to pipeline variants based on `mode` parameter (`"full"`, `"no_memory"`, `"no_router"`, or a specific agent name)
  - Handles sync-to-async bridge via `utils.run_coro_sync()`
  - Collects artifacts and provenance from pipeline results

### 2. Pipeline Orchestration Layer

- **`src/feature_forge/methods/malmas/pipeline/iterative.py`** — Multi-round iterative pipeline
  - `BaseIterativePipeline` — Template method: `_select_agents()` → `_build_agent_context()` → `CorePipeline.run()` → `_post_round()`
  - `IterativePipeline` — Full pipeline with router + memory
  - Each round: router picks agents → agents generate specs → code generated → sandboxed execution → CV evaluation → memory update → router performance update
- **`src/feature_forge/methods/malmas/pipeline/core.py`** — Single-round core pipeline
  - `CorePipeline.run()`: runs agents in parallel (semaphore-bounded) → generates code via LLM → executes in sandbox → evaluates via CV → selects top-k effective features
- **`src/feature_forge/methods/malmas/pipeline/ablations.py`** — Ablation variants
  - `NoMemoryPipeline`, `NoRouterPipeline`, `SingleAgentPipeline`

### 3. Agent Layer

- **`src/feature_forge/methods/malmas/agents/base.py`** — `Agent` ABC and `BaseFeatureAgent` base class
  - Each agent has a system prompt (loaded from `src/feature_forge/prompts/`) and implements `generate()` → returns `list[FeatureSpec]`
  - `AgentRegistry` — Discovers agents via Python entry points or built-in module map
- **Built-in agents**: `unary.py`, `cross_compositional.py`, `aggregation.py`, `temporal.py`, `local_transform.py`, `local_pattern.py`
- **`src/feature_forge/methods/malmas/agents/router.py`** — `RouterAgent` with strategies: `data_driven`, `performance_driven`, `hybrid`, `llm`

### 4. Memory Layer

- **`src/feature_forge/methods/malmas/memory/base.py`** — `AgentMemory` with 3-tier MALMAS-style memory: procedural, feedback, conceptual
- Per-agent persistence to JSON files in `memory_files/agent_memories/`

### 5. LLM Abstraction Layer

- **`src/feature_forge/llm/base.py`** — `LLMClient` ABC with template method pattern: providers implement `_call_api()`, `_extract_content()`, `_extract_usage()` hooks; base class handles logging, timing, retry, JSON schema injection, and error wrapping
- **`src/feature_forge/llm/factory.py`** — `create_llm_client()` auto-detects provider from model name
- **`src/feature_forge/llm/cache.py`** — Disk-backed response cache
- **Providers**: `openai.py`, `deepseek.py` (extends OpenAIProvider), `anthropic.py`, `litellm_provider.py`

### 6. Evaluation Layer

- **`src/feature_forge/evaluation/sandbox.py`** — `SandboxedExecutor` with AST validation + process isolation + timeout/memory limits
- **`src/feature_forge/evaluation/cv.py`** — `CVEvaluator` with K-fold CV + baseline comparison + per-feature gain
- **`src/feature_forge/evaluation/metrics.py`** — Metric functions (auc, acc, f1, rmse, mae, r2)
- **`src/feature_forge/evaluation/model_factory.py`** — Model factory (XGBoost by default)

### 7. Supporting Layers

- **Configuration** (`config.py`) — Pydantic Settings with YAML + env var + constructor priority
- **Observability** (`observability/`) — structlog logging + Langfuse tracing
- **Experiment** (`experiment/`) — Matrix generation, runner, tracker (W&B / MLflow), reporter
- **Baselines** (`baselines/`) — OpenFE, CAAFE, LLMFE, Malmus baselines
- **Artifacts** (`artifacts/`) — ArtifactExporter ABC, storage, schema, diff, comparison, dashboard
- **Data** (`data/`) — Kaggle fetcher, dataset registry

## Data Flow

```
User code
  │
  ▼
FeatureForge.fit(X, y)                  ← api.py
  │
  ▼
IterativePipeline.run(X, y)            ← pipeline/iterative.py
  │
  ├─▶ RouterAgent.select_agents()      ← agents/router.py
  │     (picks agent subset per round)
  │
  ├─▶ AgentMemory.generate_prompt_section()  ← memory/base.py
  │     (injects memory context per agent)
  │
  ├─▶ CorePipeline.run(agents, X, y)   ← pipeline/core.py
  │     │
  │     ├─▶ Agent.generate() → FeatureSpec[]  (parallel, semaphore-bounded)
  │     │     └─▶ LLMClient.complete()        ← llm/
  │     │
  │     ├─▶ CodeGenerator.generate_code(specs) → Python code string
  │     │
  │     ├─▶ SandboxedExecutor.execute(code, X) → DataFrame  (subprocess)
  │     │
  │     └─▶ CVEvaluator.evaluate_feature() → gain per feature
  │           └─▶ ModelFactory + sklearn CV
  │
  ├─▶ AgentMemory.record_procedure/feedback()  ← memory update
  │
  └─▶ RouterAgent.update_performance()         ← router update
  │
  ▼
Result: {X_train_enhanced, selected_features, feature_codes, round_artifacts, ...}
```

## Configuration Flow

```
config/settings.yaml          ← YAML defaults
       │
       ▼
.env (dotenvx encrypted)      ← Secrets (API keys)
       │
       ▼
FF_* environment variables    ← Runtime overrides
       │
       ▼
Settings (Pydantic)           ← config.py
  ├─ LLMConfig
  ├─ RouterConfig
  ├─ MemoryConfig
  ├─ EvaluationConfig
  ├─ RetryConfig
  └─ TrackerConfig
```

## Key Abstractions

1. **`Agent`** (ABC) → Specialized feature generators with unified interface
2. **`LLMClient`** (ABC) → Provider-agnostic LLM calls via template method pattern
3. **`ArtifactExporter`** (ABC/mixin) → Consistent artifact access across all methods
4. **`Settings`** (Pydantic) → Immutable, validated configuration
5. **`SandboxedExecutor`** → Safe code execution with AST validation + process isolation
6. **`AgentMemory`** → Per-agent 3-tier memory system

## Component Wiring

- **API → Pipeline**: `FeatureForge` instantiates pipeline variants via `_get_pipeline()`, delegates `fit()` to `pipeline.run()`
- **Pipeline → Agents**: `IterativePipeline` uses `AgentRegistry` to instantiate agents by name
- **Pipeline → Router**: `IterativePipeline` delegates agent selection to `RouterAgent.select_agents()` each round
- **Pipeline → Memory**: `IterativePipeline` manages per-agent `AgentMemory` instances
- **Agents → LLM**: Each `Agent` holds an `LLMClient` reference
- **CorePipeline → Sandbox**: Generated code executed via `SandboxedExecutor.execute()` in subprocess
- **CorePipeline → Evaluator**: `CVEvaluator` measures feature gains against baseline
- **All → Config**: `Settings` instance flows through constructors to all components
- **All → Observability**: `structlog` logger obtained via `get_logger(__name__)` everywhere

## Directory Tree

```
feature_forge/
├── config/                    # Non-secret configuration
│   ├── logging.yaml           # Structlog config
│   └── settings.yaml          # Default settings
├── data/                      # Dataset storage
│   ├── kaggle/                # Cached Kaggle downloads
│   ├── raw/                   # Raw data
│   └── samples/               # Sample datasets
├── docs/                      # Documentation (MkDocs)
├── experiments/               # Experiment results and configs
├── memory_files/              # Agent memory persistence
│   ├── agent_memories/        # Per-agent JSON memory files
│   └── llm_cache/             # LLM response cache
├── notebooks/                 # Quarto notebooks (source of truth)
├── scripts/                   # Utility scripts
├── src/feature_forge/         # Main package
│   ├── api.py                 # FeatureForge (sklearn API)
│   ├── config.py              # Pydantic settings
│   ├── exceptions.py          # Exception hierarchy
│   ├── types.py               # NewType aliases
│   ├── utils.py               # Async bridge, markdown utils
│   ├── agents/                # Feature generation agents + router
│   ├── artifacts/             # Artifact storage and export
│   ├── baselines/             # Baseline methods (OpenFE, CAAFE, etc.)
│   ├── data/                  # Data loading and registry
│   ├── evaluation/            # Sandboxed execution + CV evaluation
│   ├── experiment/            # Experiment matrix, runner, tracker
│   ├── llm/                   # LLM abstraction + providers
│   ├── memory/                # Agent memory system
│   ├── observability/         # Logging and tracing
│   ├── pipeline/              # Core + iterative + ablation pipelines
│   └── prompts/               # LLM prompt templates (.txt)
├── tests/
│   ├── conftest.py            # Shared fixtures (FakeLLM, etc.)
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── benchmarks/            # Performance benchmarks
└── pyproject.toml             # Project metadata, deps, tool config
```

## Key Locations

| Concern | Path |
|---|---|
| User-facing API | `src/feature_forge/api.py` |
| Pipeline orchestration | `src/feature_forge/methods/malmas/pipeline/iterative.py` |
| Core single-round logic | `src/feature_forge/methods/malmas/pipeline/core.py` |
| Agent definitions | `src/feature_forge/methods/malmas/agents/*.py` |
| Agent selection (router) | `src/feature_forge/methods/malmas/agents/router.py` |
| LLM provider factory | `src/feature_forge/llm/factory.py` |
| Configuration | `src/feature_forge/config.py` + `config/settings.yaml` |
| Secrets | `.env` (dotenvx) + `.env.example` |
| Prompt templates | `src/feature_forge/prompts/*.txt` |
| Memory system | `src/feature_forge/methods/malmas/memory/base.py` |
| Sandboxed execution | `src/feature_forge/evaluation/sandbox.py` |
| Feature evaluation | `src/feature_forge/evaluation/cv.py` |
| Baselines | `src/feature_forge/methods/*.py` |
| Experiment runner | `src/feature_forge/experiment/runner.py` |
| Observability | `src/feature_forge/observability/structlog_config.py` |

## Entry Points

| Entry Point | Location | Purpose |
|---|---|---|
| Package init | `src/feature_forge/__init__.py` | Configures logging, exports `__version__` |
| sklearn API | `src/feature_forge/api.py::FeatureForge` | Primary user-facing class |
| Pipeline | `src/feature_forge/methods/malmas/pipeline/iterative.py::IterativePipeline` | Full pipeline orchestration |
| Core pipeline | `src/feature_forge/methods/malmas/pipeline/core.py::CorePipeline` | Single-round execution |
| Agent entry points | `pyproject.toml [project.entry-points."feature_forge.agents"]` | Plugin discovery |
| Baseline entry points | `pyproject.toml [project.entry-points."feature_forge.baselines"]` | Baseline discovery |

## Naming Conventions

- **Source package**: `src/feature_forge/` — single package, no namespace
- **Module naming**: `snake_case.py` (e.g., `cross_compositional.py`, `model_factory.py`)
- **Classes**: `PascalCase` (e.g., `FeatureForge`, `CVEvaluator`)
- **Abstract bases**: Prefixed with `Base` or suffixed `ABC` (e.g., `BaseFeatureAgent`, `LLMClient(ABC)`)
- **Config models**: Suffix `Config` (e.g., `LLMConfig`, `RouterConfig`)
- **Private methods**: Leading underscore (`_select_agents()`, `_build_user_prompt()`)
- **Exception hierarchy**: Suffix `Error`, rooted at `FeatureForgeError`
- **Environment variables**: `FF_` prefix with `__` nested delimiter (e.g., `FF_LLM__API_KEY`)
- **Type aliases**: `NewType` for domain strings (e.g., `AgentName`, `DatasetName`)

## Architectural Conventions

- **Dependency injection**: Settings and LLMClient flow through constructors
- **Lazy imports**: Heavy modules imported only when needed
- **Plugin discovery**: Agents and baselines registered via `pyproject.toml` entry points
- **Async-first**: Core logic is async; sync bridge in `utils.run_coro_sync()`
- **Structured logging**: `structlog` with `get_logger(__name__)` pattern everywhere
- **Pydantic validation**: All configuration uses validated Pydantic models with field validators
