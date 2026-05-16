# Feature Forge — Code Style & Conventions

## File Organization

### Where Things Go

```
feature_forge/
  src/feature_forge/              # Main package
    api.py                        # Public sklearn-compatible API (FeatureForge class)
    platform.py                   # ExperimentalPlatform facade
    config.py                     # Pydantic-settings (Settings, LLMConfig, etc.)
    types.py                      # Shared NewTypes, TypeVars, FeatureSpec model
    exceptions.py                 # Exception hierarchy (FeatureForgeError base)
    utils.py                      # Cross-cutting helpers (run_coro_sync, etc.)
    __init__.py                   # Package init with lazy __getattr__ exports
    py.typed                      # PEP 561 marker
    methods/                      # Feature engineering methods
      base.py                     # BaseMethod ABC, MethodProtocol, MethodRegistry
      _prompting.py               # PromptRegistry, Prompt model (YAML-backed)
      malmas/                     # MALMAS multi-agent method
        method.py                 # MALMASMethod adapter (wraps FeatureForge)
        types.py                  # AgentName NewType
        agents/                   # Agent classes
          base.py                 # Agent ABC, BaseFeatureAgent, AgentRegistry
          unary.py                # UnaryFeatureAgent
          cross_compositional.py  # CrossCompositionalAgent
          aggregation.py          # AggregationConstructAgent
          temporal.py             # TemporalFeatureAgent
          local_transform.py      # LocalTransformAgent
          local_pattern.py        # LocalPatternAgent
          router.py               # RouterAgent
        pipeline/                 # Pipeline orchestration
          core.py                 # CorePipeline, CodeGenerator
          iterative.py            # IterativePipeline (multi-round)
          ablations.py            # Ablation variants (NoMemory, SingleAgent, etc.)
        memory/                   # Agent memory subsystem
          base.py                 # Memory ABC
          conceptual.py           # ConceptualMemory
          persistence.py          # Disk persistence
          prompts.py              # Memory-related prompts
        prompts/                  # YAML prompt templates per agent
          <agent_name>.yaml
      caafe/                      # CAAFE method
      llmfe/                      # LLMFE method
      openfe/                     # OpenFE wrapper method
      malmus/                     # Malmus method
    llm/                          # LLM abstraction layer
      base.py                     # LLMClient ABC, LLMResponse
      factory.py                  # create_llm_client() factory
      cache.py                    # Response caching
      retry.py                    # Tenacity-based retry
      langfuse_wrapper.py         # Langfuse tracing wrapper
      providers/                  # Provider implementations
        openai.py                 # OpenAIProvider
        deepseek.py               # DeepSeekProvider
        anthropic.py              # AnthropicProvider
        litellm_provider.py       # LiteLLMProvider
    evaluation/                   # Model evaluation
      cv.py                       # CVEvaluator
      metrics.py                  # Metric functions + MetricRegistry
      model_factory.py            # ModelFactory + ModelRegistry
      sandbox.py                  # SandboxedExecutor
    experiment/                   # Experiment tracking
      runner.py                   # ExperimentRunner (serial + parallel)
      tracker.py                  # ExperimentTracker ABC, NoOpTracker
      wandb_backend.py            # WandB tracker
      mlflow_backend.py           # MLflow tracker
      reporter.py                 # Reporter (markdown tables)
      matrix.py                   # Config matrix builder
    artifacts/                    # Artifact storage & export
      base.py                     # ArtifactExporter ABC, ArtifactConfig
      storage.py                  # DataFrameStorage, LazyDataFrameRef
      schema.py                   # Pydantic schemas (ArtifactBundle, etc.)
      diff.py                     # Artifact diffing
      comparison.py               # Method comparison
      dashboard.py                # Dashboard generation
    data/                         # Dataset handling
      registry.py                 # DatasetRegistry
      ingestion.py                # Data loading & validation
    observability/                # Logging & tracing
      structlog_config.py         # Structlog setup, get_logger()
      langfuse_tracer.py          # Langfuse tracing
  tests/                          # Test suite
    conftest.py                   # Shared fixtures (FakeLLM, sample data)
    strategies.py                 # Hypothesis strategies for property tests
    unit/                         # Fast isolated tests
    integration/                  # Cross-module / plugin discovery tests
    benchmarks/                   # Performance smoke tests
  config/                         # Configuration files
    settings.yaml                 # Default settings
    logging.yaml                  # Logging config
    experiments/                  # Experiment YAML configs
  scripts/                        # Dev/CI scripts
    check_repo_hygiene.py
    check_docs_references.py
    run_pip_audit.py
  docs/                           # MkDocs documentation source
  notebooks/                      # Quarto notebooks (.qmd)
  data/                           # Raw and sample datasets
    raw/
    samples/
  experiments/                    # Experiment output directories
  memory_files/                   # Agent memory persistence
    agent_memories/
    llm_cache/
```

## Naming Conventions

### Python

| Element | Convention | Example |
|---------|-----------|---------|
| Package dirs | `snake_case` | `methods/malmas/`, `llm/providers/` |
| Module files | `snake_case` | `model_factory.py`, `structlog_config.py` |
| Classes | `PascalCase` | `FeatureForge`, `CVEvaluator`, `BaseMethod` |
| Abstract classes | `Base` or `ABC` prefix | `BaseMethod`, `Agent(ABC)`, `ArtifactExporter(ABC)` |
| Pydantic models | `PascalCase` | `FeatureSpec`, `LLMConfig`, `ArtifactBundle` |
| Functions / methods | `snake_case` | `run_coro_sync()`, `evaluate_baseline()` |
| Private methods | `_leading_underscore` | `_build_user_prompt()`, `_validate_code_ast()` |
| Constants | `UPPER_SNAKE_CASE` | `_BANNED_IMPORTS`, `_SINGLE_AGENT_MODES` |
| Type aliases | `PascalCase` | `DatasetName`, `MetricName`, `TaskType` |
| NewType aliases | `PascalCase` | `AgentName`, `Seed`, `RoundNumber` |
| Entry point groups | `dot.separated` | `feature_forge.methods`, `feature_forge.metrics` |
| Test files | `test_<module>.py` | `test_agents.py`, `test_pipeline_core.py` |
| Fixtures | `snake_case` | `fake_llm`, `sample_config`, `sample_dataframe` |
| Prompt YAML files | `snake_case.yaml` | `unary.yaml`, `code_generation.yaml` |
| Config YAML files | `snake_case.yaml` | `settings.yaml`, `logging.yaml` |

### YAML (Prompts)

| Element | Convention | Example |
|---------|-----------|---------|
| Filename | `<agent_or_purpose>.yaml` | `unary.yaml`, `router.yaml` |
| Top-level keys | `system`, `description` | `system: "You are..."` |

## Python Patterns

### Configuration (pydantic-settings)
- Root `Settings(BaseSettings)` with `env_prefix="FF_"` and `env_nested_delimiter="__"`
- Nested config via `BaseModel` subclasses: `LLMConfig`, `TrackerConfig`, `RouterConfig`, `MemoryConfig`, `RetryConfig`, `EvaluationConfig`
- Validation via `@field_validator` class methods with `_validate_` prefix
- Config priority: constructor args > env vars (FF_*) > YAML (`config/settings.yaml`)
- Factory function `get_settings()` for convenience

### Method System (Plugin Architecture)
- `MethodProtocol` — `@runtime_checkable` Protocol for third-party methods (no import dependency)
- `BaseMethod(ArtifactExporter)` — abstract base with `fit()`/`transform()`/`fit_transform()` sklearn interface
- `MethodRegistry` — discovers methods via `importlib.metadata.entry_points(group="feature_forge.methods")`
- Each method lives in `methods/<name>/method.py` with a `method.py` entry point
- Method adapters wrap internal pipeline (e.g., `MALMASMethod` wraps `FeatureForge`)

### Agent System
- `Agent(ABC)` — abstract base with `generate()` async method
- `BaseFeatureAgent(Agent)` — concrete base with LLM interaction, prompt building, response parsing
- Each agent class sets `prompt_key` and `agent_name` class attributes
- `AgentRegistry` — discovers agents via entry points + `get_builtin_agents()`
- Agent prompts stored as YAML in `methods/<method>/prompts/<agent_name>.yaml`
- `PromptRegistry` lazily loads YAML and caches `Prompt(system, description)` objects

### LLM Client System
- `LLMClient(ABC)` — abstract with provider hooks: `_call_api()`, `_extract_content()`, `_extract_usage()`
- Public API: `complete()`, `complete_json()` — both with automatic retry via `_retry()`
- `LLMResponse` — structured response with `content`, `model`, token counts
- `create_llm_client(LLMConfig)` — factory with auto provider inference from model name
- Provider modules imported lazily via `importlib.import_module()`

### Pipeline Pattern
- `CorePipeline` — single-round: agents → code gen → sandbox → CV eval → select top-k
- `IterativePipeline` — multi-round orchestration
- Ablation variants in `pipeline/ablations.py`: `NoMemoryPipeline`, `SingleAgentPipeline`, etc.
- Async-first: all pipeline methods are `async def`, sync callers use `run_coro_sync()`
- `asyncio.Semaphore` for LLM rate limiting, `asyncio.gather()` for parallel agent execution

### Async/Sync Bridge
- `_AsyncBridge` in `utils.py` provides a daemon-thread event loop
- `run_coro_sync()` bridges sync callers into async code (handles already-running-loop case)
- Used by `FeatureForge.fit()` and `Platform.run()`

### Exception Hierarchy
- `FeatureForgeError` root → domain-specific children: `LLMError`, `AgentError`, `PipelineError`, `EvaluationError`, `CodeExecutionError`, `DatasetError`, `ConfigurationError`, `TrackingError`
- Sandbox errors: `CodeExecutionError` → `SandboxValidationError`, `SandboxTimeoutError`, `SandboxWorkerError`

### Artifact System
- `ArtifactExporter(ABC)` mixin — `generated_scripts`, `intermediate_dataframes`, `feature_metadata`, `get_artifacts()`
- `ArtifactConfig` dataclass: `storage_mode` (memory/disk/hybrid), `storage_format` (parquet/csv/feather)
- `LazyDataFrameRef` for disk-backed DataFrames
- `ArtifactBundle` Pydantic model for validated serialization

### Logging
- `structlog` with `get_logger(__name__)` in every module
- Key-value structured events: `logger.info("event_name", key1=val1, key2=val2)`
- TTY → pretty console; non-TTY → JSON
- OpenTelemetry span injection via `add_open_telemetry_spans` processor
- Log level configurable via `FF_LOG_LEVEL` env var

### Imports
- `from __future__ import annotations` in every file
- stdlib → third-party → local (standard grouping)
- Heavy imports (providers, wandb, mlflow) are lazy — inside functions or via `importlib`
- `TYPE_CHECKING` guard for type-only imports

### Data Types
- `NewType` for domain strings: `DatasetName`, `MetricName`, `PromptName`, `AgentName`, `Seed`
- Pydantic `BaseModel` for structured data: `FeatureSpec`, `Prompt`, config models
- `Literal` for constrained strings: `TaskType`, `LLMProvider`, `RouterStrategy`

## Testing

### Python (pytest + hypothesis)
- **Runner**: `uv run pytest` from project root
- **Async**: `asyncio_mode = "auto"` — async test functions detected automatically
- **File naming**: `test_<module>.py` co-located in `tests/unit/` or `tests/integration/`
- **Test organization**: Class-based (`class TestFoo`) for related tests; standalone functions for simple cases
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.llm`, `@pytest.mark.property`, `@pytest.mark.metamorphic`, `@pytest.mark.contract`, `@pytest.mark.differential`, `@pytest.mark.baseline`
- **Auto-marking**: `conftest.py` auto-marks tests by directory (`unit/` → `unit`, `integration/` → `integration`)
- **Fixtures**: Defined in `conftest.py` — `FakeLLM`, `fake_llm`, `sample_config`, `sample_dataframe`, `sample_series`
- **Mocking**: `FakeLLM(LLMClient)` subclass with predetermined responses — no `unittest.mock`
- **Property testing**: Hypothesis strategies in `tests/strategies.py` — `pd_dataframes()`, `feature_specs()`, `binary_classification_data()`, `markdown_fenced_code()`, `valid_metrics()`
- **Coverage**: `--cov=feature_forge` with `--cov-report=term-missing` and `--cov-report=html`

## Linting & Formatting

### Python
- **Ruff** (linter + formatter): `target-version = "py311"`, `line-length = 100`
  - Rule set: E, F, I (isort), UP, B, C4, DTZ, T10, ISC, PIE, PT, RUF
  - `E501` ignored (line length handled by formatter)
  - `quote-style = "double"`, `indent-style = "space"`
  - `convention = "google"` for pydocstyle
- **mypy**: `strict = true`, `python_version = "3.11"`, `warn_return_any`, `warn_unused_ignores`
  - `ignore_missing_imports = true` for untyped third-party packages (sklearn, xgboost, wandb, etc.)
- **pre-commit hooks**:
  - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files` (max 1024KB)
  - `uv-lock` — keeps lockfile in sync
  - `ruff-check --fix` + `ruff-format` (excludes `notebooks/`)
  - `conventional-pre-commit` for commit messages
  - Local: `pip-audit`, `mypy` (excludes `tests/`), `quarto-render` for `.qmd` changes, `repo-hygiene`

## Build/Dev Commands

```
uv sync                                    → Install all dependencies from lockfile
uv run pytest                              → Run full test suite with coverage
uv run pytest tests/unit/                  → Run only unit tests
uv run pytest -m "not slow and not llm"    → Run tests excluding slow and LLM tests
uv run pytest -m property                  → Run only Hypothesis property tests
uv run ruff check                          → Lint all Python files
uv run ruff format                         → Format all Python files
uv run mypy src/                           → Type-check source (strict mode)
uv run pre-commit run --all-files          → Run all pre-commit hooks
make docs                                  → Build MkDocs documentation
make docs-serve                            → Serve docs with live reload
make notebooks                             → Render all Quarto notebooks
uv run python scripts/check_repo_hygiene.py → Check for tracked cache artifacts
uv run python scripts/run_pip_audit.py     → Security audit dependencies
```
