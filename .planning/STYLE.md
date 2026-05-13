# Feature Forge — Code Style & Conventions

## File Organization

### Where Things Go

```
feature_forge/
  src/feature_forge/            # Main package (src layout)
    __init__.py                 # Version, logging init
    api.py                      # Sklearn-compatible public API (FeatureForge class)
    config.py                   # Pydantic-settings configuration hierarchy
    types.py                    # Shared NewTypes, TypeVars, type aliases
    exceptions.py               # Exception hierarchy (all inherit FeatureForgeError)
    utils.py                    # Shared utilities (async bridge, markdown stripping)
    agents/                     # Feature generation agents
      base.py                   # Agent ABC, BaseFeatureAgent, AgentRegistry
      router.py                 # RouterAgent (strategy-based agent selection)
      unary.py                  # Concrete agent: single-column transforms
      cross_compositional.py    # Concrete agent: multi-column features
      aggregation.py            # Concrete agent: group-by aggregations
      temporal.py               # Concrete agent: time-based features
      local_transform.py        # Concrete agent: local numeric transforms
      local_pattern.py          # Concrete agent: distributional pattern features
    llm/                        # LLM provider abstraction
      base.py                   # LLMClient ABC, LLMResponse dataclass
      factory.py                # create_llm_client() factory with auto-detection
      cache.py                  # LLM response caching
      retry.py                  # Tenacity-based async retry wrapper
      langfuse_wrapper.py       # Langfuse tracing integration
      providers/                # Per-provider implementations
        openai.py               # OpenAIProvider (base for OpenAI-compatible APIs)
        deepseek.py             # DeepSeekProvider (inherits OpenAIProvider)
        anthropic.py            # AnthropicProvider
        litellm_provider.py     # LiteLLMProvider (universal fallback)
    pipeline/                   # Pipeline orchestration
      core.py                   # CorePipeline (single-round: agents → code → sandbox → eval)
      iterative.py              # IterativePipeline (multi-round with memory/router)
      ablations.py              # Ablation variants (NoRouter, NoMemory, SingleAgent)
    evaluation/                 # Feature evaluation
      metrics.py                # Metric registry (auc, acc, f1, rmse, mae, r2, nrmse)
      cv.py                     # CVEvaluator (cross-validation scoring)
      sandbox.py                # SandboxedExecutor (safe code execution)
      model_factory.py          # Model instantiation for evaluation
    memory/                     # Agent memory system (3-tier: procedural/feedback/conceptual)
      base.py                   # AgentMemory class
      persistence.py            # JSON file persistence
      conceptual.py             # LLM-summarized conceptual rules
    baselines/                  # Baseline method implementations
      base.py                   # Baseline ABC, BaselineRegistry
      openfe.py                 # OpenFE baseline
      caafe.py                  # CAAFE baseline
      llmfe.py                  # LLMFE baseline
      malmus.py                 # Malmus baseline
    artifacts/                  # Artifact collection and storage
      base.py                   # ArtifactConfig, ArtifactExporter
      schema.py                 # Pydantic schema models (ArtifactBundle, ProvenanceRecord, etc.)
      storage.py                # DataFrameStorage (memory/disk/hybrid)
      diff.py                   # Artifact diff utilities
      comparison.py             # Cross-method artifact comparison
      dashboard.py              # Dashboard generation
    data/                       # Dataset loading and registry
      ingestion.py              # Data loading and validation
      registry.py               # Dataset registry
    experiment/                 # Experiment management
      runner.py                 # Experiment runner
      matrix.py                 # Experiment matrix (hyperparameter sweeps)
      tracker.py                # ExperimentTracker interface
      wandb_backend.py          # W&B backend
      mlflow_backend.py         # MLflow backend
      reporter.py               # Experiment report generation
    observability/              # Logging and tracing
      structlog_config.py       # Structlog configuration (TTY=pretty, CI=JSON)
      langfuse_tracer.py        # Langfuse tracing integration
    prompts/                    # Agent prompt templates (plain .txt files)
      unary.txt, aggregation.txt, etc.
  tests/                        # Test suite
    conftest.py                 # Shared fixtures (FakeLLM, sample_dataframe, etc.)
    unit/                       # Unit tests (one test file per source module)
    integration/                # Integration tests (pipeline end-to-end)
    benchmarks/                 # Performance smoke tests
  config/                       # Runtime configuration files
    settings.yaml               # Default settings (non-sensitive)
    logging.yaml                # Logging configuration
    experiments/                # Experiment matrix definitions
  data/                         # Data files (gitignored large files)
    raw/                        # Raw datasets
    samples/                    # Sample/small datasets for testing
  notebooks/                    # Marimo/Quarto notebooks (.py files)
  scripts/                      # Build/utility scripts
  docs/                         # Generated documentation (MkDocs)
  memory_files/                 # Persisted agent memories (JSON)
```

## Naming Conventions

### Python

| Element | Convention | Example |
|---------|-----------|---------|
| Package directories | `snake_case` | `feature_forge/`, `llm/providers/` |
| Module files | `snake_case.py` | `cross_compositional.py`, `structlog_config.py` |
| Classes | `PascalCase` | `FeatureForge`, `IterativePipeline`, `OpenAIProvider` |
| Abstract base classes | `PascalCase` with `ABC` suffix or `Base`/`Abstract` prefix | `Agent(ABC)`, `LLMClient(ABC)`, `Baseline(ABC)`, `BaseFeatureAgent` |
| Pydantic models | `PascalCase` | `Settings`, `LLMConfig`, `ArtifactBundle` |
| Functions | `snake_case` | `create_llm_client()`, `get_metric()`, `strip_markdown_fences()` |
| Async functions | `snake_case` with `async def` | `async def generate()`, `async def _call_api()` |
| Private methods | `_leading_underscore` | `_build_user_prompt()`, `_parse_response()` |
| Constants / registries | `UPPER_SNAKE_CASE` | `METRIC_REGISTRY`, `ENTRY_POINT_GROUP` |
| Type aliases | `PascalCase` | `FeatureSpec`, `AgentName`, `MemoryEntry` |
| NewType aliases | `PascalCase` | `AgentName = NewType("AgentName", str)` |
| Exceptions | `PascalCase` with `Error` suffix | `LLMError`, `PipelineError`, `SandboxTimeoutError` |
| Test classes | `PascalCase` with `Test` prefix | `TestAgentRegistry`, `TestRouterAgent` |
| Test functions | `snake_case` with `test_` prefix | `test_generate_parses_features()` |
| Pytest fixtures | `snake_case` | `fake_llm`, `sample_dataframe`, `sample_config` |
| Entry point groups | `snake_case` dotted | `feature_forge.agents`, `feature_forge.baselines` |
| Config env prefix | `UPPER_SNAKE_` with double-underscore nesting | `FF_LLM__MODEL`, `FF_LLM__API_KEY` |
| Prompt files | `snake_case.txt` | `unary.txt`, `code_generation.txt` |
| Class variables (config) | `UPPER_SNAKE_CASE` | `AGENT_CAPABILITIES`, `_BUILTIN_AGENT_MODULES` |

## Python Patterns

### Configuration
- **Pydantic-settings hierarchy**: `Settings(BaseSettings)` → `LLMConfig(BaseModel)`, `RouterConfig(BaseModel)`, etc. Nested configs are composed via `Field(default_factory=...)`.
- **Config priority** (highest to lowest): constructor args → env vars (`FF_*` prefix) → `.env` (dotenvx) → `config/settings.yaml`.
- **Validation**: `@field_validator` with `@classmethod` on every constrained field. Validators are private (`_validate_*` or `_empty_string_to_none`).
- **Factory function**: `get_settings(**overrides)` is the primary way to obtain configuration.
- **Secrets**: `SecretStr` for API keys. Never logged or exposed.

### ABC / Inheritance Pattern
- All extensibility points use `ABC` with `@abstractmethod`: `Agent.generate()`, `LLMClient._call_api()`, `Baseline.fit()` / `transform()`.
- **Template method**: Base classes define the public API (`complete()`, `generate()`) and subclasses override hooks (`_call_api()`, `_extract_content()`).
- **Concrete agents** inherit from `BaseFeatureAgent` and only set `prompt_filename` and `agent_name` class attributes (see `agents/unary.py:8-12`).
- **DeepSeekProvider** inherits from `OpenAIProvider` since DeepSeek is OpenAI-compatible (see `llm/providers/deepseek.py:16`).

### Registry Pattern
- **Entry-point-based discovery**: `AgentRegistry.discover()` and `BaselineRegistry.discover()` load from `importlib.metadata.entry_points()`.
- **Built-in registry**: `_BUILTIN_AGENT_MODULES` dict maps names to `"module.path:ClassName"` strings, loaded lazily via `_load_agent()`.
- **Metric registry**: Plain dict `METRIC_REGISTRY` mapping string names to callables.
- **Factory**: `create_llm_client()` uses `_PROVIDER_REGISTRY` dict for lazy module import.

### Async / Sync Bridge
- All core operations are `async`. `utils.py` provides `run_coro_sync()` which handles both "no running loop" (`asyncio.run()`) and "loop already running" (background thread via `_AsyncBridge`).
- `FeatureForge.fit()` is synchronous but calls `run_coro_sync(self.async_fit(...))`.

### Logging
- **structlog** everywhere. Obtain via `get_logger(__name__)` at module top.
- Log events use `snake_case` keys: `logger.info("pipeline_start", agents=[...], num_agents=6)`.
- TTY → pretty colored console. Non-TTY (CI, notebooks) → JSON.
- Level controlled by `FF_LOG_LEVEL` env var (default: `info` in TTY, `warning` otherwise).

### Exception Handling
- All custom exceptions inherit from `FeatureForgeError`.
- Exception types map to subsystems: `LLMError`, `AgentError`, `PipelineError`, `EvaluationError`, `CodeExecutionError`, etc.
- Base classes catch generic `Exception` and re-raise as domain-specific errors with `from exc` chaining.

### Pydantic Schemas
- `frozen=True` for immutable config-like models (`ArtifactConfigSchema`).
- `frozen=False` for mutable data models (`FeatureMetadata`, `ArtifactBundle`).
- `Field(default_factory=...)` for mutable defaults (lists, dicts).
- `@field_validator` for deduplication (`_dedup_scripts`), range checks, and coercion.

### Artifacts / Provenance
- `ArtifactExporter` base class provides `get_artifacts()` and `generated_scripts`.
- `ArtifactBundle` Pydantic model validates and stores all artifacts.
- `ProvenanceRecord` tracks full lineage per feature (method, agent, round, code, gain).

## Testing

### Framework & Runner
- **Runner**: `uv run pytest` from project root.
- **Framework**: pytest with `pytest-asyncio` (auto mode), `pytest-cov`.
- **Testpaths**: `tests/` (configured in `pyproject.toml`).
- **Pythonpath**: `src/` added via `pythonpath = ["src"]`.

### File Naming & Location
- **Unit tests**: `tests/unit/test_<module>.py` — one file per source module.
- **Integration tests**: `tests/integration/test_<module>.py`.
- **Benchmarks**: `tests/benchmarks/test_<name>.py`.
- **Fixtures**: `tests/conftest.py` for shared fixtures. Per-test fixtures defined inline in test classes.

### Test Organization
- **Class-based grouping**: Tests are organized in classes (`TestAgentRegistry`, `TestRouterAgent`, `TestAllAgentsInstantiate`).
- **Async tests**: Use `@pytest.mark.asyncio` decorator. `asyncio_mode = "auto"` in config.
- **Marker-based filtering**:
  - `@pytest.mark.slow` — deselect with `-m not slow`
  - `@pytest.mark.llm` — tests calling real LLM APIs
  - `@pytest.mark.baseline` — tests requiring optional baseline packages
  - `@pytest.mark.integration` — integration tests

### Mocking
- **FakeLLM**: A `FakeLLM(LLMClient)` test double is defined in `conftest.py` and duplicated in test files that need custom behavior. Returns predetermined responses.
- **FakeAgent**: `FakeAgent(Agent)` for tests that need deterministic specs.
- No `unittest.mock` used — preference for hand-crafted fakes.

### Coverage
- Target: `--cov=feature_forge` with `--cov-report=term-missing` and `--cov-report=html`.
- Exclusions in `coverage.report.exclude_lines`: `pragma: no cover`, `if TYPE_CHECKING:`, `raise NotImplementedError`, `__repr__`.

### CI Commands
```bash
uv run pytest                              # All tests
uv run pytest tests/unit/                  # Unit tests only
uv run pytest tests/integration/           # Integration tests only
uv run pytest -m "not slow"                # Exclude slow tests
uv run pytest -m "not llm"                 # Exclude LLM API tests
uv run pytest --cov=feature_forge          # With coverage
uv run pytest tests/benchmarks/            # Performance smoke tests
```

## Linting & Formatting

### Python
- **Ruff** (v0.15+): Combined linter + formatter (replaces flake8, isort, black).
  - Config in `pyproject.toml` under `[tool.ruff]`.
  - `target-version = "py311"`, `line-length = 100`.
  - Rule selection: `E, F, I, UP, B, C4, DTZ, T10, ISC, PIE, PT, RUF`.
  - `E501` ignored (line length handled by formatter).
  - `quote-style = "double"`, `indent-style = "space"`.
  - Docstring convention: `google`.
- **mypy** (strict mode): `python_version = "3.11"`, `strict = true`, `warn_return_any = true`.
  - Third-party ignores for `openfe`, `caafe`, `xgboost`, `wandb`, `mlflow`, `sklearn.*`, etc.
- **pre-commit** (`.pre-commit-config.yaml`):
  - `ruff-check --fix` + `ruff-format` (excludes `notebooks/`).
  - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files` (1MB limit), `debug-statements`.
  - `uv-lock` — keeps `uv.lock` in sync.
  - `conventional-pre-commit` — enforces Conventional Commits format.
  - `pip-audit` — security vulnerability scanning.
  - `mypy` — runs `uv run mypy src`.
  - `quarto-render` — auto-renders changed `.qmd` notebooks.
  - `repo-hygiene` — custom script checking no tracked cache/OS artifacts.

### Commands
```bash
uv run ruff check src/ tests/              # Lint
uv run ruff check --fix src/ tests/        # Lint with auto-fix
uv run ruff format src/ tests/             # Format
uv run mypy src                            # Type-check
pre-commit run --all-files                 # Run all pre-commit hooks
```

## Build/Dev Commands

```
uv sync                                    # Install all dependencies
uv run pytest                              # Run test suite
uv run pytest tests/unit/                  # Unit tests only
uv run pytest -m "not slow and not llm"    # Fast tests only
uv run ruff check src/ tests/              # Lint
uv run ruff format src/ tests/             # Format
uv run mypy src                            # Type-check
uv run python scripts/check_repo_hygiene.py  # Repo hygiene check
make docs                                  # Build documentation (quarto + mkdocs)
make docs-serve                            # Serve docs locally
make notebooks                             # Render all notebooks
uv add <package>                           # Add a dependency
uv run pip-audit --skip-editable           # Security audit
```
