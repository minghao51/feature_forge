# Feature Forge — Code Style & Conventions

## File Organization

### Where Things Go

```
src/feature_forge/              # Main package
  __init__.py                   # Public API re-exports (ExperimentalPlatform, __version__)
  api.py                        # Sklearn-compatible transformer (FeatureForge)
  platform.py                   # Experiment facade (ExperimentalPlatform)
  config.py                     # pydantic-settings (Settings, LLMConfig, etc.)
  types.py                      # Shared NewTypes, FeatureSpec, TypeVars
  exceptions.py                 # Exception hierarchy (FeatureForgeError → subclasses)
  utils.py                      # Shared utilities (run_coro_sync, strip_markdown_fences)
  methods/                      # Feature engineering methods (BaseMethod + registry)
    base.py                     # BaseMethod, MethodProtocol, MethodRegistry
    _prompting.py               # Shared PromptRegistry + Prompt model
    malmas/                     # Multi-agent method
      agents/                   # 6 feature agents + RouterAgent + AgentRegistry
        base.py                 # Agent (ABC), BaseFeatureAgent, AgentRegistry
        router.py               # RouterAgent
        unary.py                # UnaryFeatureAgent (example thin subclass)
      pipeline/                 # Core → Iterative + ablation variants
        core.py                 # CorePipeline (single-round orchestration)
        codegen.py              # CodeGenerator + _validate_code_ast
        result.py               # PipelineResult dataclass
        iterative.py            # BaseIterativePipeline → IterativePipeline
        ablations.py            # NoMemoryPipeline, SingleAgentPipeline, etc.
      memory/                   # 3-tier memory (procedural, feedback, conceptual)
      prompts/                  # YAML prompt templates (one per agent)
      types.py                  # AgentName NewType
    caafe/                      # CAAFE method (unified + fidelity variants)
    llmfe/                      # LLMFE method (single_shot + iterative)
    malmus/                     # Malmus method (JSON-mode structured output)
    openfe/                     # OpenFE method (non-LLM baseline wrapper)
  llm/                          # LLM abstraction layer
    base.py                     # LLMClient (ABC), LLMResponse
    factory.py                  # create_llm_client() provider factory
    cache.py                    # DiskCache with SHA-256 keys
    retry.py                    # Tenacity async retry builder
    providers/                  # DeepSeek, OpenAI, Anthropic, LiteLLM
  evaluation/                   # Model evaluation
    cv.py                       # CVEvaluator (k-fold cross-validation)
    sandbox.py                  # SandboxedExecutor + BANNED_IMPORTS policy
    metrics.py                  # Metric functions + MetricRegistry
    model_factory.py            # Model factories + ModelRegistry
  data/                         # Dataset management
    registry.py                 # DatasetRegistry (entry-point + Kaggle + local)
    ingestion.py                # Data loading and validation
  experiment/                   # Experiment harness
    tracker.py                  # ExperimentTracker ABC + NoOpTracker
    wandb_backend.py            # WandBTracker
    mlflow_backend.py           # MLflowTracker
    factory.py                  # create_tracker_from_config()
    reporter.py                 # Markdown report generation
    case_executor.py            # ExperimentCaseExecutor
    execution.py                # Sequential + ProcessPool adapters
  observability/                # Logging and tracing
    structlog_config.py         # structlog setup (TTY pretty / JSON)
    langfuse_tracer.py          # Langfuse integration
  artifacts/                    # Artifact storage
    base.py                     # ArtifactExporter ABC, ArtifactConfig
    storage.py                  # DataFrameStorage (memory/disk/hybrid)

config/                         # Non-secret configuration
  settings.yaml                 # Default settings (committed in plaintext)
  logging.yaml                  # Logging configuration
  experiments/                  # Experiment config files

tests/                          # Test suite
  conftest.py                   # Shared fixtures (FakeLLM, sample_data)
  strategies.py                 # Shared Hypothesis strategies
  unit/                         # Fast isolated tests (auto-marked `pytest.mark.unit`)
  integration/                  # Integration tests (auto-marked `pytest.mark.integration`)
  benchmarks/                   # Performance smoke tests

notebooks/                      # Jupyter notebooks + shared utils
scripts/                        # Utility scripts (lint, audit, hygiene)
docs/                           # MkDocs documentation source
data/                           # Data files
  raw/                          # Raw datasets
  samples/                      # Sample datasets
experiments/                    # Experiment ablation directories
.planning/                      # Agent context files (OVERVIEW, STYLE, STATE)
```

## Naming Conventions

### Python

| Element | Convention | Example |
|---------|-----------|---------|
| Package | `snake_case` | `feature_forge` |
| Module files | `snake_case.py` | `model_factory.py`, `cv.py`, `base.py` |
| Classes | `PascalCase` | `FeatureForge`, `BaseMethod`, `LLMClient` |
| Pydantic config models | `PascalCase` + `Config` suffix | `LLMConfig`, `RouterConfig`, `EvaluationConfig` |
| Pydantic data models | `PascalCase` | `FeatureSpec`, `Prompt`, `ArtifactBundle` |
| ABC classes | `PascalCase` + `Base` prefix or `ABC` suffix | `BaseMethod`, `BaseFeatureAgent`, `LLMClient(ABC)` |
| Registry classes | `PascalCase` + `Registry` suffix | `MethodRegistry`, `AgentRegistry`, `MetricRegistry` |
| Functions / methods | `snake_case` | `get_settings()`, `fit_transform()`, `create_llm_client()` |
| Private methods | `_leading_underscore` | `_build_user_prompt()`, `_extract_content()`, `_do_complete()` |
| Class-level constants | `_UPPER_SNAKE_CASE` | `_SINGLE_AGENT_MODES`, `_BUILTIN_PROMPT_AGENTS` |
| Module-level constants | `UPPER_SNAKE_CASE` | `ENTRY_POINT_GROUP` |
| NewTypes | `PascalCase` | `DatasetName`, `MetricName`, `AgentName`, `Seed` |
| Type aliases | `PascalCase` | `JSONValue`, `TaskType`, `TrackerBackend` |
| TypeVars | `PascalCase` (single letter or short) | `T`, `XType`, `YType` |
| Test files | `test_<module>.py` | `test_agents.py`, `test_property.py` |
| Test classes | `Test<Feature>` | `TestAgentRegistry`, `TestStripMarkdownFencesProperties` |
| Test methods | `test_<behavior>` | `test_idempotent()`, `test_get_builtin_agents()` |
| Pytest markers | `snake_case` | `@pytest.mark.slow`, `@pytest.mark.property` |
| Fixtures | `snake_case` | `fake_llm`, `sample_config`, `sample_dataframe` |
| Prompt YAML files | `snake_case.yaml` | `unary.yaml`, `cross_compositional.yaml` |
| Config YAML files | `snake_case.yaml` | `settings.yaml`, `logging.yaml` |
| Entry point groups | `dotted.path` | `feature_forge.methods`, `feature_forge.metrics` |

## Python Patterns

### Module Boilerplate
- Every source file starts with `from __future__ import annotations`
- Module docstring follows immediately (Google style)
- Imports grouped: stdlib → third-party → `feature_forge` internal
- `logger = get_logger(__name__)` at module level when logging is needed
- `if TYPE_CHECKING:` blocks for heavy or circular imports

### Pydantic Models
- `BaseModel` for data schemas (`FeatureSpec`, `Prompt`)
- `BaseSettings` for configuration (`Settings`) with `env_prefix="FF_"` and `env_nested_delimiter="__"`
- Sensitive fields use `SecretStr` (e.g., `api_key: SecretStr | None`)
- Validation via `@field_validator` classmethods with `@classmethod` decorator
- Private validator methods named `_validate_<field>` or `_empty_string_to_none`
- `model_config = SettingsConfigDict(...)` as class attribute on Settings
- `Field(default_factory=...)` for nested config sub-models

### Abstract Base Classes
- `ABC` + `@abstractmethod` for interface contracts
- `LLMClient` uses template method pattern: `_call_api()`, `_extract_content()`, `_extract_usage()` as hooks
- Concrete agents are generated dynamically via `_make_prompt_agent(name, prompt_key)` in `base.py`, not as separate class files
- Methods extend `BaseMethod` and implement `fit()`, `transform()`, `get_artifacts()`

### Plugin Discovery
- `MethodRegistry`, `AgentRegistry`, `MetricRegistry`, `ModelRegistry`, `DatasetRegistry` — all use `importlib.metadata.entry_points()`
- Entry points declared in `pyproject.toml` under `[project.entry-points."<group>"]`
- Registry classes use `ClassVar` for class-level caches with `_discovered: ClassVar[dict | None]`
- `get_builtin_*()` returns hard-coded methods; `get_all_*()` merges built-in + discovered
- `discover()` wrapped in `try/except` with `warnings.warn(..., RuntimeWarning)`

### Async/Sync Bridge
- `run_coro_sync()` in `utils.py` bridges async → sync using a daemon background event loop
- All LLM calls are async (`async def complete(...)`, `async def generate(...)`)
- Public sklearn API (`fit`, `transform`) is synchronous, calls `run_coro_sync(self.async_fit(...))`

### Structured Logging
- `get_logger(__name__)` returns a structlog `BoundLogger`
- Log events use `snake_case` keys: `logger.info("fit_start", mode=self.mode, latency_ms=...)`
- No f-string interpolation in log messages — all context passed as keyword args

### Error Handling
- Exception hierarchy: `FeatureForgeError` → `ConfigurationError`, `LLMError`, `FeatureGenerationError`, `CodeExecutionError`, `AgentError`, etc.
- `LLMError` wraps all provider-specific exceptions
- `tenacity` retry with exponential backoff for transient LLM failures

### Method Structure
- Each method lives in its own subdirectory: `methods/<method_name>/method.py`
- Method directories contain `method.py`, optional `prompts/` subdirectory
- Thin adapter pattern: `MALMASMethod` wraps `FeatureForge` to implement `BaseMethod`
- Iterative methods accumulate `_artifacts: dict[str, Any]` with `generated_code`, `iterations`, `gains`

### Prompt Templates
- YAML files in `prompts/` directory with `system:` and `description:` keys
- `PromptRegistry` lazy-loads and caches `Prompt` Pydantic models
- Agent class attribute `prompt_key` maps to YAML filename (without extension)

### Configuration Layering
- Priority (highest → lowest): constructor args → env vars (`FF_*`) → `.env` (dotenvx) → `config/settings.yaml`
- `Settings.settings_customise_sources()` defines the priority chain
- Sub-configs as nested `BaseModel`: `LLMConfig`, `TrackerConfig`, `RouterConfig`, `MemoryConfig`, `RetryConfig`, `EvaluationConfig`

## Testing

### Python (pytest + hypothesis)

- **Runner**: `uv run pytest` from project root
- **Async**: `asyncio_mode = "auto"` — async test functions work without decorators
- **File naming**: `test_<module>.py` co-located by domain in `tests/unit/` or `tests/integration/`
- **Test organization**: Class-based grouping (`TestAgentRegistry`, `TestStripMarkdownFencesProperties`) with `test_<behavior>` methods
- **Mocking**: `FakeLLM(LLMClient)` test double in `conftest.py` — overrides `_do_complete`, `_do_complete_json`, `_call_api`, `_extract_content`, `_extract_usage`
- **Fixtures**: Defined in `tests/conftest.py` — `fake_llm`, `sample_config`, `sample_dataframe`, `sample_series`
- **Auto-marking**: `pytest_collection_modifyitems` auto-applies `pytest.mark.unit` / `pytest.mark.integration` based on directory
- **Property testing**: Hypothesis strategies in `tests/strategies.py` (`pd_dataframes`, `feature_specs`, `binary_classification_data`, etc.)
- **Markers** (declared in `pyproject.toml`):
  - `unit` — fast isolated tests
  - `integration` — multi-component tests
  - `slow` — deselect with `-m not slow`
  - `llm` — calls real LLM APIs (expensive)
  - `baseline` — requires optional baseline packages
  - `property` — hypothesis-based property tests
  - `metamorphic` — metamorphic relation tests
  - `contract` — API surface contract tests
  - `differential` — reference comparison tests
- **Coverage**: `--cov=feature_forge --cov-report=term-missing --cov-report=html`

## Linting & Formatting

### Ruff
- **Config**: `[tool.ruff]` in `pyproject.toml`
- **Target**: Python 3.11
- **Line length**: 100 (E501 ignored)
- **Quote style**: double quotes
- **Indent**: spaces
- **Selected rules**: E, F, I, UP, B, C4, DTZ, T10, ISC, PIE, PT, RUF
- **Excluded**: `*.ipynb` files
- **Docstring convention**: Google (`convention = "google"`)

### Mypy
- **Config**: `[tool.mypy]` in `pyproject.toml`
- **Mode**: `strict = true`
- **Python version**: 3.11
- **Warnings**: `warn_return_any`, `warn_unused_ignores`, `warn_redundant_casts`
- **Ignore missing imports**: for third-party libs without stubs (openfe, xgboost, sklearn, wandb, etc.)
- **Skipped**: `tests/` directory (pre-commit hook excludes tests)

### Pre-commit
- trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files (1024 KB), check-merge-conflict, debug-statements
- `uv-lock` — keeps lockfile in sync
- `ruff-check --fix` + `ruff-format` (excludes notebooks/)
- `conventional-pre-commit` for commit messages
- Local hooks: `pip-audit`, `repo-hygiene`, `docs-references`, `mypy` (src only), `quarto-render` (notebooks)

## Build/Dev Commands

```
uv sync                                    → Install all dependencies
uv run pytest                              → Run full test suite with coverage
uv run pytest tests/unit/                  → Run unit tests only
uv run pytest -m "not slow and not llm"    → Run fast local tests
uv run pytest -m property                  → Run property-based tests only
uv run ruff check src/                     → Lint source code
uv run ruff check src/ --fix               → Lint and auto-fix
uv run ruff format src/                    → Format source code
uv run mypy src/                           → Type-check source code
uv run python -m feature_forge             → (Not applicable — library, not CLI)
make docs                                  → Build documentation site
make docs-serve                            → Serve docs locally with live reload
make notebooks                             → Render all notebooks
```
