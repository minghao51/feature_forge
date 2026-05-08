# Session Handoff Plan

**Slug:** feature-forge-phases-4-13
**Readable Summary:** Continue Feature Forge Implementation (Phases 4-13)
**Date:** 2026-05-02

---

## 1. Primary Request and Intent

The user wants to build **feature_forge**, a modular experimentation platform for LLM-based multi-agent automated feature engineering. It is a refactoring/enhancement of the MALMAS (Memory-Augmented LLM-based Multi-Agent System) research codebase into a production-ready Python package.

Key requirements:
- **Reference MALMAS codebase** at `/Users/minghao/Desktop/personal/MALMAS`
- **Reference technical roadmap** at `docs/MALMAS_Technical_Roadmap.md`
- Break down methods individually for isolated experimentation
- Be ambitious and aggressive in optimization
- Use modern Python project structure (`src/` layout, `uv`, `ruff`, `pytest`, `pydantic-settings`)
- **Dynamic data ingestion** from Kaggle (starting simple, scaling to multi-table)
- **Experiment tracking** with WandB default + MLflow optional from day one
- **Enforced LLM caching** by default to prevent accidental API costs
- **structlog** for structured logging + **Langfuse** for LLM observability
- Support both Python-first and YAML experiment configs

---

## 2. Key Technical Concepts

- **MALMAS Architecture**: Router Agent + 6 specialized agents (unary, cross_compositional, aggregation, temporal, local_transform, local_pattern) + 3-tier memory (procedural, feedback, conceptual)
- **Plugin Architecture**: Python entry points for agents and baselines
- **Pydantic-Settings**: YAML + env var configuration engine with `FF_` prefix
- **Sklearn Compatibility**: `MALMASFeatureEngineer` inherits `BaseEstimator` + `TransformerMixin`
- **Sandboxed Execution**: AST-validated code execution with restricted builtins
- **LLM Response Caching**: DiskCache with SHA-256 keys, enforced default ON
- **Experiment Matrix**: Cartesian product of datasets × methods × seeds × models × rounds
- **Baselines**: MALMAS + OpenFE + CAAFE + LLM-FE (top 4 methods per 2026 rankings)
- **Observability Stack**: structlog (JSON/dev dual mode) + Langfuse (@observe decorators) + OpenTelemetry
- **Tracking Stack**: WandB (default, free academic tier) + MLflow (optional, self-hosted)

---

## 3. Files and Code Sections

### Project Root

All files are in `/Users/minghao/Desktop/personal/feature_forge/`.

### docs/plan/ (Implementation Plan)

**Already written:**
- `00_index.md` — Overview, goals, decision log
- `01_architecture.md` — Layered architecture diagram, data flow, security model
- `02_directory_structure.md` — Complete directory tree
- `03_key_design_decisions.md` — Config engine, caching, sandboxing, plugins, async
- `04_implementation_phases.md` — 13-phase roadmap (40 days)
- `05_dependencies.md` — Complete `pyproject.toml` spec
- `06_data_strategy.md` — Kaggle-first data ingestion
- `07_observability.md` — structlog + Langfuse + OTel
- `08_experiment_tracking.md` — WandB + MLflow abstraction
- `09_baseline_selection.md` — Why MALMAS + OpenFE + CAAFE + LLM-FE

### Completed Phases 1-3

#### Phase 1: Skeleton & Tooling

**Files created:**
- `pyproject.toml` — Package metadata, dependencies, tool configs (ruff, mypy, pytest, coverage)
- `.gitignore` — Comprehensive ignore list including `.env.keys`, `memory_files/`, `wandb/`
- `.github/workflows/ci.yml` — GitHub Actions CI (ruff, mypy, pytest)
- `.pre-commit-config.yaml` — Pre-commit hooks (ruff, conventional commits)
- `README.md` — Basic quick start
- `config/settings.yaml` — Default configuration
- `config/logging.yaml` — structlog config override
- `tests/conftest.py` — Pytest fixtures
- All `__init__.py` files across `src/` and `tests/`

#### Phase 2: Config & Types

**`src/feature_forge/config.py`** — Pydantic-settings configuration engine:

```python
class LLMConfig(BaseModel):
    model: str = "deepseek-chat"
    api_key: SecretStr | None = Field(default=None)
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int = 4096
    cache_responses: bool = True  # ENFORCED DEFAULT
    max_concurrent_calls: int = 3

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FF_",
        env_nested_delimiter="__",
        yaml_file="config/settings.yaml",
    )
    task: Literal["classification", "regression"] = "classification"
    metric: str = "auc"
    n_rounds: int = 4
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
```

**`src/feature_forge/exceptions.py`** — Rich exception hierarchy:
- `FeatureForgeError` (base)
- `ConfigurationError`, `LLMError`, `FeatureGenerationError`, `CodeExecutionError`, `AgentError`, `MemoryError`, `DatasetError`, `TrackingError`, `EvaluationError`, `PipelineError`

**`src/feature_forge/types.py`** — Shared type aliases:
- `AgentName`, `DatasetName`, `MetricName`, `PromptName`, `Seed`, `RoundNumber`
- `FeatureSpec`, `MemoryEntry`, `TaskType`, `MetricType`, `RouterStrategy`, `TrackerBackend`, `LLMProvider`

**Tests:** `tests/unit/test_config.py` — 19 tests, all passing

#### Phase 3: Observability

**`src/feature_forge/observability/structlog_config.py`**:
- `configure_logging()` — Auto-detects TTY for pretty vs JSON output
- `add_open_telemetry_spans()` — Injects `trace_id`/`span_id` into log events
- `get_logger()` — Returns structlog logger

**`src/feature_forge/observability/langfuse_tracer.py`**:
- `get_langfuse()` — Lazy-init global Langfuse client
- `trace_agent()` — `@observe(as_type="agent")` decorator
- `trace_generation()` — `@observe(as_type="generation")` decorator
- `trace_tool()` — `@observe(as_type="tool")` decorator
- `trace_pipeline()` — `@observe(as_type="span")` decorator for root traces

**Tests:** `tests/unit/test_observability.py` — 5 tests, all passing

---

## 4. Problem Solving

### Solved:
1. **uv sync failure** — `openfe>=0.1` doesn't exist on PyPI. Changed to `openfe` (no version constraint) in `[project.optional-dependencies]`.
2. **pytest-cov not installed** — `uv sync --dev` didn't install dev dependencies on retry. Fixed with `uv pip install pytest-cov pytest-asyncio mypy ruff pre-commit`.
3. **SecretStr alias issue** — `Field(default=None, alias="LLM_API_KEY")` caused test failures because pydantic v2 requires alias for assignment by default. Removed alias (redundant with `env_nested_delimiter`).
4. **Ruff PIE790** — Exception classes with `pass` failed linting. Removed all `pass` statements from exceptions.
5. **RUF005** — List concatenation in structlog config. Fixed with iterable unpacking `[*shared_processors, ...]`.
6. **Langfuse import path** — `langfuse.decorators` doesn't exist in v4.5.1. Correct import is `from langfuse import observe`.

### Ongoing:
- None

---

## 5. Pending Tasks (Phases 4-13)

| Phase | Focus | Status |
|-------|-------|--------|
| **4** | LLM Layer — LLMClient ABC, DiskCache, providers (OpenAI, DeepSeek, Anthropic), Langfuse wrapper | **NOT STARTED** |
| **5** | Agent System — Agent ABC, Registry, Router, all 6 agents, prompt templates | **NOT STARTED** |
| **6** | Memory System — Procedural, Feedback, Conceptual memory, persistence | **NOT STARTED** |
| **7** | Evaluation — Metrics, CV, ModelFactory, sandboxed execution | **NOT STARTED** |
| **8** | Pipeline & API — Core pipeline, Iterative pipeline, Ablations, sklearn API | **NOT STARTED** |
| **9** | Baselines — OpenFE, CAAFE, LLM-FE implementations | **NOT STARTED** |
| **10** | Experiment Harness — Tracker abstraction, WandB/MLflow backends, Matrix, Reporter | **NOT STARTED** |
| **11** | Data Layer — Kaggle ingestion, sample datasets, registry | **NOT STARTED** |
| **12** | Tests & Docs — Unit/integration tests, marimo notebooks | **NOT STARTED** |
| **13** | Benchmarks — Full benchmark suite, README, migration guide | **NOT STARTED** |

---

## 6. Current Work

**Immediately before handoff:** Phase 4 (LLM Layer) was being set up. The todo list was updated to mark Phase 4 as `in_progress`. No Phase 4 files have been written yet.

The next files to implement are:
1. `src/feature_forge/llm/base.py` — LLMClient ABC with `complete()` method
2. `src/feature_forge/llm/cache.py` — DiskCache with SHA-256 key generation
3. `src/feature_forge/llm/providers/openai.py` — OpenAI provider
4. `src/feature_forge/llm/providers/deepseek.py` — DeepSeek provider
5. `src/feature_forge/llm/providers/anthropic.py` — Anthropic provider
6. `src/feature_forge/llm/langfuse_wrapper.py` — Auto-tracing wrapper around LLMClient
7. `tests/unit/test_llm_cache.py` — Tests for caching

---

## 7. Next Step

Implement Phase 4: LLM Layer. Start with `src/feature_forge/llm/base.py` defining the abstract base class:

```python
class LLMClient(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict], **kwargs) -> str: ...
```

Then implement `DiskCache`, then the three providers, then the Langfuse wrapper. Run `uv run ruff check src tests && uv run pytest tests/unit/test_llm_cache.py -v` after each file.

After Phase 4, proceed to Phase 5 (Agent System) which is the core of the system and will take the most time.

---

## How to Pick Up This Work

Use the pickup skill or command:
```
/pickup feature-forge-phases-4-13
```

Or read the handoff file:
```
/Users/minghao/Desktop/personal/feature_forge/.claude/handoffs/2026-05-02-feature-forge-phases-4-13.md
```

---

## Context References

- **MALMAS codebase**: `/Users/minghao/Desktop/personal/MALMAS`
- **Technical roadmap**: `/Users/minghao/Desktop/personal/feature_forge/docs/MALMAS_Technical_Roadmap.md`
- **Implementation plan**: `/Users/minghao/Desktop/personal/feature_forge/docs/plan/`
- **Project root**: `/Users/minghao/Desktop/personal/feature_forge`
