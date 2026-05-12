# Development Guide

## Tech Stack

**Project:** feature-forge v0.1.0 — Python >=3.11, hatchling build, uv package manager.

### Core Dependencies

| Package | Purpose |
|---|---|
| pydantic >=2.0 | Data models, validation |
| pydantic-settings >=2.0 | Settings with YAML + env var loading |
| openai >=1.0 | OpenAI/DeepSeek API client |
| litellm >=1.40 | Unified multi-provider LLM interface |
| pandas >=2.0, numpy >=1.24 | DataFrame + numerical computing |
| scikit-learn >=1.3 | ML models, CV, sklearn-compatible API |
| xgboost >=2.0 | Gradient boosting for evaluation |
| diskcache >=5.6 | SQLite-backed LLM response cache |
| structlog >=24.0 | Structured logging |
| langfuse >=2.0 | LLM observability / tracing |
| wandb >=0.16 | Experiment tracking (default) |
| pyarrow >=23.0 | Parquet I/O |

**Optional:** openfe, caafe (baselines), mlflow >=2.10 (alt tracker).

### Dev Dependencies

pytest >=8.0, pytest-cov >=4.0, pytest-asyncio >=0.23, mypy >=1.0 (strict), ruff >=0.3, pre-commit >=3.6, marimo >=0.5.

---

## Code Conventions

### Naming

| Element | Style | Example |
|---|---|---|
| Modules | `snake_case` | `model_factory.py` |
| Classes | `PascalCase` | `CorePipeline`, `AgentMemory` |
| Functions/methods | `snake_case` | `generate_code()` |
| Constants | `UPPER_SNAKE_CASE` | `FORBIDDEN_NAMES` |
| Private methods | `_leading_underscore` | `_build_user_prompt()` |
| Type aliases | `PascalCase` NewType | `AgentName`, `Seed` |
| Config models | `*Config` suffix | `LLMConfig`, `RouterConfig` |

### Formatting & Linting

Configured in `pyproject.toml`:
- **Ruff**: target py311, line-length=100, double quotes, space indent, Google docstrings. Rules: `E, F, I, UP, B, C4, DTZ, T10, ISC, PIE, PT, RUF`
- **mypy**: strict mode, Python 3.11 target. Missing import overrides for optional deps.
- **pre-commit**: trailing-whitespace, end-of-file-fixer, check-yaml, uv-lock, ruff lint+format, mypy, conventional commits, repo hygiene, quarto render

### Type Checking

- `from __future__ import annotations` in every file
- Modern union syntax: `str | None`, `dict[str, Any]`
- `NewType` for domain-specific type safety (`types.py`)
- Pydantic v2 with `field_validator` (not `@validator`)

### Imports Convention

- `from __future__ import annotations` as first import
- Stdlib, then third-party, then local (enforced by ruff `I`)
- Module-level logger: `logger = get_logger(__name__)`

### Error Handling

All exceptions inherit from `FeatureForgeError`:
```
FeatureForgeError
  ├── ConfigurationError
  ├── LLMError
  ├── FeatureGenerationError
  ├── CodeExecutionError
  │   ├── SandboxValidationError
  │   ├── SandboxTimeoutError
  │   └── SandboxWorkerError
  ├── AgentError
  ├── AgentMemoryError
  ├── DatasetError
  ├── TrackingError
  ├── EvaluationError
  └── PipelineError
```

Patterns: chain exceptions with `raise X(...) from exc`, best-effort by default (failures logged and skipped unless `fail_on_*` flags set).

### Logging

- **structlog** with OpenTelemetry span injection
- Logger via `get_logger(__name__)` at module level
- Key-value structured events: `logger.info("event_name", key=value)`
- Auto-detects TTY for pretty vs JSON output
- Level via `FF_LOG_LEVEL` env var or `config/logging.yaml`

### Async Patterns

- Async/await throughout the pipeline
- `asyncio.Semaphore` for concurrency limiting
- `asyncio.gather(..., return_exceptions=True)` for parallel agent execution
- `run_coro_sync()` utility for bridging sync/async contexts

### Design Patterns

- **Abstract base classes**: `Agent`, `LLMClient`
- **Template method**: `BaseFeatureAgent` → `generate()`, `LLMClient` → `_call_api()`/`_extract_content()`/`_extract_usage()` hooks
- **Registry pattern**: `AgentRegistry` with Python entry-point discovery
- **Factory pattern**: `ModelFactory`, `create_llm_client()`

### Shared Utilities

- `utils.py`: `strip_markdown_fences()`, `run_coro_sync()`, `_AsyncBridge`
- `types.py`: `NewType` aliases, `FeatureSpec`, `MemoryEntry`, type variables
- `exceptions.py`: Full exception hierarchy

---

## Testing

### Commands

```bash
uv run pytest                          # All tests with coverage
uv run pytest tests/unit/              # Unit tests only
uv run pytest tests/integration/       # Integration tests only
uv run pytest -m "not slow"            # Skip slow tests
uv run pytest -m "not llm"             # Skip LLM tests
```

### Markers (registered in pyproject.toml, `--strict-markers`)

| Marker | Purpose |
|---|---|
| `@pytest.mark.slow` | Slow-running tests |
| `@pytest.mark.llm` | Tests calling real LLM APIs |
| `@pytest.mark.baseline` | Tests requiring optional baseline packages |
| `@pytest.mark.integration` | Integration tests |

### Fixtures (tests/conftest.py)

- `fake_llm` — `FakeLLM` instance for deterministic LLM testing
- `sample_config` — minimal valid config dict
- `sample_dataframe` — small synthetic DataFrame
- `sample_series` — binary classification target

### Patterns

- Tests organized by feature area in classes (e.g., `class TestAgentRegistry`)
- `FakeLLM(LLMClient)` — imports from `tests/conftest.py`, returns predetermined responses
- `pytest.raises(ExceptionType, match="pattern")` for error testing
- No `unittest.mock` — hand-crafted fake implementations that pass mypy strict

### Coverage

Source: `src/feature_forge`. Reports: terminal (missing lines), HTML (`htmlcov/`). Excluded: pragma no cover, `__repr__`, `raise AssertionError`, `raise NotImplementedError`, `TYPE_CHECKING` blocks.

---

## LLM Integrations

### Providers

| Provider | SDK | Auth | Notes |
|---|---|---|---|
| DeepSeek | openai (AsyncOpenAI) | `DEEPSEEK_API_KEY` | OpenAI-compatible, native JSON mode |
| OpenAI | openai (AsyncOpenAI) | `OPENAI_API_KEY` | GPT-*, o1-* models |
| Anthropic | anthropic (AsyncAnthropic) | `ANTHROPIC_API_KEY` | System message extraction |
| LiteLLM | litellm >=1.40 | Provider-specific | 100+ providers as fallback |

### Provider Factory

`create_llm_client(config)` auto-detects provider from model name prefix or uses explicit `config.provider`. Only the needed provider module is imported.

---

## Observability

### Langfuse

Traces every LLM call, agent execution, sandbox execution, and pipeline span. Auth via `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.

### Weights & Biases (default tracker)

Auth via `WANDB_API_KEY`. Config: `config/settings.yaml` → `tracker.backend: "wandb"`.

### MLflow (optional)

Requires `mlflow` optional dep. Configurable `tracking_uri` for remote server.

---

## Environment Variables

### Secrets

| Variable | Purpose |
|---|---|
| `FF_LLM__API_KEY` | Unified LLM API key |
| `DEEPSEEK_API_KEY` | DeepSeek-specific key |
| `OPENAI_API_KEY` | OpenAI-specific key |
| `ANTHROPIC_API_KEY` | Anthropic-specific key |
| `WANDB_API_KEY` | Weights & Biases auth |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse observability |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle dataset access |

### Config Variables (non-secret, `FF_` prefix)

| Variable | Default | Purpose |
|---|---|---|
| `FF_TASK` | `classification` | ML task type |
| `FF_METRIC` | `auc` | Evaluation metric |
| `FF_N_ROUNDS` | `4` | Pipeline rounds |
| `FF_LLM__MODEL` | `deepseek-chat` | LLM model |
| `FF_LLM__TEMPERATURE` | `0.2` | Sampling temperature |
| `FF_LLM__MAX_TOKENS` | `4096` | Max response tokens |
| `FF_LOG_LEVEL` | `info` / `warning` | Logging level |

### Secret Management

**dotenvx**: `.env` (encrypted, committed), `.env.keys` (decryption key, gitignored). Use `dotenvx run --` before Python starts.

---

## CI/CD (GitHub Actions)

| Workflow | Trigger | Purpose |
|---|---|---|
| CI (`ci.yml`) | push/PR to main/develop | lint, typecheck, test (3.11/3.12/3.13), security audit, hygiene |
| Deploy Docs (`docs.yml`) | push to main | Quarto render + MkDocs build + GitHub Pages deploy |
| Benchmark (`benchmark.yml`) | Weekly Sunday + manual | Run performance benchmark suite |
