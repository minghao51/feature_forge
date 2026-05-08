# Dependencies & pyproject.toml

## Complete `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "feature-forge"
version = "0.1.0"
description = "Modular experimentation platform for LLM-based multi-agent automated feature engineering"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Feature Forge Team"},
]
keywords = [
    "feature-engineering",
    "llm",
    "machine-learning",
    "automl",
    "multi-agent",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    # Core
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",

    # Data
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",

    # ML Models
    "xgboost>=2.0",

    # LLM
    "openai>=1.0",

    # Caching
    "diskcache>=5.6",

    # Progress + CLI
    "tqdm>=4.65",
    "rich>=13.0",

    # Logging
    "structlog>=24.0",

    # Observability
    "langfuse>=2.0",
    "opentelemetry-api>=1.20",

    # Experiment Tracking (default)
    "wandb>=0.16",
]

[project.optional-dependencies]
# Optional functionality (baselines + alternative trackers)
base = [
    "openfe>=0.1",
    "caafe>=0.1",
    "mlflow>=2.10",
]

# Documentation tooling
docs = [
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
    "mike>=2.0",
]

# Opinionated dev tooling (linting, testing, notebooks)
opinion = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.23",
    "mypy>=1.0",
    "ruff>=0.3",
    "pre-commit>=3.6",
    "marimo>=0.5",
    "notebook>=7.0",
]

# Entry Points — Agents
[project.entry-points."feature_forge.agents"]
unary = "feature_forge.agents.unary:UnaryFeatureAgent"
cross_compositional = "feature_forge.agents.cross_compositional:CrossCompositionalAgent"
aggregation = "feature_forge.agents.aggregation:AggregationConstructAgent"
temporal = "feature_forge.agents.temporal:TemporalFeatureAgent"
local_transform = "feature_forge.agents.local_transform:LocalTransformAgent"
local_pattern = "feature_forge.agents.local_pattern:LocalPatternAgent"

# Entry Points — Baselines
[project.entry-points."feature_forge.baselines"]
openfe = "feature_forge.baselines.openfe:OpenFEBaseline"
caafe = "feature_forge.baselines.caafe:CAAFEBaseline"
llmfe = "feature_forge.baselines.llmfe:LLMFEBaseline"

[project.urls]
Homepage = "https://github.com/your-org/feature-forge"
Documentation = "https://feature-forge.readthedocs.io"
Repository = "https://github.com/your-org/feature-forge"
Issues = "https://github.com/your-org/feature-forge/issues"

# ─────────────────────────────────────────────────────────────
# Tool Configurations
# ─────────────────────────────────────────────────────────────

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # Pyflakes
    "I",    # isort
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "DTZ",  # flake8-datetimez
    "T10",  # flake8-debugger
    "ISC",  # flake8-implicit-str-concat
    "PIE",  # flake8-pie
    "PT",   # flake8-pytest
    "RUF",  # Ruff-specific rules
]
ignore = ["E501"]  # Line length handled by formatter

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true
show_error_codes = true

[[tool.mypy.overrides]]
module = [
    "openfe.*",
    "caafe.*",
    "xgboost.*",
    "wandb.*",
    "mlflow.*",
    "langfuse.*",
    "diskcache.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
pythonpath = ["src"]
addopts = [
    "-ra",
    "-q",
    "--strict-markers",
    "--cov=feature_forge",
    "--cov-report=term-missing",
    "--cov-report=html",
]
markers = [
    "slow: marks tests as slow (deselect with '-m not slow')",
    "llm: marks tests that call LLM APIs (expensive)",
    "baseline: marks tests requiring optional baseline packages",
    "integration: marks integration tests",
]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src/feature_forge"]
omit = [
    "*/tests/*",
    "*/test_*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

## Dependency Rationale

| Package | Purpose | Why This Version |
|---------|---------|-----------------|
| `pydantic>=2.0` | Data validation, settings | v2 has Rust core, much faster |
| `pydantic-settings>=2.0` | YAML + env var config | Official pydantic extension |
| `pyyaml>=6.0` | YAML parsing | Required by pydantic-settings |
| `openai>=1.0` | LLM API client | v1 is complete rewrite, async support |
| `pandas>=2.0` | Data manipulation | pyarrow backend, better perf |
| `numpy>=1.24` | Numerical ops | Minimum for pandas 2.0 |
| `scikit-learn>=1.3` | ML utilities, base classes | `BaseEstimator`, `TransformerMixin` |
| `xgboost>=2.0` | Gradient boosting | Primary evaluation model |
| `diskcache>=5.6` | LLM response cache | SQLite-backed, fast, reliable |
| `tqdm>=4.65` | Progress bars | Standard for Python |
| `rich>=13.0` | Pretty console output | Used by structlog dev renderer |
| `structlog>=24.0` | Structured logging | 2x faster than stdlib, OTel integration |
| `langfuse>=2.0` | LLM observability | `@observe` decorators, cost tracking |
| `opentelemetry-api>=1.20` | Distributed tracing | Standard, framework-agnostic |
| `wandb>=0.16` | Experiment tracking | Default backend, W&B Weave for LLM |
| `mlflow>=2.10` | Alt experiment tracking | Optional, for data sovereignty |
| `openfe>=0.1` | Baseline method | Strongest non-LLM baseline |
| `caafe>=0.1` | Baseline method | Context-aware LLM baseline |

## Development Dependencies

| Package | Purpose |
|---------|---------|
| `pytest>=8.0` | Testing framework |
| `pytest-cov>=4.0` | Coverage reporting |
| `pytest-asyncio>=0.23` | Async test support |
| `mypy>=1.0` | Static type checking |
| `ruff>=0.3` | Linting + formatting (replaces black, isort, flake8) |
| `pre-commit>=3.6` | Git hooks |
| `marimo>=0.5` | Reactive notebooks (stored as `.py`) |
| `notebook>=7.0` | Jupyter notebook support |

## Optional Dependency Groups

```bash
# Minimal install (core only)
uv pip install -e .

# With baselines
uv pip install -e ".[baselines]"

# With MLflow instead of WandB
uv pip install -e ".[mlflow]"

# Everything
uv pip install -e ".[all]"

# Development
uv pip install -e ".[dev]"

# Development + all features
uv pip install -e ".[all,dev]"
```
