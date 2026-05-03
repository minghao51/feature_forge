# Directory Structure

```
feature_forge/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # ruff, mypy, pytest on every PR
│       └── benchmark.yml           # Nightly benchmark suite
│
├── config/
│   ├── settings.yaml               # Default hyperparameters, LLM configs
│   ├── logging.yaml                # structlog configuration
│   └── experiments/                # Per-experiment config overrides
│       └── titanic_ablation.yaml
│
├── data/
│   ├── samples/                    # Small sample datasets (<1MB each)
│   │   ├── titanic_sample.csv
│   │   └── house_prices_sample.csv
│   └── raw/                        # .gitignored downloaded datasets
│
├── notebooks/                      # Marimo notebooks (stored as .py)
│   ├── 01_agent_comparison.py
│   ├── 02_memory_ablation.py
│   └── 03_baseline_comparison.py
│
├── experiments/                    # Git-committed experiment definitions
│   ├── agent_isolation/
│   ├── memory_ablations/
│   └── router_strategies/
│
├── src/
│   └── feature_forge/
│       ├── __init__.py             # Public API exports
│       ├── _version.py             # Version string
│       ├── config.py               # Pydantic-settings engine
│       ├── exceptions.py           # Rich exception hierarchy
│       ├── types.py                # Shared type aliases
│       ├── api.py                  # MALMASFeatureEngineer (sklearn)
│       │
│       ├── agents/                 # Plugin-ready agents
│       │   ├── __init__.py
│       │   ├── base.py             # Agent ABC + AgentRegistry
│       │   ├── unary.py
│       │   ├── cross_compositional.py
│       │   ├── aggregation.py
│       │   ├── temporal.py
│       │   ├── local_transform.py
│       │   ├── local_pattern.py
│       │   └── router.py           # RouterAgent
│       │
│       ├── baselines/              # First-class baselines
│       │   ├── __init__.py
│       │   ├── base.py             # Baseline ABC + BaselineRegistry
│       │   ├── openfe.py
│       │   ├── caafe.py
│       │   └── llmfe.py
│       │
│       ├── llm/                    # Provider-agnostic LLM layer
│       │   ├── __init__.py
│       │   ├── base.py             # LLMClient ABC
│       │   ├── cache.py            # DiskCache (enforced default)
│       │   ├── providers/
│       │   │   ├── openai.py
│       │   │   ├── deepseek.py
│       │   │   └── anthropic.py
│       │   └── langfuse_wrapper.py # Auto-tracing + cost tracking
│       │
│       ├── memory/                 # Tiered memory system
│       │   ├── __init__.py
│       │   ├── base.py             # MemoryEntry + Memory ABC
│       │   ├── procedural.py
│       │   ├── feedback.py
│       │   ├── conceptual.py
│       │   └── persistence.py      # JSON/dill serializers
│       │
│       ├── evaluation/             # Feature evaluation & models
│       │   ├── __init__.py
│       │   ├── metrics.py          # AUC, ACC, NRMSE, custom
│       │   ├── cv.py               # K-fold CV evaluator
│       │   ├── model_factory.py    # XGB, LGB, CatBoost, RF, MLP
│       │   └── sandbox.py          # AST-validated execution
│       │
│       ├── pipeline/               # Orchestration
│       │   ├── __init__.py
│       │   ├── core.py             # Single-round pipeline
│       │   ├── iterative.py        # N-round with memory + router
│       │   └── ablations.py        # Isolated experiments
│       │
│       ├── experiment/             # Experiment harness
│       │   ├── __init__.py
│       │   ├── runner.py           # Execute experiment matrix
│       │   ├── matrix.py           # Cartesian product definitions
│       │   ├── tracker.py          # Unified tracking interface
│       │   ├── wandb_backend.py    # WandB implementation
│       │   ├── mlflow_backend.py   # MLflow implementation
│       │   └── reporter.py         # Auto markdown/HTML reports
│       │
│       ├── data/                   # Dataset adapters
│       │   ├── __init__.py
│       │   ├── base.py             # Dataset ABC
│       │   ├── loader.py           # Generic CSV + JSON loader
│       │   ├── ingestion.py        # Kaggle/OpenML fetchers
│       │   └── registry.py         # Built-in dataset registry
│       │
│       ├── observability/          # Logging + tracing
│       │   ├── __init__.py
│       │   ├── structlog_config.py # JSON/dev dual mode
│       │   └── langfuse_tracer.py  # @observe wrappers
│       │
│       └── prompts/                # Version-controlled prompt templates
│           ├── __init__.py
│           ├── unaryfeature.txt
│           ├── crosscompositional.txt
│           ├── aggregationconstruct.txt
│           ├── temporalfeature.txt
│           ├── localtransform.txt
│           ├── localpattern.txt
│           ├── codegeneration.txt
│           └── router.txt
│
├── tests/
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/                       # Fast, isolated tests
│   │   ├── test_config.py
│   │   ├── test_agents.py
│   │   ├── test_memory.py
│   │   ├── test_sandbox.py
│   │   └── test_llm_cache.py
│   ├── integration/                # Slower, component tests
│   │   ├── test_pipeline.py
│   │   ├── test_llm_providers.py
│   │   └── test_data_ingestion.py
│   └── benchmarks/                 # Full pipeline benchmarks
│       └── test_titanic_baseline.py
│
├── memory_files/                   # .gitignored runtime cache
│   ├── llm_cache/                  # LLM response cache
│   └── agent_memories/             # Persisted agent memories
│
├── docs/
│   ├── plan/                       # This implementation plan
│   └── api_reference.md            # Generated API docs
│
├── .env                            # .gitignored (dotenvx encrypted)
├── .env.keys                       # .gitignored (dotenvx keys)
├── pyproject.toml                  # Package metadata + tool config
├── uv.lock                         # Committed for determinism
└── README.md
```

## Rationale for Key Decisions

### `src/` Layout
Tests run against the **installed package**, not local source. This catches packaging errors early and matches modern Python best practices.

### `config/` Directory (3+ files)
Per `python-project-structure` skill: when you have 3+ config files, use a `config/` directory instead of root-level files.

### `notebooks/` as `.py` Files
Marimo notebooks stored as `.py` files are:
- Git-friendly (diffable)
- Importable as modules
- Runnable in VS Code Interactive Mode (`# %%`)

### `memory_files/` as `.gitignored`
Runtime artifacts (cache, memories) should never be committed. Clear separation between code and state.

### `data/samples/` Committed
Small sample datasets (<1MB) are committed so tests and quick demos work without downloading anything.

### `experiments/` Committed
Experiment configuration files (Python/YAML) are committed as they are the "source of truth" for reproducibility.

### `prompts/` in Package
Prompt templates are version-controlled alongside code. This enables:
- Git history for prompt changes
- A/B testing via prompt versions
- Langfuse prompt management integration
