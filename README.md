# Feature Forge

> Modular experimentation platform for LLM-based multi-agent automated feature engineering.

[![CI](https://github.com/minghao51/feature_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/minghao51/feature_forge/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Feature Forge is a production-ready refactoring of the MALMAS (Memory-Augmented LLM-based Multi-Agent System) research codebase into a modular, experiment-first Python package. It treats every method as a first-class, independently runnable, composable experiment unit.

## Key Features

- **6 Specialized Agents**: Unary, Cross-Compositional, Aggregation, Temporal, Local Transform, Local Pattern
- **3-Tier Memory**: Procedural, Feedback, and Conceptual memory with LLM summarization
- **Dynamic Router**: Data-driven, performance-driven, hybrid, and LLM-based agent selection
- **Enforced LLM Caching**: DiskCache with SHA-256 keys prevents accidental API costs
- **Sandboxed Execution**: AST-validated code execution for LLM-generated features
- **ExperimentalPlatform**: One-line experiment matrix — cartesian expansion of datasets × methods × seeds × models through the Hamilton-default engine; rounds remain settings-driven
- **Methods**: OpenFE [[2]](#ref-2), CAAFE [[3]](#ref-3), LLM-FE [[4]](#ref-4), Malmus, MALMAS (structured JSON, plugin arch)
- **Observability**: structlog + Langfuse + OpenTelemetry
- **Tracking**: Opt-in — defaults to `none` (no external tracker); WandB and MLflow available
- **Sklearn Compatible**: `FeatureForge` inherits `BaseEstimator` + `TransformerMixin`

## Installation

```bash
# Clone the repository
git clone https://github.com/minghao51/feature_forge.git
cd feature-forge

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

The core install ships the standard default model (`random_forest`) and the
first-party LLM methods. Heavyweight third-party methods/models are extras:

```bash
pip install 'feature-forge[openfe]'    # OpenFE baseline method
pip install 'feature-forge[caafe]'     # CAAFE fidelity variant
pip install 'feature-forge[xgboost]'   # XGBoost evaluation model
pip install 'feature-forge[lightgbm]'  # LightGBM evaluation model
pip install 'feature-forge[catboost]'  # CatBoost evaluation model
```

The standard installation includes Apache Hamilton, the sole case execution
engine. It publishes verified Bronze, Silver, Gold, and Platinum evidence and
keeps Hamilton caching independent of the mandatory LLM response cache. The
legacy imperative engine was removed per ADR 0016: stale `engine=legacy`
configuration fails fast with actionable migration guidance (see the
[migration guide](docs/migration_guide.md)), and operational rollback is a
downgrade to the prior compatibility release. Inspect a source-independent plan with, for example:

```bash
uv run feature-forge run plan --format json \
  --dataset titanic --method malmus --model random_forest
```

## Intel Acceleration (OpenMP)

On Intel hardware you can enable Intel Extension for Scikit-learn (`sklearnex`)
for DAAL-accelerated estimators. It is opt-in behind the `intel` extra and the
`intel_acceleration` setting (default on):

```bash
uv sync --extra intel            # installs scikit-learn-intelex
FF_INTEL_ACCELERATION=true uv run ...
```

`sklearnex` loads the Intel OpenMP runtime (`libiomp5`) while the XGBoost/
LightGBM wheels use `libgomp`. If you install the optional `xgboost` or
`lightgbm` extras alongside `intel`, the package patches sklearn **first** on
import, sets `KMP_DUPLICATE_LIB_OK=TRUE`, pins boosting estimators to
`n_jobs=1`, and keeps evaluation in the same process with the `threading`
backend (avoiding fork-after-OpenMP deadlocks) when acceleration is on. See
`docs/decisions/0010-intel-openmp-bootstrap.md`. Disable with
`FF_INTEL_ACCELERATION=false`.

## Quick Start

### Sklearn API

```python
from feature_forge.api import FeatureForge

fe = FeatureForge()
fe.fit(X_train, y_train)
X_test_enhanced = fe.transform(X_test)

# Use in a sklearn Pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ("fe", FeatureForge()),
    ("clf", RandomForestClassifier()),
])
pipeline.fit(X_train, y_train)
```

### ExperimentalPlatform (Recommended)

```python
from feature_forge import ExperimentalPlatform

platform = ExperimentalPlatform()

results = platform.run(
    datasets=["titanic", "house_prices"],
    methods=["malmus", "caafe", "openfe", "llmfe"],  # openfe needs the `openfe` extra
    models=["random_forest"],  # core default; xgboost needs the `xgboost` extra
    mode="single_shot",
    cv_folds=5,
)

print(platform.report(results))
# ┌──────────┬──────────┬───────────────┬──────────┬──────────┐
# │ dataset  │ method   │ model         │ cv_score │ gain     │
# ├──────────┼──────────┼───────────────┼──────────┼──────────┤
# │ titanic  │ malmus   │ random_forest │ 0.8523   │ +0.0312  │
# │ ...      │ ...      │ ...           │ ...      │ ...      │
# └──────────┴──────────┴───────────────┴──────────┴──────────┘

platform.report_best(results)     # best per dataset
df = platform.to_dataframe(results)  # raw pandas DataFrame
```

Notes:
- `parallel=True` uses a process-pool seam and supports registry-discovered methods.
- Instance-local methods added via `platform.register_method(...)` are not process-serializable and must run with `parallel=False`.
- Results are normalized to an `ExperimentResult`-like shape with a nullable `error` field.
- Scheduling supports continue (default) / `fail_fast` failure policies and cooperative case-boundary cancellation via `run(failure_policy=..., cancellation_token=...)`; cancellation is not hard termination — running cases finish. See [Operations](docs/operations.md).

### Experiment matrix

`ExperimentalPlatform.run()` expands the cartesian product of
`datasets × methods × seeds × models` internally and executes
each case through the Hamilton-default engine. No separate matrix/runner
classes are needed. See [Operations](docs/operations.md) for cache status,
garbage collection, and artifact list/verify commands.

### Custom Agent

```python
from feature_forge.methods.malmas.agents import BaseFeatureAgent

class DomainAgent(BaseFeatureAgent):
    prompt_key = "domain"
    agent_name = "domain"
```

Register in your `pyproject.toml`:
```toml
[project.entry-points."feature_forge.methods.malmas.agents"]
domain = "my_package:DomainAgent"
```

## Configuration

Configuration priority (highest to lowest):
1. Constructor arguments
2. Environment variables (`FF_*` prefix)
3. `.env` file (local, gitignored, plaintext — never committed)
4. YAML files (`config/settings.yaml`)

```bash
export FF_TASK=classification
export FF_LLM__MODEL=deepseek-chat
export FF_LLM__API_KEY=sk-...
export FF_TRACKER__PROJECT=my-project
```

## Architecture

```
Experiment Layer    → ExperimentalPlatform, HamiltonLayerExecutor, Tracker, Reporter
Methods Layer       → MethodRegistry, BaseMethod, 5 method packages (malmas, caafe, llmfe, malmus, openfe)
Pipeline Layer      → FeatureForge, CorePipeline, IterativePipeline
Agent Layer         → 6 Agents + Router + Registry (MALMAS-specific)
Memory Layer        → Procedural, Feedback, Conceptual (MALMAS-specific)
LLM Layer           → LLMClient, DiskCache, LangfuseWrapper
Evaluation Layer    → Metrics, CV, ModelFactory, Sandbox
Data Layer          → KaggleFetcher, DatasetRegistry
Observability Layer → structlog, Langfuse, OpenTelemetry
```

## Development

```bash
# Run all tests
uv run pytest

# With coverage report
uv run pytest --cov=feature_forge --cov-report=html

# Linting (required gate)
uv run ruff check src
# Optional debt track visibility
uv run ruff check notebooks docs/notebooks

# Type checking (required gate)
uv run mypy src
# Optional debt track visibility
uv run mypy tests

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Documentation

- [Methods & References](docs/methods.md) — Full documentation of all methods, pipelines, and academic sources
- [Implementation Plan](docs/plan/)
- [API Reference](docs/api_reference.md)
- [Migration Guide](docs/migration_guide.md)
- [Operations](docs/operations.md)
- [Generated Stage DAGs](docs/generated/stage_dags.md)
- [Quick Start](docs/quick_start.md)
- [MALMAS Technical Roadmap](docs/MALMAS_Technical_Roadmap.md)

## References

<a id="ref-1"></a>\[1\] **MALMAS** — "Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data"
MINE-USTC. arXiv:[2604.20261](https://arxiv.org/abs/2604.20261), ACL ARR 2026. [GitHub](https://github.com/MINE-USTC/MALMAS)

<a id="ref-2"></a>\[2\] **OpenFE** — "OpenFE: Automated Feature Generation with Expert-level Performance"
Zhang et al. ICML 2023. arXiv:[2211.12507](https://arxiv.org/abs/2211.12507). [GitHub](https://github.com/IIIS-Li-Group/OpenFE)

<a id="ref-3"></a>\[3\] **CAAFE** — "LLMs for Semi-Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering"
Hollmann, Müller, Hutter. NeurIPS 2023. arXiv:[2305.03403](https://arxiv.org/abs/2305.03403). [GitHub](https://github.com/noahho/CAAFE)

<a id="ref-4"></a>\[4\] **LLM-FE** — "LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers"
Abhyankar, Shojaee, Reddy. arXiv:[2503.14434](https://arxiv.org/abs/2503.14434), 2025. [GitHub](https://github.com/nikhilsab/llmfe)

## License

MIT
