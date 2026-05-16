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
- **Experiment Matrix**: Cartesian product of datasets × methods × seeds × models × rounds
- **Methods**: OpenFE [[2]](#ref-2), CAAFE [[3]](#ref-3), LLM-FE [[4]](#ref-4), Malmus (structured JSON), MALMAS (multi-agent)
- **Observability**: structlog + Langfuse + OpenTelemetry
- **Tracking**: WandB (default) + MLflow (optional)
- **Sklearn Compatible**: `FeatureForge` inherits `BaseEstimator` + `TransformerMixin`

## Installation

```bash
# Clone the repository
git clone https://github.com/minghao51/feature_forge.git
cd feature-forge

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[base,docs,opinion]"
```

## Quick Start

### Sklearn API

```python
from feature_forge.api import FeatureForge

fe = FeatureForge()
fe.fit(X_train, y_train)
X_test_enhanced = fe.transform(X_test)

# Use in a sklearn Pipeline
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

pipeline = Pipeline([
    ("fe", FeatureForge()),
    ("clf", XGBClassifier()),
])
pipeline.fit(X_train, y_train)
```

### Experiment Matrix

```python
from feature_forge.experiment import ExperimentMatrix, ExperimentRunner, Reporter

matrix = (
    ExperimentMatrix()
    .datasets(["titanic", "house-prices"])
    .methods({"malmas": ["full"], "openfe": ["openfe"]})
    .seeds([0, 1, 2])
    .models(["xgboost", "lightgbm"])
    .rounds([1, 2, 4])
)

runner = ExperimentRunner()
results = runner.run(matrix.generate(), run_experiment)

reporter = Reporter(results)
print(reporter.to_markdown())
```

### Custom Method

```python
from feature_forge.methods.base import BaseMethod

class MyMethod(BaseMethod):
    def __init__(self):
        super().__init__("my_method")

    def fit(self, X_train, y_train):
        return self

    def transform(self, X):
        return pd.DataFrame({"my_feature": X.iloc[:, 0] * 2}, index=X.index)
```

Register in your `pyproject.toml`:
```toml
[project.entry-points."feature_forge.methods"]
my_method = "my_package:MyMethod"
```

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
3. `.env` file (dotenvx encrypted)
4. YAML files (`config/settings.yaml`)

```bash
export FF_TASK=classification
export FF_LLM__MODEL=deepseek-chat
export FF_LLM__API_KEY=sk-...
export FF_TRACKER__PROJECT=my-project
```

## Architecture

```
Experiment Layer    → ExperimentMatrix, ExperimentRunner, Tracker, Reporter
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

- [Methods & References](methods.md) — Full documentation of all methods, pipelines, and academic sources
- [Implementation Plan](plan/00_index.md)
- [API Reference](api_reference.md)
- [Migration Guide](migration_guide.md)
- [Quick Start](quick_start.md)
- [MALMAS Technical Roadmap](MALMAS_Technical_Roadmap.md)

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
