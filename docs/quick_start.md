# Quick Start

## Installation

```bash
# Clone the repository
git clone https://github.com/minghao51/feature_forge.git
cd feature-forge

# Install with uv
uv sync --all-extras

# Or with pip
pip install -e ".[all,dev]"
```

## Basic Usage

### Sklearn API

```python
from feature_forge.api import FeatureForge

fe = FeatureForge()
fe.fit(X_train, y_train)
X_test_enhanced = fe.transform(X_test)
```

### Experiment Matrix

```python
from feature_forge.experiment import ExperimentMatrix, ExperimentRunner

matrix = (
    ExperimentMatrix()
    .datasets(["titanic"])
    .methods({"malmas": ["full"], "openfe": ["openfe"]})
    .seeds([0, 1, 2])
)

runner = ExperimentRunner()
results = runner.run(matrix.generate(), run_experiment)
```

### Custom Agent

```python
from feature_forge.agents import Agent, BaseFeatureAgent

class MyAgent(BaseFeatureAgent):
    prompt_filename = "my_prompt.txt"
    agent_name = "my_agent"
```

Register in `pyproject.toml`:
```toml
[project.entry-points."feature_forge.agents"]
my_agent = "my_package:MyAgent"
```

## Configuration

Environment variables:
```bash
export FF_TASK=classification
export FF_LLM__MODEL=deepseek-chat
export FF_LLM__API_KEY=sk-...
```

Or edit `config/settings.yaml`.

## Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=feature_forge --cov-report=html

# Linting
uv run ruff check src tests
```
