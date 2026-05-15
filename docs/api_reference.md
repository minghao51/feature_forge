# API Reference

## Core Classes

### `FeatureForge`

Sklearn-compatible feature engineering transformer.

```python
from feature_forge.api import FeatureForge

fe = FeatureForge()
fe.fit(X_train, y_train)
X_test_enhanced = fe.transform(X_test)
```

**Parameters:**
- `config`: `Settings` instance (optional)
- `llm_client`: `LLMClient` instance (optional)
- `mode`: One of `'full'`, `'no_memory'`, `'no_router'`, or agent name

**Methods:**
- `fit(X, y)`: Run iterative feature engineering
- `transform(X)`: Apply generated features
- `fit_transform(X, y)`: Fit and return enhanced data

---

### `Settings`

Configuration engine using pydantic-settings.

```python
from feature_forge.config import Settings

settings = Settings()
print(settings.llm.model)  # 'deepseek-chat'
```

**Environment variable examples:**
- `FF_TASK=regression`
- `FF_LLM__MODEL=gpt-4`
- `FF_LLM__API_KEY=sk-...`

---

## LLM Layer

### `LLMClient`

Abstract base for LLM providers.

```python
from feature_forge.llm import LLMClient, DiskCache, LangfuseLLMWrapper
from feature_forge.llm.providers import OpenAIProvider, DeepSeekProvider, AnthropicProvider

client = DeepSeekProvider(api_key="sk-...")
cached_client = LangfuseLLMWrapper(client, cache=DiskCache())
```

**Providers:**
- `OpenAIProvider`: Any OpenAI-compatible API
- `DeepSeekProvider`: DeepSeek API
- `AnthropicProvider`: Claude API

**`DeepSeekProvider` additional parameters:**
- `thinking_enabled: bool = False` — Enable thinking/reasoning mode
- `reasoning_effort: str = "medium"` — One of `"low"`, `"medium"`, `"high"`, `"max"`

---

### Prompt Registry

YAML-driven prompt templates loaded from `config/prompts/*.yaml`.

```python
from feature_forge.prompts import get_registry

registry = get_registry()
prompt = registry.get("unary")
print(prompt.system)
print(prompt.description)
```

**`Prompt` model:**
- `system: str` (required) — System prompt text
- `description: str = ""` — Human-readable description

**Available prompts:** `unary`, `cross_compositional`, `aggregation`, `temporal`, `local_transform`, `local_pattern`, `router`, `code_generation`

---

## Agent System

### `Agent`

Abstract base for feature generation agents.

```python
from feature_forge.methods.malmas.agents import Agent, AgentRegistry
```

Agents are MALMAS-specific and discovered via the `feature_forge.methods.malmas.agents` entry-point group.

**Built-in agents:**
- `UnaryFeatureAgent`: Single-column transformations
- `CrossCompositionalAgent`: Cross-column features
- `AggregationConstructAgent`: Group-by aggregations
- `TemporalFeatureAgent`: Time-based features
- `LocalTransformAgent`: Quantile/rank/outlier transforms
- `LocalPatternAgent`: Distribution pattern features

### `RouterAgent`

Dynamic agent selection with strategies:
- `data_driven`: Based on dataset characteristics
- `performance_driven`: Based on historical gains
- `hybrid`: Union of both
- `llm`: LLM-based selection

---

## Memory System

### `AgentMemory`

Per-agent 3-tier memory:
- **Procedural**: Successful transforms
- **Feedback**: Feature gains/losses
- **Conceptual**: LLM-summarized rules

```python
from feature_forge.methods.malmas.memory import AgentMemory

memory = AgentMemory("unary", "memory_files/unary.json")
memory.record_procedure(["age"], "log", "age_log", "numerical", "log transform", 0)
memory.record_feedback("age_log", "auc", 0.05, True, 0, ["age"], "numerical")
memory.save()
```

---

## Methods

### `MethodProtocol`

```python
from feature_forge.methods.base import MethodProtocol
```

Runtime-checkable protocol. Any class with `name`, `fit()`, `transform()`, `fit_transform()`, `generated_scripts`, `feature_metadata`, `get_artifacts()` satisfies it — no imports needed.

### `BaseMethod`

```python
from feature_forge.methods.base import BaseMethod
```

Abstract base inheriting `ArtifactExporter`. Constructor takes `name` and optional `artifact_config`. Subclasses must implement `fit()` and `transform()`. Provides `fit_transform()`, `generated_scripts`, `get_artifacts()`.

### `MethodRegistry`

```python
from feature_forge.methods import MethodRegistry
```

Entry-point discovery via `feature_forge.methods` group.

- `get_builtin_methods()` → dict of 5 built-in methods
- `get_all_methods()` → built-in + entry-point discovered
- `discover()` → raw entry-point scan

Built-in methods: `malmas`, `openfe`, `caafe`, `llmfe`, `malmus`

```python
from feature_forge.methods import MethodRegistry

methods = MethodRegistry.get_builtin_methods()
for name, cls in methods.items():
    print(f'{name}: {cls.__name__}')
# malmas: MALMASMethod, openfe: OpenFEMethod, caafe: CAAFEMethod, llmfe: LLMFEMethod, malmus: MalmusMethod
```

---

## Evaluation

### `CVEvaluator`

Cross-validation feature evaluator.

```python
from feature_forge.evaluation import CVEvaluator

evaluator = CVEvaluator()
baseline = evaluator.evaluate_baseline(X, y)
gain = evaluator.evaluate_feature(X, y, new_features, baseline_score=baseline)
```

### `SandboxedExecutor`

Safe code execution for LLM-generated features.

```python
from feature_forge.evaluation import SandboxedExecutor

executor = SandboxedExecutor()
features = executor.execute(code, df)
```

---

## Experiment Harness

### `ExperimentMatrix`

Cartesian product experiment definitions.

```python
from feature_forge.experiment import ExperimentMatrix

matrix = (
    ExperimentMatrix()
    .datasets(["titanic"])
    .methods({"malmas": ["full"], "openfe": ["openfe"]})
    .seeds([0, 1, 2])
)
configs = matrix.generate()
```

### `ExperimentRunner`

Execute experiment configurations.

```python
from feature_forge.experiment import ExperimentRunner, NoOpTracker

runner = ExperimentRunner(tracker=NoOpTracker())
results = runner.run(configs, experiment_fn)
```

### `Reporter`

Generate comparison reports.

```python
from feature_forge.experiment import Reporter

reporter = Reporter(results)
print(reporter.to_markdown())
```

### `ExperimentalPlatform`

High-level experiment runner.

```python
from feature_forge import ExperimentalPlatform

platform = ExperimentalPlatform()
results = platform.run(
    datasets=["titanic"],
    methods=["malmus", "caafe"],
    models=["xgboost"],
)
```

**Methods:**
- `run(datasets, methods, models)` — Run experiments (param: `methods=`, not `baselines=`)
- `register_method(name, cls)` — Register a custom method where `cls` is `type[BaseMethod]`
- `list_methods()` — List available methods

Result dict key is `method` (not `baseline`).
