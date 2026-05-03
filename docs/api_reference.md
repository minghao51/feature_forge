# API Reference

## Core Classes

### `MALMASFeatureEngineer`

Sklearn-compatible feature engineering transformer.

```python
from feature_forge.api import MALMASFeatureEngineer

fe = MALMASFeatureEngineer()
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

---

## Agent System

### `Agent`

Abstract base for feature generation agents.

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
from feature_forge.memory import AgentMemory

memory = AgentMemory("unary", "memory_files/unary.json")
memory.record_procedure(["age"], "log", "age_log", "numerical", "log transform", 0)
memory.record_feedback("age_log", "auc", 0.05, True, 0, ["age"], "numerical")
memory.save()
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

---

## Baselines

### `Baseline`

Abstract base for baseline methods.

**Built-in baselines:**
- `OpenFEBaseline`: OpenFE wrapper
- `CAAFEBaseline`: CAAFE wrapper
- `LLMFEBaseline`: Simple LLM-based FE

```python
from feature_forge.baselines import OpenFEBaseline, LLMFEBaseline

openfe = OpenFEBaseline()
X_enhanced = openfe.fit_transform(X_train, y_train)
```
