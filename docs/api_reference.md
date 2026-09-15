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
- `mode`: One of `'full'`, `'no_memory'`, `'no_memory_static_router'`,
  `'no_router'`, or a registered agent name. Unknown values fail before an LLM call.
- `warm_start`: Reuse persisted MALMAS memory/router learning (`False` by default).
  Normal repeated fits start from fresh learned state.

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
print(settings.max_selected_features)  # per-round selection cap
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
from feature_forge.llm import LLMClient, DiskCache
from feature_forge.llm.providers import OpenAIProvider, DeepSeekProvider, AnthropicProvider

client = DeepSeekProvider(api_key="sk-...", cache=DiskCache(), tracing_enabled=True)
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

YAML-driven prompt templates loaded from method-local `prompts/*.yaml` packages.

```python
from feature_forge.methods.malmas.prompts import get_registry

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

Agents are MALMAS-specific and discovered via the `feature_forge.methods.malmas.agents` entry-point group. Agents are generated dynamically via `_make_prompt_agent()` and accessed by name through the registry.

**Built-in agent names:**
- `unary`: Single-column transformations
- `cross_compositional`: Cross-column features
- `aggregation`: Group-by aggregations
- `temporal`: Time-based features
- `local_transform`: Quantile/rank/outlier transforms
- `local_pattern`: Distribution pattern features

```python
from feature_forge.methods.malmas.agents import AgentRegistry

agents = AgentRegistry.get_builtin_agents()
agent_cls = AgentRegistry.get_agent("unary")
```

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
(openfe requires the optional `openfe` extra:
`pip install 'feature-forge[openfe]'`)

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

`ExperimentalPlatform` is the supported entry point for experiment matrices.
The standalone `ExperimentMatrix` / `ExperimentRunner` classes were removed
(see ADR 0006); matrix expansion is now built into `ExperimentalPlatform.run()`.

```python
from feature_forge import ExperimentalPlatform

platform = ExperimentalPlatform()
results = platform.run(
    datasets=["titanic"],
    methods=["malmas", "openfe"],  # openfe needs the `openfe` extra
    models=["random_forest"],
    seeds=[0, 1, 2],
)
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
    models=["random_forest"],  # default when models=None
)
```

**Methods:**
- `run(datasets, methods, models)` — Run experiments (param: `methods=`, not `baselines=`); `models=None` defaults to the core-installed `random_forest`
- `run(..., failure_policy=...)` — Optional per-run override of `settings.execution.failure_policy` (`"continue"` / `"fail_fast"`, ADR 0017). Under `fail_fast` the scheduler stops scheduling new cases after the first terminal failed result — identically on the sequential and process paths — and later cases are returned as typed cancelled results. The override wins only for that invocation, is recorded in tracker provenance, and never mutates the cached/global `Settings` instance.
- `run(..., cancellation_token=...)` — Optional parent-process `CancellationToken` checked at case boundaries on both paths. A case in flight when cancellation arrives finishes with its real result; never-started cases come back as cancelled rows. See [Operations](operations.md#execution-failure-policy-and-cancellation) for the cooperative-cancellation model.
- `register_method(name, cls)` — Register a custom method where `cls` is `type[BaseMethod]`
- `list_methods()` — List available methods

Behavior notes:
- `parallel=True` uses process-pool execution via top-level worker seam; methods must be registry-discovered (entry points / built-ins).
- `register_method(...)` methods are instance-local and rejected when `parallel=True`.
- `run(...)` returns normalized case results with keys:
  - `dataset`, `method`, `model`, `seed`
  - `cv_score`, `gain`, `baseline_score`, `num_features_generated`
  - `error` (nullable; set when case execution fails)
  - `state` (additive, ADR 0017): `"succeeded"` / `"failed"` / `"cancelled"`. Every requested case yields exactly one row in original matrix order, including cancelled never-started cases. Cancelled rows carry no scores and no stage packages; their `error` holds the redacted cancellation record. The default run remains continue-on-error with unchanged row count/order.

Result dict key is `method` (not `baseline`).

Engine notes:
- The default execution engine is Hamilton (`FF_DATAFLOW__ENGINE=hamilton`,
  per ADR 0016, which supersedes ADR 0012). Each case runs through `HamiltonLayerExecutor`, which publishes
  verified Bronze, Silver, Gold, and Platinum artifact packages under
  `experiments/artifacts`.
- The legacy imperative engine was removed per ADR 0016. Stale
  `FF_DATAFLOW__ENGINE=legacy` / `dataflow.engine: legacy` configuration fails
  fast with actionable migration guidance; the operational rollback path is a
  package/version downgrade to the last compatibility release, not a runtime
  engine switch.
- Hamilton caching is enabled by default and is independent of the mandatory LLM
  `DiskCache`. Cache deletion is safe — cases recompute from verified packages.
  See [Operations](operations.md) for cache status, GC, clear, and artifact
  list/verify commands.

### Execution Primitives

```python
from feature_forge.experiment import (
    ExperimentCase,
    ExperimentResult,
    ExecutionBackend,
    SequentialExecutionAdapter,
    ProcessPoolExecutionAdapter,
)
from feature_forge.experiment.execution import CancellationToken
```

- `ExperimentCase`: serializable run-unit dataclass.
- `ExperimentResult`: normalized output dataclass (includes nullable `error`).
  - `state`: explicit terminal state (`RunState.SUCCEEDED` / `FAILED` / `CANCELLED`); `None` keeps existing constructors back-compatible.
  - `resolved_state`: the effective terminal state — explicit `state` when set; otherwise derived as `failed` when an `error` or `failure` record is present and `succeeded` otherwise. `cancelled` is only ever set explicitly, for cases that never started, and an already-running case is never reclassified as cancelled after producing a real success or failure. Public serialization (`ExperimentalPlatform.run()` rows) always surfaces the resolved value.
- `ExecutionBackend`: backend seam interface.
- `SequentialExecutionAdapter` / `ProcessPoolExecutionAdapter`: concrete execution adapters.
  - `ProcessPoolExecutionAdapter.run(...)` uses a **bounded submission window** of at most `max_workers` futures, refilled only while the failure policy and token permit (ADR 0017, plan 22 §2.5). After a stop condition nothing new is submitted; queued futures are cancelled with `Future.cancel()` and, with the never-submitted tail, become typed cancelled results. Already-running workers finish and keep their real results — bounded cooperative cancellation, not hard termination. Original order and cardinality are preserved regardless of completion order, and the executor is shut down and joined before returning.
- `CancellationToken`: parent-process cooperative cancellation signal (import from `feature_forge.experiment.execution`).
  - `cancel(reason: str = "operator_request") -> None` — thread-safe and idempotent; the first reason wins.
  - `is_cancelled() -> bool` — whether cancellation was requested.
  - `reason: str | None` — the winning reason, or `None`.
  - `CancellationToken.cancelled(reason=...)` classmethod builds an already-cancelled token.
  - Parent-process only: the token is never serialized into `CaseComputationInput` and never crosses the process seam into a Hamilton DAG, sandbox, provider call, or cache key. Workers observe cancellation only through results returned by the parent.
  - A cancelled case carries a typed `FailureRecord` with `failure_class="cancelled"` (stable, non-sensitive `error_type` `"CaseCancelled"`), the parent-allocated case/attempt identity, no stage packages, and no fabricated score. Records are redacted: category, stable error type, and case identity only — never exception text, prompts, feature values, inputs, or secrets. One redacted `case_cancelled` lifecycle event is journaled per never-started case.
