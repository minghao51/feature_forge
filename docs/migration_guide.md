# Migration Guide

## From MALMAS to Feature Forge

### Configuration

**Before (MALMAS):**
```python
import global_config
# Mutable global state
global_config.task = "classification"
global_config.metric = "auc"
```

**After (Feature Forge):**
```python
from feature_forge.config import Settings

# Immutable, validated, instance-based
settings = Settings(task="classification", metric="auc")
# Or via env var: FF_TASK=classification
```

**Secrets Management:**
- Non-sensitive defaults are now in `config/settings.yaml`
- Only secrets (API keys) go in `.env` (local, gitignored — never committed)
- Use `FF_LLM__API_KEY` for a single key across all providers, or set provider-specific keys (DEEPSEEK_API_KEY, OPENAI_API_KEY, etc.)

### LLM Client

**Before:**
```python
from main_demo.main_func import generate_response
response = generate_response(model, api_key, base_url, prompt, user_msg, temp)
```

**After:**
```python
from feature_forge.llm import DiskCache
from feature_forge.llm.providers import DeepSeekProvider

client = DeepSeekProvider(api_key="sk-...", cache=DiskCache(), tracing_enabled=True)
response = await client.complete(messages=[...])
```

### Agents

**Before:**
```python
# Agents were functions with hardcoded prompt paths
# No registry, no plugin system
```

**After:**
```python
from feature_forge.methods.malmas.agents import AgentRegistry

agents = AgentRegistry.get_builtin_agents()
agent = agents["unary"](config, llm_client)
```

### Memory

**Before:**
```python
from main_demo.memory import AgentMemory
memory = AgentMemory("unary", "project", "cache_dir", 0)
```

**After:**
```python
from feature_forge.methods.malmas.memory import AgentMemory

memory = AgentMemory("unary", "memory_files/unary_memory.json")
```

### Pipeline

**Before:**
```python
# Monolithic pipeline in notebooks
```

**After:**
```python
from feature_forge.api import FeatureForge

fe = FeatureForge()
fe.fit(X_train, y_train)
X_test_enhanced = fe.transform(X_test)
```

### Experiment Tracking

**Before:**
```python
# No built-in tracking
```

**After:** experiment tracking is opt-in and defaults to `none` (no
external tracker). `ExperimentalPlatform` builds the tracker from
`TrackerConfig` automatically, or you can pass one explicitly:

```python
from feature_forge.config import TrackerConfig
from feature_forge.experiment import create_tracker_from_config, WandBTracker

# Opt-in: default backend is "none"; wandb/mlflow require credentials
config = TrackerConfig(backend="wandb", project="feature-forge")
tracker = create_tracker_from_config(config)  # or WandBTracker(project="feature-forge")
```

`ExperimentalPlatform.run(tracker=...)` accepts the same instance to override
the configured backend.

## Baseline → Method Migration (v0.1 → v0.2)

The `baselines/` directory was restructured into `methods/` with each FE method as a self-contained sub-package.

### Class Renames

| Old | New |
|-----|-----|
| `Baseline` | `BaseMethod` |
| `BaselineProtocol` | `MethodProtocol` |
| `BaselineRegistry` | `MethodRegistry` |
| `OpenFEBaseline` | `OpenFEMethod` |
| `CAAFEBaseline` | `CAAFEMethod` |
| `LLMFEBaseline` | `LLMFEMethod` |
| `MalmusBaseline` | `MalmusMethod` |

### Import Path Changes

| Old | New |
|-----|-----|
| `feature_forge.baselines` | `feature_forge.methods` |
| `feature_forge.baselines.base` | `feature_forge.methods.base` |
| `feature_forge.baselines.openfe` | `feature_forge.methods.openfe.method` |
| `feature_forge.baselines.caafe` | `feature_forge.methods.caafe.method` |
| `feature_forge.baselines.llmfe` | `feature_forge.methods.llmfe.method` |
| `feature_forge.baselines.malmus` | `feature_forge.methods.malmus.method` |
| `feature_forge.agents` | `feature_forge.methods.malmas.agents` |
| `feature_forge.agents.base` | `feature_forge.methods.malmas.agents.base` |
| `feature_forge.agents.router` | `feature_forge.methods.malmas.agents.router` |
| `feature_forge.memory` | `feature_forge.methods.malmas.memory` |
| `feature_forge.pipeline` | `feature_forge.methods.malmas.pipeline` |

### Entry Point Group Changes

| Old | New |
|-----|-----|
| `feature_forge.baselines` | `feature_forge.methods` |
| `feature_forge.agents` | `feature_forge.methods.malmas.agents` |

### ExperimentalPlatform API Changes

```python
# Before
platform.run(datasets=["titanic"], baselines=["malmus"], models=["xgboost"])
platform.register_baseline("custom", MyBaseline)
platform.list_baselines()

# After
platform.run(datasets=["titanic"], methods=["malmus"], models=["random_forest"])
platform.register_method("custom", MyMethod)
platform.list_methods()
```

### ExperimentalPlatform Execution Seam Changes

- `ExperimentalPlatform.run()` now executes through `ExperimentCase` + `HamiltonLayerExecutor` + `ExecutionBackend`.
- Parallel mode uses process-pool execution via top-level `run_case(payload)`.
- `parallel=True` accepts registry-discovered methods and rejects instance-local `register_method(...)` methods due to process serialization boundaries.
- Run outputs are normalized to `ExperimentResult` shape and now include nullable `error`.

### Prompt Migration

Agent prompts migrated from `.txt` files to YAML-backed registry:

```python
# Before: loaded from file path relative to package
prompt_path = Path(__file__).parent / "prompts/unary.txt"

# After: loaded from method-local YAML registry
from feature_forge.methods.malmas.prompts import get_registry
prompt = get_registry().get("unary")
system_text = prompt.system
```

Prompt YAML files live in method packages (e.g., `src/feature_forge/methods/malmas/prompts/unary.v1.yaml`).

## Hamilton-only execution (landed)

The imperative legacy execution engine and the `legacy` artifact-policy branch
have been removed (ADR 0016); Hamilton is the sole execution engine. Before
upgrading to the removal release:

- Remove `FF_DATAFLOW__ENGINE=legacy`, YAML `dataflow.engine: legacy`, and
  `artifact_policy: legacy` overrides. The removal release rejects stale legacy
  configuration with an actionable migration error pointing here; it is never
  silently reinterpreted as Hamilton, and engines are never mixed in one
  attempt.
- Run evidence is published as verified Bronze/Silver/Gold/Platinum packages on
  the Hamilton path.
- The runtime rollback switch is gone. Operational rollback is a package/version
  downgrade to the last compatibility release — retain that release if you may
  need emergency rollback.
- The historical Hamilton/legacy parity evidence is preserved at
  `experiments/legacy_removal/2026-09-14/`.

## Execution failure policy and cancellation (additive, ADR 0017)

The optional `execution.failure_policy` setting controls the outer experiment
scheduler (plan 22). The default `continue` keeps scheduling every requested
case regardless of failed cases — no behavior change. `fail_fast` stops
scheduling new cases after the first terminal failed case result; in-flight
work always finishes safely (this is never a hard kill).

- YAML: `execution.failure_policy: continue` (default) or `fail_fast`
- Environment: `FF_EXECUTION__FAILURE_POLICY=fail_fast`
- Invalid values fail at configuration construction, before any dataset or
  provider work.

Runtime enforcement has landed (plan 22 PRs 1–4): sequential fail-fast
scheduling, bounded process-pool submission, and the cooperative cancellation
token are all active. Additions since PR 1:

- **Per-run override** — `ExperimentalPlatform.run(failure_policy=...)` wins
  over `settings.execution.failure_policy` for that invocation only, is
  recorded in tracker provenance, and never mutates the cached/global
  `Settings` instance.
- **Cooperative cancellation token** —
  `ExperimentalPlatform.run(cancellation_token=...)` accepts a
  parent-process `CancellationToken` (`feature_forge.experiment.execution`)
  checked at case boundaries on both the sequential and process paths. A case
  in flight when cancellation arrives keeps its real result; on the process
  path at most `max_workers` cases may still complete after a stop condition
  (bounded, documented — not hard termination).
- **Result `state` field** — result rows now carry an additive `state`
  (`succeeded` / `failed` / `cancelled`). Every requested case yields exactly
  one row in original matrix order, including cancelled never-started cases
  (no scores, no stage packages, redacted cancellation record). Existing
  `error` handling is unchanged.

**No migration action is required**: the default remains `continue`
on-error, row count/order is unchanged, and the new keyword arguments are
optional. See [Operations](operations.md#execution-failure-policy-and-cancellation),
`docs/decisions/0017-failure-policy-and-cancellation-contract.md`, and
`docs/plan/22_fail_fast_cancellation_contract.md`. Case-level retry is
explicitly out of scope: the `ResourceConfig.max_attempts` fields remain
dormant, and fail-fast evaluates only after a terminal result.

## Breaking Changes

1. **Global config removed** → Use `Settings()` instances
2. **Agent names changed** → Use `AgentRegistry` for discovery
3. **Memory path format** → Now uses single JSON file per agent
4. **LLM interface** → Now async with `LLMClient.complete()`
5. **Metrics** → Now in `feature_forge.evaluation.metrics`
6. **API Key Configuration** → Use `FF_LLM__API_KEY` for all providers, or set provider-specific keys directly. Keys are passed to LLM clients via config, not auto-propagated to environment variables.
7. **Baseline → Method rename** — `Baseline` → `BaseMethod`, `BaselineRegistry` → `MethodRegistry`, all `*Baseline` → `*Method`
8. **Import paths restructured** — `agents/`, `baselines/`, `memory/`, `pipeline/` merged into `methods/` namespace
9. **Entry point groups renamed** — `feature_forge.baselines` → `feature_forge.methods`, `feature_forge.agents` → `feature_forge.methods.malmas.agents`
10. **ExperimentalPlatform API** — `baselines=` → `methods=`, `list_baselines()` → `list_methods()`, `register_baseline()` → `register_method()`
11. **Parallel seam constraint** — `parallel=True` disallows instance-local `register_method(...)` methods; use entry-point/registry methods or set `parallel=False`.
12. **Result normalization** — Case results follow `ExperimentResult` shape with nullable `error`.
13. **Heavyweight methods/models are optional extras** — the core install ships the `random_forest` default model. Use the named `openfe`, `caafe`, `xgboost`, `lightgbm`, or `catboost` extra when needed; the `ExperimentalPlatform` default model is now `random_forest` (previously `xgboost`).
14. **Default model changed** — `ExperimentalPlatform.run(models=None)` and `ModelFactory.get_model(None, ...)` now resolve to `random_forest` instead of `xgboost`.
15. **Selection cap renamed** — `Settings.max_selected_features` replaces the misleading `min_effective` name. Constructor input using `min_effective` remains a migration alias, but serialized configuration uses the new name.
16. **MALMAS fit isolation** — unknown modes/agents now fail closed, and normal fits reset memory/router learning. Pass `warm_start=True` explicitly to reuse persisted learned state.
