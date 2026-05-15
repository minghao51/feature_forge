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
- Only secrets (API keys) go in `.env` (encrypted with dotenvx)
- Use `FF_LLM__API_KEY` for a single key across all providers, or set provider-specific keys (DEEPSEEK_API_KEY, OPENAI_API_KEY, etc.)

### LLM Client

**Before:**
```python
from main_demo.main_func import generate_response
response = generate_response(model, api_key, base_url, prompt, user_msg, temp)
```

**After:**
```python
from feature_forge.llm import LangfuseLLMWrapper, DiskCache
from feature_forge.llm.providers import DeepSeekProvider

client = LangfuseLLMWrapper(
    DeepSeekProvider(api_key="sk-..."),
    cache=DiskCache(),
)
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
from feature_forge.methods.malmas.agents import AgentRegistry, UnaryFeatureAgent

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

**After:**
```python
from feature_forge.experiment import ExperimentRunner, WandBTracker

tracker = WandBTracker(project="feature-forge")
runner = ExperimentRunner(tracker=tracker)
```

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
platform.run(datasets=["titanic"], methods=["malmus"], models=["xgboost"])
platform.register_method("custom", MyMethod)
platform.list_methods()
```

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

Prompt YAML files live in method packages (e.g., `src/feature_forge/methods/malmas/prompts/unary.yaml`).

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
