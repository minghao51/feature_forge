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
from feature_forge.agents import AgentRegistry, UnaryFeatureAgent

# Discover all agents
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
from feature_forge.memory import AgentMemory
memory = AgentMemory("unary", "memory_files/unary_memory.json")
```

### Pipeline

**Before:**
```python
# Monolithic pipeline in notebooks
```

**After:**
```python
from feature_forge.api import MALMASFeatureEngineer

fe = MALMASFeatureEngineer()
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

## Breaking Changes

1. **Global config removed** → Use `Settings()` instances
2. **Agent names changed** → Use `AgentRegistry` for discovery
3. **Memory path format** → Now uses single JSON file per agent
4. **LLM interface** → Now async with `LLMClient.complete()`
5. **Metrics** → Now in `feature_forge.evaluation.metrics`
