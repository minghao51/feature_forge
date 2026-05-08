# Key Design Decisions

## 1. Configuration Engine: pydantic-settings

### Why
- Replaces `global_config.py` mutable globals
- Thread-safe, instance-based
- Validation at startup (fail fast)
- Environment variable override (CI-friendly)
- YAML defaults + `.env` secrets

### Implementation

```python
# src/feature_forge/config.py
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

class LLMConfig(BaseModel):
    model: str = "deepseek-chat"
    api_key: SecretStr = Field(default="...", alias="LLM_API_KEY")
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int = 4096
    cache_responses: bool = True  # ENFORCED DEFAULT — no YAML override

class TrackerConfig(BaseModel):
    backend: Literal["wandb", "mlflow", "none"] = "wandb"
    project: str = "feature-forge"
    entity: str | None = None

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FF_",
        env_nested_delimiter="__",
        yaml_file="config/settings.yaml",
    )
    task: Literal["classification", "regression"] = "classification"
    metric: str = "auc"
    n_rounds: int = 4
    llm: LLMConfig
    tracker: TrackerConfig
    random_state: int = 42

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )
```

### Priority Order
1. **Init args**: `Settings(task="regression")`
2. **Environment vars**: `FF_TASK=regression`, `FF_LLM__MODEL=gpt-4`
3. **`.env` file**: dotenvx encrypted secrets
4. **YAML file**: `config/settings.yaml` defaults

### Secrets: dotenvx
```bash
dotenvx set LLM_API_KEY "sk-..."
dotenvx set WANDB_API_KEY "..."
dotenvx encrypt
```

`.env` is encrypted and safe to commit. `.env.keys` is `.gitignored`.

---

## 2. LLM Response Caching: Enforced Default

### Why
- Re-running experiments costs $0
- Ablations that only change downstream logic don't re-call LLM
- Perfect reproducibility
- Prevents accidental $500 API bills

### Implementation

```python
# src/feature_forge/llm/cache.py
from diskcache import Cache
import hashlib
import json

class LLMCache:
    """Deterministic disk cache for LLM responses."""

    def __init__(self, cache_dir: str = "memory_files/llm_cache"):
        self.cache = Cache(cache_dir)

    def get_key(self, messages: list, model: str, temperature: float, max_tokens: int) -> str:
        content = json.dumps({
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        return self.cache.get(key)

    def set(self, key: str, value: dict) -> None:
        self.cache[key] = value
```

### Enforcement
- `LLMConfig.cache_responses: bool = True` — default in code
- No YAML override path for this field
- Explicit opt-out: `FF_LLM__CACHE_RESPONSES=false` env var only

---

## 3. Sandboxed Code Execution

### Why
- LLM-generated code is inherently untrusted
- Must prevent data exfiltration, file system access, arbitrary imports
- AST validation is deterministic and auditable

### Implementation

```python
# src/feature_forge/evaluation/sandbox.py
import ast
import builtins

class SandboxedExecutor:
    """AST-validated, restricted-builtin code execution."""

    FORBIDDEN_NAMES = {
        "eval", "exec", "compile", "open", "input",
        "__import__", "exit", "quit",
    }

    ALLOWED_BUILTINS = {
        "abs", "all", "any", "bool", "dict", "float", "int",
        "len", "list", "map", "max", "min", "range", "round",
        "sorted", "str", "sum", "tuple", "zip",
    }

    def execute(self, code: str, globals_dict: dict) -> dict:
        """Execute code with restricted builtins."""
        # Parse and validate AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise CodeExecutionError(f"Invalid syntax: {e}")

        self._validate_ast(tree)

        # Build restricted globals
        safe_globals = {
            "__builtins__": {name: getattr(builtins, name) for name in self.ALLOWED_BUILTINS},
        }
        safe_globals.update(globals_dict)

        local_vars = {}
        try:
            exec(code, safe_globals, local_vars)
        except Exception as e:
            raise CodeExecutionError(f"Execution failed: {e}") from e

        return local_vars

    def _validate_ast(self, tree: ast.AST) -> None:
        """Check AST for forbidden operations."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise CodeExecutionError("Imports not allowed in generated code")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.FORBIDDEN_NAMES:
                    raise CodeExecutionError(f"Forbidden function: {node.func.id}")
```

---

## 4. Plugin Architecture: Entry Points

### Why
- Core repo stays lightweight
- Research groups publish independent pip packages
- No code changes needed to add agents/baselines

### Implementation

```python
# src/feature_forge/agents/base.py
import importlib.metadata
from abc import ABC, abstractmethod

class Agent(ABC):
    """Abstract base for feature generation agents."""

    def __init__(self, name: str, config: "Settings"):
        self.name = name
        self.config = config
        self.memory = []

    @abstractmethod
    async def generate(self, X, y, context: dict) -> list["FeatureSpec"]:
        pass

class AgentRegistry:
    """Discover agents via Python entry points."""

    ENTRY_POINT_GROUP = "feature_forge.agents"

    @classmethod
    def discover(cls) -> dict[str, type[Agent]]:
        agents = {}
        for ep in importlib.metadata.entry_points(group=cls.ENTRY_POINT_GROUP):
            agents[ep.name] = ep.load()
        return agents

    @classmethod
    def get_builtin_agents(cls) -> dict[str, type[Agent]]:
        """Return built-in agents without entry point discovery."""
        from .unary import UnaryFeatureAgent
        from .cross_compositional import CrossCompositionalAgent
        # ... etc
        return {
            "unary": UnaryFeatureAgent,
            "cross_compositional": CrossCompositionalAgent,
            # ...
        }
```

### Entry Points in pyproject.toml
```toml
[project.entry-points."feature_forge.agents"]
unary = "feature_forge.agents.unary:UnaryFeatureAgent"
cross_compositional = "feature_forge.agents.cross_compositional:CrossCompositionalAgent"
# ... etc

[project.entry-points."feature_forge.baselines"]
openfe = "feature_forge.baselines.openfe:OpenFEBaseline"
caafe = "feature_forge.baselines.caafe:CAAFEBaseline"
llmfe = "feature_forge.baselines.llmfe:LLMFEBaseline"
```

---

## 5. Async Concurrency Model

### Why
- Agent-level parallelism speeds up multi-agent rounds
- LLM calls are I/O-bound and benefit from async
- Must prevent API rate limiting

### Implementation

```python
# Agent-level parallelism
async def run_round(self, agents: list[Agent], X, y):
    semaphore = asyncio.Semaphore(self.config.llm.max_concurrent_calls or 3)

    async def run_with_limit(agent):
        async with semaphore:
            return await agent.generate(X, y, context=self.memory.get_context())

    results = await asyncio.gather(*[run_with_limit(a) for a in agents])
    return results
```

---

## 6. Error Handling: Rich Exception Hierarchy

```python
# src/feature_forge/exceptions.py

class FeatureForgeError(Exception):
    """Base exception."""
    pass

class ConfigurationError(FeatureForgeError):
    """Invalid configuration."""
    pass

class LLMError(FeatureForgeError):
    """LLM API call failed."""
    pass

class FeatureGenerationError(FeatureForgeError):
    """Feature generation failed."""
    pass

class CodeExecutionError(FeatureForgeError):
    """Generated code execution failed."""
    pass

class AgentError(FeatureForgeError):
    """Agent operation failed."""
    pass

class DatasetError(FeatureForgeError):
    """Dataset loading or ingestion failed."""
    pass

class TrackingError(FeatureForgeError):
    """Experiment tracking failed."""
    pass
```

---

## 7. Type Safety

All public APIs use type hints. Key type aliases:

```python
# src/feature_forge/types.py
from typing import TypeVar, NewType
import pandas as pd

FeatureSpec = dict[str, any]  # {"name": str, "code": str, "logic": str}
AgentName = NewType("AgentName", str)
DatasetName = NewType("DatasetName", str)
MetricName = NewType("MetricName", str)
Seed = NewType("Seed", int)

# Generic type vars
T = TypeVar("T")
XType = TypeVar("XType", bound=pd.DataFrame)
YType = TypeVar("YType", bound=pd.Series)
```
