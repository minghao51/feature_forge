# Plan: Code Generation Improvements

**Date:** May 2026
**Status:** Draft
**Precedes:** N/A (addresses immediate pipeline reliability)

## Problem

The 2-step LLM pipeline (Agent → Specs → CodeGenerator → Python) has a high failure rate during Step 2 (code generation). Errors observed:

| Error Type | Example | Root Cause |
|---|---|---|
| Syntax (truncation) | `'(' was never closed (line 267)` | `max_tokens=4096` too low; LLM output truncated mid-expression |
| Syntax (truncation) | `unterminated string literal (line 241)` | Same |
| Runtime (dtype) | `'numpy.ndarray' has no attribute 'fillna'` | No dtype info in code gen prompt; LLM treats array as Series |
| Runtime (shape) | `fp and xp are not of the same length` | No data shape context provided |
| Runtime (return) | `generate_features must return a DataFrame, got NoneType` | LLM forgets `return result` in long output |
| Runtime (assignment) | `Cannot set a DataFrame with multiple columns to single column` | LLM mishandles multi-column intermediate |

## Root Causes

1. **Output truncation** — `max_tokens=4096` is insufficient for 12+ features (~250 lines). LLM cuts off mid-parenthesis.
2. **No data context in code generation** — `CodeGenerator` receives only `FeatureSpec` dicts (name/logic/transform). No dtypes, shape, or sample values.
3. **Thinking mode not enabled** — DeepSeek supports thinking mode via `reasoning_effort` + `extra_body`. Currently not configured.
4. **JSON mode not used for spec generation** — Agents use raw `complete()` + manual JSON parsing instead of `complete_json()` with `response_format: json_object`.
5. **Specs sent as Python repr** — `core.py:52` formats specs via f-string interpolation of Python dicts instead of clean JSON.

## Changes

### Phase 0: Prompt System (Separate YAML + Pydantic Validation)

**Problem**: Prompts are stored as 8 `.txt` files in `src/feature_forge/prompts/`, loaded via fragile `Path(__file__).parent / "../prompts" / filename` in 3 different places. No validation, no templating, no schema enforcement.

**Solution**: Migrate each prompt to its own `.yaml` file under `config/prompts/`, with a Pydantic `Prompt` model for validation and a `PromptRegistry` for loading. YAML handles multi-line strings naturally via `|` pipe syntax. Pydantic validates structure at load time.

#### 0.1 Create per-prompt YAML files in `config/prompts/`

Each prompt gets its own `.yaml` file. The YAML holds the raw content (template); Pydantic handles validation and formatting logic.

```yaml
# config/prompts/unary.yaml
system: |
  You are UnaryFeatureAgent, a feature engineering agent focused on
  generating derived features from a single field.
  ...
description: "Single-column feature generation"
```

```yaml
# config/prompts/code_generation.yaml
system: |
  You are CodeGenerationAgent, a feature engineering agent that
  specializes in generating pandas or numpy code snippets...
description: "Generate Python code from feature specs"
```

File mapping (one `.yaml` per current `.txt`):

| Current `.txt` | New `.yaml` |
|---|---|
| `src/feature_forge/prompts/unary.txt` | `config/prompts/unary.yaml` |
| `src/feature_forge/prompts/cross_compositional.txt` | `config/prompts/cross_compositional.yaml` |
| `src/feature_forge/prompts/aggregation.txt` | `config/prompts/aggregation.yaml` |
| `src/feature_forge/prompts/temporal.txt` | `config/prompts/temporal.yaml` |
| `src/feature_forge/prompts/local_transform.txt` | `config/prompts/local_transform.yaml` |
| `src/feature_forge/prompts/local_pattern.txt` | `config/prompts/local_pattern.yaml` |
| `src/feature_forge/prompts/code_generation.txt` | `config/prompts/code_generation.yaml` |
| `src/feature_forge/prompts/router.txt` | `config/prompts/router.yaml` |

#### 0.2 Create `Prompt` model and `PromptRegistry`

File: `src/feature_forge/prompts/__init__.py` (rewrite)

```python
import yaml
from pathlib import Path
from pydantic import BaseModel

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config" / "prompts"

class Prompt(BaseModel):
    """Single prompt with validated structure and template injection."""
    system: str
    description: str = ""

    def inject(self, **kwargs) -> str:
        """Format the system template with f-string style variables."""
        try:
            return self.system.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing prompt variable: {e}") from e

class PromptRegistry:
    """Load and cache individual prompt YAML files from config/prompts/."""

    def __init__(self, prompts_dir: Path = _PROMPTS_DIR) -> None:
        self._dir = prompts_dir
        self._cache: dict[str, Prompt] = {}

    def get(self, name: str) -> Prompt:
        if name not in self._cache:
            path = self._dir / f"{name}.yaml"
            if not path.exists():
                raise KeyError(f"Prompt '{name}' not found at {path}")
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            self._cache[name] = Prompt(**data)
        return self._cache[name]

# Module-level singleton
_registry: PromptRegistry | None = None

def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
```

Key design decisions:
- **Separate files** — each prompt is self-contained, easy to edit in isolation
- **Lazy loading** — only loads a YAML when first accessed, then caches
- **Pydantic validation** — `Prompt` validates that `system` is a non-empty string at load time
- **Template injection** — `inject(**kwargs)` supports `{variable}` placeholders (e.g. for future dynamic prompts)
- **No central index** — no need to register prompt names; filename IS the key

#### 0.3 Migrate all consumers

Replace all `Path(__file__).parent / "../prompts" / *.txt` reads with registry lookups:

| Consumer | Current | After |
|---|---|---|
| `agents/base.py:100-103` | `Path(__file__).parent / "../prompts" / self.prompt_filename` | `get_registry().get(self.prompt_key).system` |
| `pipeline/core.py:39-42` | `Path(__file__).parent / "../prompts/code_generation.txt"` | `get_registry().get("code_generation").system` |
| `agents/router.py:86-87` | `Path(__file__).parent / "../prompts/router.txt"` | `get_registry().get("router").system` |

Agent classes change from:
```python
prompt_filename: str = "unary.txt"  # file reference
```
To:
```python
prompt_key: str = "unary"  # registry key (= yaml filename without extension)
```

#### 0.4 Remove old `.txt` files

After migration, delete all `.txt` files from `src/feature_forge/prompts/` (keep `__init__.py`).

### Phase 1: Config & DeepSeek Provider

#### 1.1 Update `config/settings.yaml`

```yaml
llm:
  model: "deepseek-v4-flash"        # was: deepseek-chat (deprecated name)
  base_url: "https://api.deepseek.com"
  temperature: 0.2                   # note: ignored when thinking is enabled
  max_tokens: 32768                  # was: 4096
  cache_responses: true
  max_concurrent_calls: 3
  thinking_enabled: true             # NEW
  reasoning_effort: "medium"         # NEW: "low"|"medium"|"high"|"max"
```

#### 1.2 Add thinking/reasoning fields to `LLMConfig`

File: `src/feature_forge/config.py` — `LLMConfig` class

Add:
- `thinking_enabled: bool = False`
- `reasoning_effort: Literal["low", "medium", "high", "max"] = "medium"`

#### 1.3 Implement thinking mode in `DeepSeekProvider`

File: `src/feature_forge/llm/providers/deepseek.py`

Override `_call_api()` to inject thinking parameters when `thinking_enabled=True`:
- Pass `reasoning_effort` as parameter
- Pass `extra_body={"thinking": {"type": "enabled"}}` when thinking is on
- Handle `reasoning_content` in response (log it, don't expose to callers)

The `LLMClient` base needs a way to receive config-level thinking settings. Options:
- **Option A**: Pass `LLMConfig` to provider constructor (requires factory change)
- **Option B**: Add `set_thinking_config()` method on `LLMClient`

**Choice: Option A** — cleaner, config flows through factory naturally.

#### 1.4 Update `LLMClient` base and factory

File: `src/feature_forge/llm/base.py`, `src/feature_forge/llm/factory.py`

- Add `thinking_enabled` and `reasoning_effort` as optional constructor params on `LLMClient`
- Factory passes them from `LLMConfig`
- `LLMResponse` gains optional `reasoning_content: str | None` field

### Phase 2: Improve Code Generation Prompt & Input

#### 2.1 Enrich code generation prompt with pandas-first rules

File: `config/prompts/code_generation.yaml`

Add rules to the system template:
- The input `df` is a pandas DataFrame with columns of known dtypes (provided below).
- `df['col']` always returns a pandas Series. Chain `.fillna()` on Series, not on numpy arrays.
- When using `np.vectorize`, `np.interp`, etc. that return arrays, wrap with `pd.Series(..., index=df.index)`.
- Keep code concise: one feature per block, avoid unnecessary intermediate variables.
- Always end with `return result`.

#### 2.2 Pass data schema to `CodeGenerator`

File: `src/feature_forge/pipeline/core.py` — `CodeGenerator.generate_code()`

Add `schema` parameter:
```python
async def generate_code(
    self,
    specs: list[FeatureSpec],
    schema: dict[str, Any] | None = None,
    error_feedback: str | None = None,
) -> str:
```

Schema structure:
```json
{
  "shape": [175, 8],
  "columns": {
    "f0": {"dtype": "float64", "nullable": false},
    "f1": {"dtype": "float64", "nullable": false}
  },
  "index_type": "RangeIndex",
  "index_length": 175
}
```

Format specs as clean JSON (not Python repr):
```python
specs_json = json.dumps(specs_dump, indent=2, ensure_ascii=False)
user_prompt = f"Data schema:\n{schema_json}\n\nGenerate code for features:\n{specs_json}"
```

#### 2.3 Build schema from DataFrame

File: `src/feature_forge/pipeline/core.py` — `CorePipeline.run()`

Before calling `code_generator.generate_code()`, build schema from `X_train`:
```python
schema = {
    "shape": list(X_train.shape),
    "columns": {
        col: {"dtype": str(X_train[col].dtype), "nullable": bool(X_train[col].isna().any())}
        for col in X_train.columns
    },
    "index_type": type(X_train.index).__name__,
    "index_length": len(X_train),
}
```

### Phase 3: Per-Agent Code Generation Batching

#### 3.1 Split code generation by agent group

File: `src/feature_forge/pipeline/core.py` — `CorePipeline.run()`

Current: All specs → one `CodeGenerator` call → one `generate_features()` function.

Proposed: Group specs by `agent_name` → one `CodeGenerator` call per group → multiple `generate_features_N()` functions → concatenate.

```python
from itertools import groupby
specs_by_agent = {
    name: list(group)
    for name, group in groupby(all_specs, key=lambda s: s.agent_name)
}

all_code_parts = []
for agent_name, agent_specs in specs_by_agent.items():
    code = await self.code_generator.generate_code(
        agent_specs, schema=schema, error_feedback=...
    )
    all_code_parts.append((agent_name, code))
```

Benefits:
- Each LLM call generates ~2-4 features (~60-80 lines) — well within token limits
- Failure in one agent's code doesn't block others
- Easier debugging — know which agent produced bad code

Tradeoff:
- More LLM calls (6 vs 1) — but each is smaller/faster/cheaper
- Need to handle partial failures gracefully

#### 3.2 Sandbox execution per code block

File: `src/feature_forge/pipeline/core.py`

Instead of one sandbox call for all features, execute each agent's code separately:
```python
features_train_parts = []
for agent_name, code in all_code_parts:
    try:
        part = self.sandbox.execute(code, X_train)
        features_train_parts.append(part)
    except Exception as exc:
        logger.warning("agent_code_execution_failed", agent=agent_name, error=str(exc))
        continue

features_train = pd.concat(features_train_parts, axis=1) if features_train_parts else pd.DataFrame()
```

### Phase 4: Use JSON Mode for Agent Spec Generation

#### 4.1 Switch agents to `complete_json()`

File: `src/feature_forge/agents/base.py` — `BaseFeatureAgent.generate()`

Change from:
```python
response = await self.llm_client.complete(messages=messages, ...)
specs = self._parse_response(response.content)
```

To:
```python
response = await self.llm_client.complete_json(
    messages=messages,
    schema_description="JSON array of feature specs with base_columns and derived_features",
    temperature=...,
)
specs = self._parse_response(json.dumps(response))
```

This leverages DeepSeek's `response_format: json_object` for guaranteed valid JSON output.

Note: The current `_parse_response()` handles both list and dict responses. Need to verify compatibility.

## Implementation Order

| Step | Files | Risk | Impact |
|---|---|---|---|
| 0.1 | `config/prompts/*.yaml` | Low | Separate YAML files for all 8 prompts |
| 0.2 | `prompts/__init__.py` | Low | Prompt model + PromptRegistry |
| 0.3-0.4 | `agents/*.py`, `pipeline/core.py`, `agents/router.py` | Low | Migrate consumers, delete `.txt` files |
| 1.2 | `config.py` | Low | Foundation for thinking config |
| 1.1 | `config/settings.yaml` | Low | Increase max_tokens to 32K |
| 1.3-1.4 | `llm/base.py`, `llm/factory.py`, `llm/providers/deepseek.py` | Medium | Enable thinking mode |
| 2.1 | `config/prompts/code_generation.yaml` | Low | Better prompt rules (pandas-first, dtype) |
| 2.2-2.3 | `pipeline/core.py` | Low | Pass data schema to code gen |
| 3.1-3.2 | `pipeline/core.py` | Medium | Per-agent code generation |
| 4.1 | `agents/base.py` | Low | JSON mode for spec generation |

**Recommended execution order:** 0.1 → 0.2 → 0.3 → 0.4 → 1.2 → 1.1 → 2.1 → 2.2 → 2.3 → 1.3 → 1.4 → 3.1 → 3.2 → 4.1

Steps 1.1 + 2.1 + 2.2 + 2.3 address the immediate truncation and data-context issues. Steps 1.3-1.4 add thinking mode for better reasoning. Step 3 reduces batch size. Step 4 hardens spec parsing.

## Verification

After each step, re-run `notebooks/02_pipeline_deep_dive.ipynb` and check:
- No more `code_gen_ast_invalid` / `code_gen_ast_retry_failed` errors
- No more `sandbox_execution_retry` errors related to dtypes/shapes
- All 6 agents produce executable features
- Generated code length stays under token limit

## Files Touched

| File | Change |
|---|---|
| `config/prompts/*.yaml` | 8 new YAML files (migrated from `.txt`) |
| `config/settings.yaml` | max_tokens=32768, add thinking_enabled + reasoning_effort |
| `src/feature_forge/config.py` | Add thinking_enabled, reasoning_effort to LLMConfig |
| `src/feature_forge/prompts/__init__.py` | Rewrite: Prompt model + PromptRegistry + lazy loading |
| `src/feature_forge/prompts/*.txt` | Delete all 8 `.txt` files |
| `src/feature_forge/agents/base.py` | prompt_filename → prompt_key, use registry |
| `src/feature_forge/agents/unary.py` | prompt_key = "unary" |
| `src/feature_forge/agents/cross_compositional.py` | prompt_key = "cross_compositional" |
| `src/feature_forge/agents/aggregation.py` | prompt_key = "aggregation" |
| `src/feature_forge/agents/temporal.py` | prompt_key = "temporal" |
| `src/feature_forge/agents/local_transform.py` | prompt_key = "local_transform" |
| `src/feature_forge/agents/local_pattern.py` | prompt_key = "local_pattern" |
| `src/feature_forge/agents/router.py` | Use registry instead of Path read |
| `src/feature_forge/llm/base.py` | Add thinking params to LLMClient constructor, reasoning_content to LLMResponse |
| `src/feature_forge/llm/factory.py` | Pass thinking config from LLMConfig to provider |
| `src/feature_forge/llm/providers/deepseek.py` | Override _call_api() with thinking mode params |
| `src/feature_forge/pipeline/core.py` | Add schema param, per-agent batching, use registry |
