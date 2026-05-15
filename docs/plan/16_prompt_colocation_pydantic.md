# Prompt Colocation & Pydantic-driven Prompt Rendering

**Date:** May 2026
**Status:** Draft
**Depends on:** `13_methods_restructure.md` (already executed)
**Related:** `12_code_generation_improvements.md`

## Motivation

Prompt text is currently split across `config/prompts/*.yaml` (8 files used by malmas agents) and hardcoded f-strings in `method.py` files (malmus, llmfe, caafe, memory). There is no validation of the variables injected into prompts — an `n_features: -1` silently produces a broken prompt.

**Problems:**
- Prompts live in two places: text in `config/prompts/`, variables in Python f-strings
- No validation bridge between YAML and runtime — typos, invalid values, missing params all fail at LLM call time
- `config/prompts/` is a misnamed directory — all 8 files are malmas-specific
- Malmas `prompts/__init__.py` is a dead stub with no callers
- Prompt text can drift from the code that uses it

**Solution:** Colocate YAML prompt files inside each method's package, and load them through per-method Pydantic models that validate both the YAML template structure AND the runtime injection parameters.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    YAML (template)                       │
│  methods/malmus/prompts/single_shot.yaml                 │
│  ─────────────────────────────                           │
│  system: "You are a {task} agent..."                     │
│  description: "..."                                      │
└─────────────────────┬───────────────────────────────────┘
                      │ yaml.safe_load()
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Pydantic: Prompt (validates YAML)           │
│  ─────────────────────────────                           │
│  class Prompt(BaseModel):                                │
│      system: str           # validated: non-empty       │
│      description: str = ""  # from YAML                 │
└──────────────┬──────────────────────────────────────────┘
               │ .format()
               ▼
┌─────────────────────────────────────────────────────────┐
│       Pydantic: *Params (validates injection params)     │
│  ─────────────────────────────                           │
│  class SingleShotParams(BaseModel):                      │
│      columns: str                                        │
│      task: Literal["classification", "regression"]       │
│      n_features: int = Field(ge=1, le=100)               │
│                                                          │
│      def render(self, template: str) -> str:             │
│          return template.format(...)                     │
└──────────────┬──────────────────────────────────────────┘
               │ .render(system_template)
               ▼
┌─────────────────────────────────────────────────────────┐
│              Final prompt string                         │
│  "You are a classification agent. Given a dataset..."    │
└─────────────────────────────────────────────────────────┘
```

## Two-layer Pydantic validation

### Layer 1: `Prompt` — validates YAML content

Sits in each method's `prompts/__init__.py`. Every prompt YAML is loaded into this model, ensuring the YAML has the required structure.

```python
class Prompt(BaseModel):
    system: str
    description: str = ""

    @field_validator("system")
    @classmethod
    def _system_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt system must be non-empty")
        return v
```

### Layer 2: `*Params` — validates injection parameters

Defined alongside the prompt loader. Each prompt template that has `{placeholders}` gets a matching Params model with typed fields.

```python
class MalmusSingleShotParams(BaseModel):
    columns: str
    task: Literal["classification", "regression"]
    n_features: int = Field(default=5, ge=1, le=100)

    def render(self, template: str) -> str:
        return template.format(
            columns=self.columns,
            task=self.task,
            n_features=self.n_features,
        )
```

## File layout

### Method packages own their prompts

```
src/feature_forge/methods/
├── malmas/
│   ├── prompts/
│   │   ├── __init__.py           # Prompt model + PromptRegistry (resolves locally)
│   │   ├── unary.yaml
│   │   ├── cross_compositional.yaml
│   │   ├── aggregation.yaml
│   │   ├── temporal.yaml
│   │   ├── local_transform.yaml
│   │   ├── local_pattern.yaml
│   │   ├── router.yaml
│   │   └── code_generation.yaml
│   └── memory/
│       └── prompts.py            # SummarizeAgentParams, SummarizeGlobalParams
│
├── malmus/
│   ├── prompts/
│   │   ├── __init__.py           # Prompt model + PromptRegistry + *Params models
│   │   ├── single_shot.yaml      # system: "..." with {columns} {task} {n_features}
│   │   └── iterative.yaml        # system: "..." with {columns} {task} {iteration} etc.
│   └── method.py                 # Loads prompts, constructs Params, calls .render()
│
├── llmfe/
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── single_shot.yaml
│   │   └── iterative.yaml
│   └── method.py
│
└── caafe/
    ├── prompts/
    │   ├── __init__.py
    │   └── unified.yaml          # {description} {iteration} {iterations} {existing} {feedback}
    └── method.py
```

Every method's `prompts/__init__.py` follows the same pattern:

```python
import importlib.resources
from pathlib import Path
import yaml
from pydantic import BaseModel, field_validator


_PROMPTS_DIR = Path(str(importlib.resources.files(__package__)))  # resolves to this package's dir


class Prompt(BaseModel):
    system: str
    description: str = ""

    @field_validator("system")
    @classmethod
    def _system_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt system must be non-empty")
        return v


class PromptRegistry:
    def __init__(self, prompts_dir: Path = _PROMPTS_DIR) -> None:
        self._dir = prompts_dir
        self._cache: dict[str, Prompt] = {}

    def get(self, name: str) -> Prompt:
        if name not in self._cache:
            path = self._dir / f"{name}.yaml"
            if not path.exists():
                raise KeyError(f"Prompt '{name}' not found at {path}")
            with open(path) as f:
                data = yaml.safe_load(f)
            self._cache[name] = Prompt(**data)
        return self._cache[name]

    def clear_cache(self) -> None:
        self._cache.clear()


_registry: PromptRegistry | None = None

def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
```

### Params models

Extend each `prompts/__init__.py` with injection-parameter models:

```python
class MalmusSingleShotParams(BaseModel):
    columns: str
    task: Literal["classification", "regression"]
    n_features: int = Field(default=5, ge=1)

    def render(self, template: str) -> str:
        return template.format(
            columns=self.columns,
            task=self.task,
            n_features=self.n_features,
        )
```

## Prompt YAML format

All prompts use `{placeholder}` syntax that maps to the Params model fields:

```yaml
# malmus/prompts/single_shot.yaml
system: |
  You are a feature engineering assistant. Given a dataset with columns: {columns},
  and a {task} task, generate {n_features} new features that could improve model
  performance.

  For each feature provide:
  - name: a valid Python identifier (snake_case)
  - code: a Python expression using 'df' as the input DataFrame
  - description: what the feature captures
  - libraries: list of required libraries (e.g. ['pandas', 'numpy'])

  The code expressions must use only pandas and numpy.
  Each expression will be assigned as: result['<name>'] = <code>
description: "Single-shot feature generation"
```

```yaml
# malmus/prompts/iterative.yaml
system: |
  You are a feature engineering assistant. Given a dataset with columns: {columns},
  and a {task} task, generate exactly ONE new feature (iteration {iteration}/{n_iterations}).
  Already created features: {existing_features}.

  For the feature provide:
  - name: a valid Python identifier (snake_case)
  - code: a Python expression using 'df' as the input DataFrame
  - description: what the feature captures
  - libraries: list of required libraries

  The code expression will be assigned as: result['<name>'] = <code>.

  {feedback}
description: "Iterative feature generation"
```

Variables with defaults or conditional content (like `feedback`) are marked with a trailing period/newline so the template renders cleanly even when empty.

## Usage in method code

### Static prompts (malmas agents — no runtime variables)

```python
# malmas/agents/base.py
from feature_forge.methods.malmas.prompts import get_registry

class BaseFeatureAgent(Agent):
    def __init__(self, config, llm_client):
        self._system_prompt = get_registry().get(self.prompt_key).system

    @property
    def system_prompt(self) -> str:
        return self._system_prompt
```

Agent subclasses only set `prompt_key` — no change from current pattern.

### Dynamic prompts (malmus, llmfe, caafe)

```python
# malmus/method.py
from feature_forge.methods.malmus.prompts import get_registry, MalmusSingleShotParams

class MalmusMethod(BaseMethod):
    async def _fit_single_shot(self, X, y):
        template = get_registry().get("single_shot").system
        params = MalmusSingleShotParams(
            columns=", ".join(X.columns),
            task="classification" if y.nunique() <= 10 else "regression",
            n_features=self.n_features,
        )
        prompt = params.render(template)
        raw_json = await self.llm_client.complete_json(
            messages=[{"role": "user", "content": prompt}],
            ...
        )
```

### Memory prompts (both system + user templates)

```python
# malmas/memory/prompts.py
import json
from pydantic import BaseModel


class SummarizeAgentParams(BaseModel):
    agent_name: str
    examples_text: str
    stats_text: str

    def render_system(self) -> str:
        return (
            f"You are {self.agent_name} agent, an expert feature engineering assistant. "
            "You will receive a list of effective features and statistics about their patterns. "
            "Your task is to generate effective, high-quality conceptual rules using concise language "
            "that can guide future feature generation."
        )

    def render_user(self) -> str:
        return (
            f"Here are the effective feature examples:\n\n{self.examples_text}\n\n"
            f"Here are the statistics about effective features:\n\n{self.stats_text}\n\n"
            "Based on both, summarize 1 to 3 concise and actionable conceptual rules "
            "to optimize future feature generation."
        )
```

Note: Memory prompts are **not** externalized to YAML — the text is embedded directly in the Params model as part of `.render_*()` methods. This is because memory prompts are tightly coupled to the summarization logic and unlikely to change independently. The YAML externalization is reserved for **call-site-agnostic prompt templates**.

## What changes

### New files

| File | Purpose |
|------|---------|
| `methods/malmus/prompts/__init__.py` | Prompt + PromptRegistry + MalmusSingleShotParams + MalmusIterativeParams |
| `methods/malmus/prompts/single_shot.yaml` | Extracted from `_build_prompt()` |
| `methods/malmus/prompts/iterative.yaml` | Extracted from `_build_iterative_prompt()` |
| `methods/llmfe/prompts/__init__.py` | Prompt + PromptRegistry + LLMFESingleShotParams + LLMFEIterativeParams |
| `methods/llmfe/prompts/single_shot.yaml` | Extracted from `_build_prompt()` |
| `methods/llmfe/prompts/iterative.yaml` | Extracted from `_build_iterative_prompt()` |
| `methods/caafe/prompts/__init__.py` | Prompt + PromptRegistry + CAAFEUnifiedParams |
| `methods/caafe/prompts/unified.yaml` | Extracted from `_build_caafe_prompt()` |
| `methods/malmas/memory/prompts.py` | SummarizeAgentParams + SummarizeGlobalParams |

### Modified files

| File | Change |
|------|--------|
| `methods/malmus/method.py` | Import from local prompts; use Params.render(); delete `_build_prompt()`, `_build_iterative_prompt()` |
| `methods/llmfe/method.py` | Same pattern — import from local prompts; delete `_build_prompt()`, `_build_iterative_prompt()` |
| `methods/caafe/method.py` | Same pattern — import from local prompts; delete `_build_caafe_prompt()`; keep `_build_dataset_description()` as static utility |
| `methods/malmas/memory/conceptual.py` | Import from `malmas.memory.prompts`; use Params models; delete hardcoded strings |
| `methods/malmas/agents/base.py` | Change import: `from feature_forge.prompts` → `from feature_forge.methods.malmas.prompts` |
| `methods/malmas/agents/router.py` | Same import change |
| `methods/malmas/pipeline/core.py` | Same import change |

### Moved files (content unchanged, path changes only)

| From | To |
|------|----|
| `config/prompts/unary.yaml` | `methods/malmas/prompts/unary.yaml` |
| `config/prompts/cross_compositional.yaml` | `methods/malmas/prompts/cross_compositional.yaml` |
| `config/prompts/aggregation.yaml` | `methods/malmas/prompts/aggregation.yaml` |
| `config/prompts/temporal.yaml` | `methods/malmas/prompts/temporal.yaml` |
| `config/prompts/local_transform.yaml` | `methods/malmas/prompts/local_transform.yaml` |
| `config/prompts/local_pattern.yaml` | `methods/malmas/prompts/local_pattern.yaml` |
| `config/prompts/router.yaml` | `methods/malmas/prompts/router.yaml` |
| `config/prompts/code_generation.yaml` | `methods/malmas/prompts/code_generation.yaml` |

### Deleted files

| File | Reason |
|------|--------|
| `config/prompts/` (entire directory) | Moved into method packages |
| `src/feature_forge/prompts/` | Shared PromptRegistry no longer needed — each method has its own |
| `src/feature_forge/methods/malmas/prompts/__init__.py` | Dead stub — replaced by the real malmas prompts init |
| `pyyaml` from `pyproject.toml` | Only used by deleted `prompts/` modules (if no other uses remain) |

### Updated docs

- `docs/methods.md` — replace `feature_forge.prompts` references
- `docs/api_reference.md` — same

## Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| YAML location | In method package (`methods/*/prompts/`) | Colocated with implementation; method is self-contained |
| Template syntax | `{placeholder}` with `str.format()` | Simple, standard, no new dependency. Pydantic validates the values, `.format()` just renders |
| Params vs Prompt | Separate models | Prompt validates YAML structure; Params validates runtime injection. Single responsibility |
| Memory prompts | Not in YAML, embedded in Params model | Tightly coupled to summarization logic; no benefit from externalizing |
| Duplicate PromptRegistry | Each method has its own copy | Self-contained packages; ~30 lines, no abstraction overhead |
| YAML `system` field naming | Reused for both system-role and user-role prompts | The field name represents "prompt content", not the message role. The role is determined by the calling code |
| `PromptRegistry` vs direct import | Registry with lazy loading + cache | Avoids re-parsing YAML on every LLM call; consistent with current pattern |

## Open questions

- [ ] What about agent-router `AGENT_CAPABILITIES` and `_build_selection_context()` — keep in Python as structured data, or externalize to YAML?
- [ ] What about `pipeline/core.py` error feedback template — is a single-line string worth a YAML file?

Current answer: Leave both as-is. They are formatting logic with fallthrough concatenation, not prompt templates.

## Success criteria

- [ ] `ruff check` passes
- [ ] `mypy` passes
- [ ] Each method loads its own prompts from its own package directory
- [ ] `config/prompts/` no longer exists
- [ ] `feature_forge.prompts` module no longer exists
- [ ] `pyyaml` can be removed from dependencies (no remaining imports)
- [ ] Invalid params (negative n_features, bad task value) raise Pydantic validation errors
- [ ] Empty prompt YAML raises validation error
