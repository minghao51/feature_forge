# Methods Directory Restructure

**Date:** May 2026
**Status:** Completed
**Completed:** May 2026
**Depends on:** Current codebase state (no other in-flight changes required)
**Related:** `09_baseline_selection.md`, `10_experimental_platform_refactor.md`

> **Implementation note:** The restructure was executed as planned with minor deviations:
> - Prompts migrated to YAML (not `.txt`) with a `PromptRegistry` singleton in `feature_forge.prompts`.
> - CAAFE, LLM-FE, and Malmus kept prompts inline in `method.py` rather than extracting to separate `prompts.py` files.
> - `MALMASMethod` wraps `FeatureForge` as planned.
> - `AgentName` moved to `methods/malmas/types.py`.
> - Old directories (`agents/`, `baselines/`, `memory/`, `pipeline/`) fully removed — clean break, no re-exports.

## Motivation

The current `agents/` directory is exclusively MALMAS components, but its name implies a generic agent registry. Meanwhile, CAAFE, LLMFE, OpenFE, and Malmus live in `baselines/` — a name that undersells them as "first-class FE methods used for comparison." The MALMAS-specific components (agents, pipeline, memory, prompts) are scattered across 4 top-level directories with no namespace tying them together.

**Problems:**
- `agents/` doesn't signal "MALMAS-only" — confusing for contributors
- MALMAS internals spread across `agents/`, `pipeline/`, `memory/`, `prompts/`
- `baselines/` implies these methods are secondary; they're peers
- No unified protocol — MALMAS (`FeatureForge`) and baselines have separate APIs
- Adding new FE methods requires deciding "is it an agent or a baseline?"

**Solution:** Restructure into `methods/` with each FE method as a self-contained sub-package.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Top-level namespace | `methods/` | Clear grouping: "these are all FE methods" |
| MALMAS sub-package | `methods/malmas/` with nested agents/pipeline/memory/prompts | Collapses 4 scattered dirs into 1 namespace |
| Baseline extraction | Each baseline → `methods/{name}/` with extracted prompts/schemas | Mirrors malmas structure, easier to grow |
| Unified protocol | `MethodProtocol` — MALMAS implements it too | All methods interchangeable in `ExperimentalPlatform` |
| Backward compat | Clean break — no re-exports from old paths | Less cruft, update everything in one shot |
| Class renames | `Baseline*` → `Method*`, `*Baseline` → `*Method` | Terminology matches "methods" namespace |
| AgentName type | Moves into `methods/malmas/` | MALMAS-specific |
| FeatureSpec type | Stays at top-level `types.py` | Shared across methods |

## Target Structure

```
src/feature_forge/
├── methods/
│   ├── __init__.py                        # MethodRegistry, discover_all()
│   ├── base.py                            # MethodProtocol, BaseMethod, MethodRegistry
│   │
│   ├── malmas/
│   │   ├── __init__.py                    # Exports MALMASMethod, all agents, pipeline, memory
│   │   ├── method.py                      # MALMASMethod(MethodProtocol) — unified entry point
│   │   ├── agents/                        # ← moved from feature_forge/agents/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                    # Agent, BaseFeatureAgent, AgentRegistry
│   │   │   ├── router.py
│   │   │   ├── unary.py
│   │   │   ├── cross_compositional.py
│   │   │   ├── aggregation.py
│   │   │   ├── temporal.py
│   │   │   ├── local_transform.py
│   │   │   └── local_pattern.py
│   │   ├── pipeline/                      # ← moved from feature_forge/pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── core.py                    # CorePipeline, CodeGenerator
│   │   │   ├── iterative.py              # IterativePipeline
│   │   │   └── ablations.py              # NoMemoryPipeline, NoRouterPipeline, SingleAgentPipeline
│   │   ├── memory/                        # ← moved from feature_forge/memory/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                    # AgentMemory
│   │   │   ├── conceptual.py             # ConceptualMemory
│   │   │   └── persistence.py            # MemoryPersistence
│   │   └── prompts/                       # ← moved from feature_forge/prompts/
│   │       ├── unary.txt
│   │       ├── cross_compositional.txt
│   │       ├── aggregation.txt
│   │       ├── temporal.txt
│   │       ├── local_transform.txt
│   │       ├── local_pattern.txt
│   │       ├── router.txt
│   │       └── code_generation.txt
│   │
│   ├── caafe/                             # ← moved from baselines/caafe.py
│   │   ├── __init__.py
│   │   ├── method.py                      # CAAFEMethod (was CAAFEBaseline)
│   │   └── prompts.py                     # _build_dataset_description, _build_caafe_prompt
│   │
│   ├── llmfe/                             # ← moved from baselines/llmfe.py
│   │   ├── __init__.py
│   │   ├── method.py                      # LLMFEMethod (was LLMFEBaseline)
│   │   └── prompts.py                     # _build_prompt, _build_iterative_prompt
│   │
│   ├── openfe/                            # ← moved from baselines/openfe.py
│   │   ├── __init__.py
│   │   └── method.py                      # OpenFEMethod (was OpenFEBaseline)
│   │
│   └── malmus/                            # ← moved from baselines/malmus.py
│       ├── __init__.py
│       ├── method.py                      # MalmusMethod (was MalmusBaseline)
│       ├── prompts.py                     # _build_prompt, _build_iterative_prompt
│       └── schemas.py                     # FeatureDefinition, StructuredFeatureOutput, JSON schema constants
│
│   # Everything below stays at top level (shared infrastructure)
│
├── api.py                                 # FeatureForge — imports from methods.malmas.pipeline
├── platform.py                            # ExperimentalPlatform — imports from methods.base
├── config.py                              # Settings (unchanged)
├── types.py                               # FeatureSpec stays; AgentName moves out
├── utils.py                               # Shared helpers (unchanged)
├── exceptions.py                          # Shared exceptions (unchanged)
├── llm/                                   # Shared LLM layer (unchanged)
├── evaluation/                            # Shared evaluation (unchanged)
├── artifacts/                             # Shared artifacts (unchanged)
├── experiment/                            # Shared experiment harness (unchanged)
├── data/                                  # Shared data layer (unchanged)
└── observability/                         # Shared logging (unchanged)
```

## Key Renames

| Old | New | File |
|-----|-----|------|
| `BaselineProtocol` | `MethodProtocol` | `methods/base.py` |
| `Baseline` | `BaseMethod` | `methods/base.py` |
| `BaselineRegistry` | `MethodRegistry` | `methods/base.py` |
| `CAAFEBaseline` | `CAFEMethod` | `methods/caafe/method.py` |
| `LLMFEBaseline` | `LLMFEMethod` | `methods/llmfe/method.py` |
| `OpenFEBaseline` | `OpenFEMethod` | `methods/openfe/method.py` |
| `MalmusBaseline` | `MalmusMethod` | `methods/malmus/method.py` |
| Entry point group: `feature_forge.baselines` | `feature_forge.methods` | `pyproject.toml` |
| Entry point group: `feature_forge.agents` | `feature_forge.methods.malmas.agents` | `pyproject.toml` |

## New: `MALMASMethod`

`methods/malmas/method.py` adapts the MALMAS pipeline to implement `MethodProtocol`, making MALMAS interchangeable with other methods in `ExperimentalPlatform`.

```python
class MALMASMethod(BaseMethod):
    """MALMAS method adapter implementing MethodProtocol."""

    def __init__(self, settings: Settings | None = None, **kwargs):
        self._settings = settings or get_settings()
        self._kwargs = kwargs
        self._forge: FeatureForge | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> MALMASMethod:
        self._forge = FeatureForge(settings=self._settings, **self._kwargs)
        self._forge.fit(X_train, y_train, **kwargs)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._forge.transform(X)

    def fit_transform(self, X_train, y_train, **kwargs) -> pd.DataFrame:
        self.fit(X_train, y_train, **kwargs)
        return self.transform(X_train)

    @property
    def generated_scripts(self) -> list[str]:
        return self._forge.generated_scripts if self._forge else []

    @property
    def feature_metadata(self) -> list:
        return self._forge.feature_metadata if self._forge else []

    def get_artifacts(self): ...
```

Key: `FeatureForge` remains the public sklearn-compatible API in `api.py`. `MALMASMethod` wraps it for the unified method protocol.

## New: `methods/base.py`

Derived from current `baselines/base.py` with renames:

```python
class MethodProtocol(Protocol):
    def fit(self, X_train, y_train, **kwargs): ...
    def transform(self, X): ...
    def fit_transform(self, X_train, y_train, **kwargs): ...
    @property
    def generated_scripts(self) -> list[str]: ...
    @property
    def feature_metadata(self) -> list: ...
    def get_artifacts(self): ...

class BaseMethod(ArtifactExporter):
    """Base class for all feature engineering methods."""
    def fit(self, X_train, y_train, **kwargs): ...
    def transform(self, X): ...
    def fit_transform(self, X_train, y_train, **kwargs): ...
    # ... (same as current Baseline, just renamed)

class MethodRegistry:
    ENTRY_POINT_GROUP = "feature_forge.methods"
    # discover(), get_builtin_methods(), get_all_methods()
    # Built-in fallbacks: malmas, caafe, llmfe, openfe, malmus
```

## Cross-Reference Map

All 134 references across 37 files need updating. No relative imports exist — all use absolute paths.

### Import path changes

| Old path | New path |
|----------|----------|
| `feature_forge.agents` | `feature_forge.methods.malmas.agents` |
| `feature_forge.agents.base` | `feature_forge.methods.malmas.agents.base` |
| `feature_forge.agents.router` | `feature_forge.methods.malmas.agents.router` |
| `feature_forge.agents.unary` | `feature_forge.methods.malmas.agents.unary` |
| `feature_forge.agents.{agent}` | `feature_forge.methods.malmas.agents.{agent}` |
| `feature_forge.baselines` | `feature_forge.methods` |
| `feature_forge.baselines.base` | `feature_forge.methods.base` |
| `feature_forge.baselines.caafe` | `feature_forge.methods.caafe.method` |
| `feature_forge.baselines.llmfe` | `feature_forge.methods.llmfe.method` |
| `feature_forge.baselines.openfe` | `feature_forge.methods.openfe.method` |
| `feature_forge.baselines.malmus` | `feature_forge.methods.malmus.method` |
| `feature_forge.pipeline` | `feature_forge.methods.malmas.pipeline` |
| `feature_forge.pipeline.core` | `feature_forge.methods.malmas.pipeline.core` |
| `feature_forge.pipeline.iterative` | `feature_forge.methods.malmas.pipeline.iterative` |
| `feature_forge.pipeline.ablations` | `feature_forge.methods.malmas.pipeline.ablations` |
| `feature_forge.memory` | `feature_forge.methods.malmas.memory` |
| `feature_forge.memory.base` | `feature_forge.methods.malmas.memory.base` |
| `feature_forge.memory.conceptual` | `feature_forge.methods.malmas.memory.conceptual` |
| `feature_forge.memory.persistence` | `feature_forge.methods.malmas.memory.persistence` |

### Source files to update (within `src/`)

| File | Changes |
|------|---------|
| `agents/__init__.py` | Update all lazy imports to `methods.malmas.agents.*` |
| `agents/base.py` | Update entry point group string, fallback paths |
| `agents/unary.py` | Update import of `BaseFeatureAgent` |
| `agents/cross_compositional.py` | Update import of `BaseFeatureAgent` |
| `agents/aggregation.py` | Update import of `BaseFeatureAgent` |
| `agents/temporal.py` | Update import of `BaseFeatureAgent` |
| `agents/local_transform.py` | Update import of `BaseFeatureAgent` |
| `agents/local_pattern.py` | Update import of `BaseFeatureAgent` |
| `agents/router.py` | Update any internal imports |
| `baselines/__init__.py` | Replace with `methods/__init__.py` |
| `baselines/base.py` | Rename classes, update entry point group |
| `baselines/caafe.py` | Split into `methods/caafe/method.py` + `prompts.py` |
| `baselines/llmfe.py` | Split into `methods/llmfe/method.py` + `prompts.py` |
| `baselines/openfe.py` | Move to `methods/openfe/method.py` |
| `baselines/malmus.py` | Split into `methods/malmus/method.py` + `prompts.py` + `schemas.py` |
| `pipeline/core.py` | Update `from feature_forge.agents.base` import |
| `pipeline/iterative.py` | Update agents/memory imports |
| `pipeline/ablations.py` | Update agents imports |
| `api.py` | Update lazy pipeline imports to `methods.malmas.pipeline.*` |
| `platform.py` | Update `from feature_forge.baselines` → `from feature_forge.methods` |
| `types.py` | Remove `AgentName` (moves to `methods/malmas/`) |

### Test files to update (within `tests/`)

| File | Changes |
|------|---------|
| `tests/unit/test_agents.py` | Update all `feature_forge.agents` imports |
| `tests/unit/test_contract.py` | Update agents + baselines imports |
| `tests/unit/test_metamorphic.py` | Update `feature_forge.agents.router` import |
| `tests/unit/test_malmus.py` | Update `feature_forge.baselines.*` imports |
| `tests/unit/test_baseline_protocol.py` | Update all baseline imports + class names |
| `tests/unit/test_artifact_exporter.py` | Update `feature_forge.baselines.llmfe` imports |
| `tests/unit/test_platform.py` | Update `feature_forge.baselines` import |
| `tests/integration/test_pipeline.py` | Update `feature_forge.agents.base` import |
| `tests/integration/test_baselines.py` | Update all baseline imports |
| `tests/integration/test_artifacts.py` | Update all baseline imports |
| `tests/integration/test_platform_e2e.py` | Update `feature_forge.baselines` import |
| `tests/integration/test_plugin_discovery.py` | Update `feature_forge.baselines.base` import |

### Config & docs to update

| File | Changes |
|------|---------|
| `pyproject.toml` | Entry point groups: `feature_forge.agents` → `feature_forge.methods.malmas.agents`, `feature_forge.baselines` → `feature_forge.methods` |
| `README.md` | Import path examples |
| `docs/index.md` | Import path examples |
| `docs/quick_start.md` | Import path examples |
| `docs/methods.md` | Method references |
| `docs/migration_guide.md` | Import path references |
| `docs/api_reference.md` | Class name references |
| `PLUGINS.md` | Entry point references |
| `.planning/STYLE.md` | Import path references |
| `.planning/codebase/ARCHITECTURE.md` | Architecture references |
| `docs/plan/01_architecture.md` | Directory references |
| `docs/plan/03_key_design_decisions.md` | Agent/baseline references |
| `docs/plan/05_dependencies.md` | Module references |
| `docs/plan/10_experimental_platform_refactor.md` | Baseline references |
| `notebooks/02_pipeline_deep_dive.ipynb` | Import paths |
| `notebooks/03_benchmarks_and_artifacts.ipynb` | Import paths |

## Execution Phases

### Phase 1: Scaffold (new directories only, no logic changes)

Create all directories and `__init__.py` files:

```
src/feature_forge/methods/
src/feature_forge/methods/malmas/
src/feature_forge/methods/malmas/agents/
src/feature_forge/methods/malmas/pipeline/
src/feature_forge/methods/malmas/memory/
src/feature_forge/methods/malmas/prompts/
src/feature_forge/methods/caafe/
src/feature_forge/methods/llmfe/
src/feature_forge/methods/openfe/
src/feature_forge/methods/malmus/
```

### Phase 2: Create `methods/base.py`

- Copy `baselines/base.py`
- Rename `BaselineProtocol` → `MethodProtocol`, `Baseline` → `BaseMethod`, `BaselineRegistry` → `MethodRegistry`
- Update `ENTRY_POINT_GROUP` to `"feature_forge.methods"`
- Update fallback built-in paths to `methods.{name}.method`

### Phase 3: Move MALMAS components

For each sub-package (agents, pipeline, memory, prompts):
1. Copy files to `methods/malmas/{subdir}/`
2. Update all internal imports to new paths
3. Update entry point group in `agents/base.py`

**Prompt loading in `BaseFeatureAgent`:** Currently loads from `prompts/<name>.txt` relative to package. Must update to load from `methods/malmas/prompts/<name>.txt`.

### Phase 4: Create `MALMASMethod`

New file `methods/malmas/method.py` implementing `MethodProtocol` by wrapping `FeatureForge` (see code sketch above).

### Phase 5: Restructure each baseline

For each method (caafe, llmfe, openfe, malmus):
1. Create `methods/{name}/method.py` — main class, renamed from `*Baseline` to `*Method`
2. Update `BaseMethod` import: `from feature_forge.methods.base import BaseMethod`
3. Extract inline prompts to `prompts.py` (caafe, llmfe, malmus)
4. Extract schemas/models to `schemas.py` (malmus only: `FeatureDefinition`, `StructuredFeatureOutput`, JSON schema constants)
5. Create `__init__.py` with re-exports

### Phase 6: Create `methods/__init__.py`

Registry + re-exports:
```python
from feature_forge.methods.base import MethodProtocol, BaseMethod, MethodRegistry
from feature_forge.methods.malmas.method import MALMASMethod
from feature_forge.methods.caafe.method import CAAFEMethod
from feature_forge.methods.llmfe.method import LLMFEMethod
from feature_forge.methods.openfe.method import OpenFEMethod
from feature_forge.methods.malmus.method import MalmusMethod
```

### Phase 7: Update consumers

- `api.py` — lazy imports: `pipeline.iterative` → `methods.malmas.pipeline.iterative`, etc.
- `platform.py` — `from feature_forge.baselines` → `from feature_forge.methods`
- `types.py` — remove `AgentName`, move to `methods/malmas/agents/base.py` or `methods/malmas/types.py`

### Phase 8: Update `pyproject.toml`

```toml
[project.entry-points."feature_forge.methods.malmas.agents"]
unary = "feature_forge.methods.malmas.agents.unary:UnaryFeatureAgent"
cross_compositional = "feature_forge.methods.malmas.agents.cross_compositional:CrossCompositionalAgent"
aggregation = "feature_forge.methods.malmas.agents.aggregation:AggregationConstructAgent"
temporal = "feature_forge.methods.malmas.agents.temporal:TemporalFeatureAgent"
local_transform = "feature_forge.methods.malmas.agents.local_transform:LocalTransformAgent"
local_pattern = "feature_forge.methods.malmas.agents.local_pattern:LocalPatternAgent"

[project.entry-points."feature_forge.methods"]
malmas = "feature_forge.methods.malmas.method:MALMASMethod"
caafe = "feature_forge.methods.caafe.method:CAAFEMethod"
llmfe = "feature_forge.methods.llmfe.method:LLMFEMethod"
openfe = "feature_forge.methods.openfe.method:OpenFEMethod"
malmus = "feature_forge.methods.malmus.method:MalmusMethod"
```

### Phase 9: Update tests

For each test file, update imports:
- `from feature_forge.agents.*` → `from feature_forge.methods.malmas.agents.*`
- `from feature_forge.baselines.*` → `from feature_forge.methods.*`
- Class names: `Baseline` → `BaseMethod`, `CAAFEBaseline` → `CAFEMethod`, etc.

### Phase 10: Update docs & notebooks

Search-replace across all docs and notebooks (see full file list in cross-reference map above).

### Phase 11: Delete old directories

```
rm -rf src/feature_forge/agents/
rm -rf src/feature_forge/baselines/
rm -rf src/feature_forge/pipeline/
rm -rf src/feature_forge/memory/
rm -rf src/feature_forge/prompts/
```

### Phase 12: Verify

```bash
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest tests/ -x
```

## Risks & Considerations

| Risk | Mitigation |
|------|------------|
| **Prompt loading path** breaks when moving `prompts/` | `BaseFeatureAgent` uses `importlib.resources` or `__file__`-relative paths — must update to resolve from `methods/malmas/prompts/` |
| **Entry point breakage** for external plugins | Clean break as decided. Document migration in changelog. |
| **`MALMASFeatureEngineer` alias** in `api.py` | Keep the alias, it just re-exports `FeatureForge` |
| **Circular imports** between `methods/base.py` and `artifacts/` | `BaseMethod` inherits `ArtifactExporter` from `artifacts.base` — same direction as current, no circular risk |
| **Test fixtures** referencing old paths | Update all in Phase 9 |
| **Large diff** — 37+ files changed | Work phase-by-phase, run tests after each phase |
| **`AgentName` type** used in `types.py` | Move to `methods/malmas/` — it's MALMAS-specific (agent names like "unary", "temporal", etc.) |

## Validation Checklist

After completion, verify:

- [ ] `uv run ruff check src/ tests/` passes
- [ ] `uv run mypy src/` passes
- [ ] `uv run pytest tests/ -x` passes
- [ ] `from feature_forge.methods import MethodRegistry` works
- [ ] `from feature_forge.methods.malmas import MALMASMethod` works
- [ ] `from feature_forge.methods.caafe import CAAFEMethod` works
- [ ] `from feature_forge import FeatureForge` still works (public API)
- [ ] `ExperimentalPlatform` can discover all methods via entry points
- [ ] No references to `feature_forge.agents` or `feature_forge.baselines` remain
- [ ] No references to old class names (`Baseline`, `*Baseline`) remain in source
