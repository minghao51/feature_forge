# Experimental Platform Refactor: Option C (Hybrid)

**Date:** May 2026
**Status:** Proposed
**Based on:** Google AI Search best practices (May 2026) + codebase audit

---

## Current State

`src/feature_forge/` already has strong foundations:

| Component | Pattern | Status |
|-----------|---------|--------|
| Plugin discovery | `importlib.metadata.entry_points` (`BaselineRegistry`, `AgentRegistry`) | Done |
| Baseline API | `fit()` / `transform()` (sklearn-compatible) | Done |
| Config | `pydantic-settings` with YAML + env | Done |
| Experiment runner | `ExperimentMatrix` + `ExperimentRunner` + `ExperimentTracker` | Done |
| 4 built-in baselines | CAAFE, Malmus, OpenFE, LLMFE | Done |
| 6 built-in agents | Unary, CrossComp, Aggregation, Temporal, LocalTransform, LocalPattern | Done |
| Sandboxed execution | AST validation + subprocess isolation | Done |
| Observability | structlog + Langfuse + WandB/MLflow | Done |

**Gap**: Components are loosely connected — users manually wire `ExperimentMatrix` + `ExperimentRunner` + `BaselineRegistry` + `CVEvaluator`. No unified "platform" facade. Plugin hooks exist for agents and baselines, but not for datasets, metrics, or models.

---

## Google AI Search Best Practices Summary

From two independent searches (May 2026):

| Practice | Recommendation | Status |
|----------|---------------|--------|
| Plugin discovery | `importlib.entry_points` for open ecosystems | Already used |
| Classpath registry | For monolithic/internal codebases | Available as fallback |
| sklearn API | `BaseEstimator` / `TransformerMixin` for ML tools | Already done |
| Process isolation | Subprocess/container per benchmark | Already done (`SandboxedExecutor`) |
| Structured config | Pydantic at boundaries | Already done |
| Artifact isolation | Centralized storage, not local disk | Hybrid mode supported |
| Hybrid approach | Built-in + entry points → best of both worlds | **Target** |

---

## Three Architectural Options (Evaluated)

### Option A: "Open Marketplace" — Entry Points + Namespace Packages

Each baseline is a separate pip-installable package. Core never imports plugins directly.

| Pros | Cons |
|------|------|
| True decoupling — core never imports plugins | Higher maintenance (N repos/CI pipelines) |
| 3rd parties contribute without touching core | Slower iteration when developing core+plugins together |
| Explicit dependency safety per plugin | Version sync overhead (core API changes break plugins) |
| Industry standard: `pytest`, `pre-commit`, `airflow` all use this | Entry point debugging is painful |

### Option B: "Marketplace-in-a-Box" — Single Repo, Lazily Loaded Registry

Keep everything in one repo with auto-discovery from a `plugins/` directory or decorator-based registration.

| Pros | Cons |
|------|------|
| Single repo = low overhead, easy to develop | Harder to permission-gate 3rd party contributions |
| Fast startup — no entry point scanning overhead | Monolithic dependency footprint |
| Drop-in simplicity for researchers (just add a .py file) | Monorepo scaling issues as baselines grow |
| AutoGluon, Featuretools use this pattern | Coupling risk between plugins |

### Option C: "Hybrid" — Built-in + Entry Points + Namespace Packages (SELECTED)

Keep built-in baselines in core for zero-config out-of-box experience. Support entry point discovery for 3rd party plugins. Expose a unified `ExperimentalPlatform` API.

| Pros | Cons |
|------|------|
| Out-of-box batteries included | Slightly more complex than pure A or B |
| Extensible for 3rd parties | Must maintain both built-in and entry point paths |
| Best of both worlds — Quickstart + Ecosystem | Documentation must cover both plugin paths |
| Backward compatible (already partially implemented) | Need clear versioning/API stability contract |

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ExperimentalPlatform  (NEW: src/feature_forge/platform.py)    │
│  One-liner API: platform.run(datasets, baselines, models)       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐         │
│  │ Experiment   │  │ Baseline     │  │ Dataset       │         │
│  │ Engine       │  │ Registry     │  │ Registry      │         │
│  │ (Runner+CV)  │  │ (builtin +   │  │ (builtin +    │         │
│  │              │  │  entry_pt)   │  │  entry_pt)    │         │
│  └──────────────┘  └──────────────┘  └───────────────┘         │
├─────────────────────────────────────────────────────────────────┤
│  Plugin Hooks (entry_points)                                     │
│  feature_forge.baselines    ← 3rd party (exists)                 │
│  feature_forge.agents       ← 3rd party (exists)                 │
│  feature_forge.datasets     ← 3rd party (NEW)                    │
│  feature_forge.models       ← 3rd party (NEW)                    │
│  feature_forge.metrics      ← 3rd party (NEW)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: `ExperimentalPlatform` — Unified Facade

**New file: `src/feature_forge/platform.py`**

Wraps existing components into a single class. No new logic — pure wiring of `ExperimentMatrix`, `ExperimentRunner`, `BaselineRegistry`, `CVEvaluator`, `Reporter`, and `DatasetRegistry`.

```python
from feature_forge import ExperimentalPlatform

platform = ExperimentalPlatform()

# One-liner: dataset × baseline × model comparison
results = platform.run(
    datasets=["titanic", "house_prices"],
    baselines=["malmus", "caafe", "openfe", "llmfe", "my_custom_baseline"],
    models=["xgboost", "lightgbm"],
    mode="single_shot",
    cv_folds=5,
    seeds=[42, 123],
)

# Reporting
platform.report(results)          # markdown comparison table
platform.report_best(results)     # best per dataset
platform.to_dataframe(results)    # raw pandas DataFrame

# Introspection
platform.list_baselines()         # built-in + discovered
platform.list_datasets()          # registered + discovered
platform.list_models()            # built-in + discovered
platform.list_metrics()           # built-in + discovered
```

**API methods:**

| Method | Purpose |
|--------|---------|
| `run(self, datasets, baselines, models, mode, cv_folds, seeds, **kwargs)` | Execute experiment matrix; returns list of result dicts |
| `register_baseline(name, cls)` | Programmatic ad-hoc baseline registration |
| `register_dataset(name, info)` | Programmatic ad-hoc dataset registration |
| `list_baselines()` | All baselines (built-in + discovered + ad-hoc) |
| `list_datasets()` | All datasets (built-in + discovered + ad-hoc) |
| `list_models()` | All models (built-in + discovered) |
| `list_metrics()` | All metrics (built-in + discovered) |
| `report(results)` | Markdown comparison table via `Reporter` |
| `report_best(results, metric, group_by)` | Best per group |
| `to_dataframe(results)` | Raw pandas DataFrame |

**Changes to `src/feature_forge/__init__.py`**:
- Export `ExperimentalPlatform` as the primary public entry point
- Keep existing exports for backward compatibility

**Files changed:** `src/feature_forge/platform.py` (create), `src/feature_forge/__init__.py` (edit)

---

### Phase 2: Plugin Hook Extensions

#### 2a. MetricRegistry → Entry Points (NEW group)

Current: `evaluation/metrics.py` uses a hardcoded `METRIC_REGISTRY` dict.

**Change**: Add `MetricRegistry` class with entry point discovery + built-in fallback.

Add to `pyproject.toml`:
```toml
[project.entry-points."feature_forge.metrics"]
auc = "feature_forge.evaluation.metrics:auc_score"
acc = "feature_forge.evaluation.metrics:acc_score"
f1 = "feature_forge.evaluation.metrics:f1_score_metric"
rmse = "feature_forge.evaluation.metrics:rmse_score"
mae = "feature_forge.evaluation.metrics:mae_score"
r2 = "feature_forge.evaluation.metrics:r2_score_metric"
nrmse = "feature_forge.evaluation.metrics:nrmse_score"
```

3rd party example:
```toml
# feature-forge-custom-metrics pyproject.toml
[project.entry-points."feature_forge.metrics"]
log_loss = "custom_metrics:log_loss_score"
```

**Files changed:** `src/feature_forge/evaluation/metrics.py` (edit), `pyproject.toml` (edit)

#### 2b. ModelRegistry → Entry Points (NEW group)

Current: `evaluation/model_factory.py` hardcodes model constructors.

**Change**: Add `ModelRegistry` class with entry point discovery.

Add to `pyproject.toml`:
```toml
[project.entry-points."feature_forge.models"]
xgboost = "feature_forge.evaluation.model_factory:create_xgboost"
# Optional extras for lightgbm, catboost
```

3rd party example — a pip-installable model plugin:
```toml
# feature-forge-catboost pyproject.toml
[project.entry-points."feature_forge.models"]
catboost = "feature_forge_catboost:create_catboost"
```

Then users install: `pip install feature-forge[catboost]`

**Files changed:** `src/feature_forge/evaluation/model_factory.py` (edit), `pyproject.toml` (edit)

#### 2c. DatasetRegistry → Entry Points (EXTEND existing)

Current: `DatasetRegistry` hardcodes Kaggle slugs + scans `data/samples/` for JSON.

**Change**: Add entry point discovery alongside existing sources. `DatasetRegistry.discover()` returns union of:
1. Built-in hardcoded datasets (Titanic, House Prices)
2. Entry-point registered datasets
3. Local sample datasets from `data/samples/`

Add to `pyproject.toml`:
```toml
[project.entry-points."feature_forge.datasets"]
titanic = "feature_forge.data.registry:titanic_loader"
house_prices = "feature_forge.data.registry:house_prices_loader"
```

**Files changed:** `src/feature_forge/data/registry.py` (edit), `pyproject.toml` (edit)

#### 2d. BaselineProtocol — Stabilize Contract (EXISTS, enhance)

Current: `Baseline` ABC in `baselines/base.py` serves as the contract.

**Addition**: Define `BaselineProtocol` as a `@runtime_checkable` protocol for 3rd-party developers who don't want to import from `feature_forge`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class BaselineProtocol(Protocol):
    """Contract for any baseline method — no dependency on feature_forge internals."""
    name: str

    def fit(self, X_train, y_train) -> Self: ...
    def transform(self, X) -> pd.DataFrame: ...
    def fit_transform(self, X_train, y_train) -> pd.DataFrame: ...

    @property
    def generated_scripts(self) -> list[str]: ...
    @property
    def feature_metadata(self) -> list[dict[str, Any]]: ...
    def get_artifacts(self) -> dict[str, Any]: ...
```

This allows a researcher to write a baseline in a standalone `.py` file and register it via entry point without importing anything from `feature_forge`.

**Files changed:** `src/feature_forge/baselines/base.py` (edit)

---

### Phase 3: Package Formalization

| File | Change |
|------|--------|
| `src/feature_forge/py.typed` | **Create** — PEP 561 marker |
| `pyproject.toml` | Split `[project.optional-dependencies]` per baseline/model |
| `PLUGINS.md` | **Create** — developer guide for custom baselines/datasets/metrics/models |
| `README.md` | Edit — add `ExperimentalPlatform` quickstart |
| `docs/plan/00_index.md` | Edit — add this document reference |

**Optional dependencies restructure:**

```toml
[project.optional-dependencies]
caafe = ["caafe>=1.0"]
openfe = ["openfe>=0.1"]
kaggle = ["kaggle"]
lightgbm = ["lightgbm>=4.0"]
catboost = ["catboost>=1.2"]
wandb = ["wandb>=0.16"]
mlflow = ["mlflow>=2.10"]
litellm = ["litellm>=1.40"]
all = ["feature-forge[caafe,openfe,kaggle,lightgbm,wandb,mlflow,litellm]"]
```

Usage: `pip install feature-forge[caafe,openfe,lightgbm]`

---

### Phase 4: `export` — Publish Baseline as Standalone Package

**New: `src/feature_forge/baselines/cli.py`** (or `scripts/export_baseline.py`)

Generates a skeleton pip-installable package from a custom baseline. This is the bridge from "research code" to "benchmarkable artifact."

```bash
uv run feature-forge export-baseline my_custom_baseline.py --name feature-forge-mymethod
```

Generated output:
```
feature-forge-mymethod/
├── pyproject.toml         # entry point → feature_forge.baselines
├── src/
│   └── mymethod/
│       ├── __init__.py
│       └── baseline.py    # user's baseline, refactored
├── README.md
└── tests/
    └── test_baseline.py
```

After generation, the baseline becomes installable and discoverable:
```bash
pip install feature-forge-mymethod
# Now platform.list_baselines() includes "mymethod"
```

---

## Summary of All Changes

| File | Action | Phase |
|------|--------|-------|
| `src/feature_forge/platform.py` | **Create** | 1 |
| `src/feature_forge/__init__.py` | Edit (exports) | 1 |
| `src/feature_forge/evaluation/metrics.py` | Edit (+MetricRegistry) | 2a |
| `src/feature_forge/evaluation/model_factory.py` | Edit (+entry point discovery) | 2b |
| `src/feature_forge/data/registry.py` | Edit (+entry point discovery) | 2c |
| `src/feature_forge/baselines/base.py` | Edit (+BaselineProtocol) | 2d |
| `pyproject.toml` | Edit (new entry point groups, optional deps) | 2, 3 |
| `src/feature_forge/py.typed` | **Create** | 3 |
| `PLUGINS.md` | **Create** | 3 |
| `README.md` | Edit (quickstart examples) | 3 |
| `docs/plan/00_index.md` | Edit (add document reference) | 3 |
| `src/feature_forge/baselines/cli.py` | **Create** | 4 |

---

## What Does NOT Change

- `api.py` `FeatureForge` class — remains as the sklearn-compatible single-baseline API
- Pipeline orchestration (`pipeline/`) — unchanged
- Agent system (`agents/`) — already has entry points, unchanged
- Sandbox executor (`evaluation/sandbox.py`) — unchanged
- CV evaluator (`evaluation/cv.py`) — unchanged
- LLM layer (`llm/`) — unchanged
- Observability (`observability/`) — unchanged
- Artifact system (`artifacts/`) — unchanged
- Memory system (`memory/`) — unchanged

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Entry point debugging pain | Dev frustration | `platform.list_baselines()` introspection + clear error messages on failure |
| 3rd party breakage on API change | Ecosystem fragmentation | `BaselineProtocol` makes contract explicit; semantic versioning |
| Startup overhead from entry point scanning | 2-3s delay | Lazy-load: `discover()` called once, cached in registry `_cache` |
| Optional dep install confusion | User support burden | `try/except ImportError` with helpful `pip install feature-forge[caafe]` message |
| Entry point name collisions | Undefined behavior | Last-registered-wins policy; warn on duplicate names |

---

## Before/After: User Experience

### Before (current)
```python
from feature_forge.baselines import BaselineRegistry, CAAFEBaseline, MalmusBaseline
from feature_forge.experiment import ExperimentMatrix, ExperimentRunner
from feature_forge.evaluation import CVEvaluator
from feature_forge.data import DatasetRegistry
from feature_forge.config import get_settings

# ~15 lines of manual wiring
settings = get_settings(task="classification", metric="auc")
baselines = BaselineRegistry.get_all_baselines()
datasets = DatasetRegistry()
matrix = ExperimentMatrix().datasets(["titanic"]).methods({
    "malmus": ["single_shot"],
    "caafe": ["unified"],
}).seeds([42]).generate()
# ... manually iterate, call fit/transform, evaluate, collect results
```

### After (target)
```python
from feature_forge import ExperimentalPlatform

platform = ExperimentalPlatform()

results = platform.run(
    datasets=["titanic", "house_prices"],
    baselines=["malmus", "caafe", "openfe", "llmfe"],
    models=["xgboost"],
    mode="single_shot",
    cv_folds=5,
)

platform.report(results)
# ┌──────────┬──────────┬──────────┬──────────┬──────────┐
# │ dataset  │ baseline │ model    │ cv_score │ gain     │
# ├──────────┼──────────┼──────────┼──────────┼──────────┤
# │ titanic  │ malmus   │ xgboost  │ 0.8523   │ +0.0312  │
# │ titanic  │ caafe    │ xgboost  │ 0.8411   │ +0.0200  │
# │ ...      │ ...      │ ...      │ ...      │ ...      │
# └──────────┴──────────┴──────────┴──────────┴──────────┘

# 3rd party baseline — zero config
# pip install feature-forge-newmethod
results = platform.run(
    datasets=["titanic"],
    baselines=["newmethod", "malmus"],  # auto-discovered
    models=["xgboost"],
)
```

---

## Verification

### Unit Tests (new)
- `tests/unit/test_platform.py` — platform initialization, registration, listing, run with mock components
- `tests/unit/test_metric_registry.py` — discovery from entry points
- `tests/unit/test_model_registry.py` — discovery from entry points
- `tests/unit/test_dataset_registry_discovery.py` — entry point discovery
- `tests/unit/test_baseline_protocol.py` — protocol compliance checks

### Integration Tests
- `tests/integration/test_platform_e2e.py` — end-to-end mini run with sample data + 2 baselines + 1 model
- `tests/integration/test_plugin_discovery.py` — verify entry-point-discovered baselines appear in `list_baselines()`

### Existing Tests
- All existing tests must continue to pass (no breaking changes)
- Run: `uv run pytest` → expected 100% pass rate

---

## Timeline Estimate

| Phase | Work | Duration |
|-------|------|----------|
| 1 | `ExperimentalPlatform` class | 1 day |
| 2a | MetricRegistry entry points | 0.5 day |
| 2b | ModelRegistry entry points | 0.5 day |
| 2c | DatasetRegistry entry points | 0.5 day |
| 2d | BaselineProtocol | 0.5 day |
| 3 | Package formalization (py.typed, deps, docs) | 1 day |
| 4 | Export CLI skeleton | 1 day |
| — | Tests for all of above | 2 days |
| **Total** | | **~7 days** |
