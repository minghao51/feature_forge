# Session Handoff Plan

**Slug:** feature-forge-unified-artifacts-implementation
**Readable Summary:** Unified Artifact Exposure Across MALMAS, LLMFE, CAAFE, OpenFE — Full Implementation Plan
**Date:** 2026-05-03
**Previous Handoff:** 2026-05-02-feature-forge-phases-4-13-complete

---

## 1. Primary Request and Intent

The user requests to **unify and expose intermediate artifacts** (generated scripts, intermediate DataFrames, feature metadata) across all four feature engineering methods in `feature_forge`:

- **MALMAS** (main method)
- **LLMFE** (LLM-based baseline)
- **CAAFE** (Context-Aware LLM baseline)
- **OpenFE** (non-LLM baseline)

**Specific intents:**
- Persist artifacts for experiments and method comparison
- Expose generated Python scripts and intermediate DataFrames
- Auto-log artifacts to WandB/MLflow (both Code artifacts and Table rows)
- Support configurable storage: memory, disk (parquet), or hybrid

**Key decisions confirmed by user:**
1. Reimplement **LLMFE** and **CAAFE** for full control over artifacts
2. Wrap-and-enhance **OpenFE** with best-effort extraction (fallback to `None` + warning)
3. Log code blocks to WandB as **both** Code artifacts and Table rows
4. CAAFE supports two variants: `unified` (our CVEvaluator/SandboxedExecutor) and `fidelity` (original internal logic)
5. Storage: parquet format, configurable via `ArtifactConfig`

---

## 2. Key Technical Concepts

- **ArtifactExporter ABC**: Mixin that all methods implement for unified artifact access
- **ArtifactConfig**: Dataclass controlling storage mode (`memory`/`disk`/`hybrid`), format (`parquet`/`csv`/`feather`), spill thresholds
- **LazyDataFrameRef**: Wrapper that loads DataFrames from disk on demand, making the API uniform regardless of storage backend
- **Per-round artifacts** (MALMAS): Each iteration generates code, full feature DataFrame, selected features, specs, agent gains, baseline score
- **Per-iteration artifacts** (LLMFE/CAAFE iterative): Each feature block generates code, features, CV gain, keep/discard decision
- **WandB dual logging**: Code blocks logged as (a) versioned Code artifacts and (b) Table rows with text columns

---

## 3. Architecture Design

### 3.1 Directory Structure (New + Modified)

```
src/feature_forge/
├── artifacts/
│   ├── __init__.py          # Public exports
│   ├── base.py              # ArtifactExporter ABC, ArtifactConfig
│   ├── storage.py           # LazyDataFrameRef, DataFrameStorage
│   └── comparison.py        # compare_methods()
├── baselines/
│   ├── base.py              # Modified: Baseline inherits ArtifactExporter
│   ├── llmfe.py             # REWRITE
│   ├── caafe.py             # REWRITE
│   └── openfe.py            # Modified: best-effort extraction
├── pipeline/
│   ├── core.py              # Modified: return all_features_train/test
│   └── iterative.py         # Modified: collect per-round artifacts
├── api.py                   # Modified: MALMASFeatureEngineer implements ArtifactExporter
├── experiment/
│   ├── tracker.py           # Modified: add log_artifacts_dict()
│   ├── wandb_backend.py     # Modified: _log_code(), _log_dataframe()
│   └── mlflow_backend.py    # Modified: artifact file logging
└── ...
```

### 3.2 New Files

#### `src/feature_forge/artifacts/__init__.py`
Exports: `ArtifactExporter`, `ArtifactConfig`, `LazyDataFrameRef`, `compare_methods`

#### `src/feature_forge/artifacts/base.py`
Contains `ArtifactConfig` dataclass and `ArtifactExporter` ABC with:
- `get_artifacts() -> dict[str, Any]`
- `generated_scripts -> list[str]` (abstract property)
- `intermediate_dataframes -> dict[str, pd.DataFrame | Any]` (property)
- `feature_metadata -> list[dict]` (property)
- `log_artifacts(tracker, prefix)` (method)

#### `src/feature_forge/artifacts/storage.py`
Contains:
- `LazyDataFrameRef(path, format)` — lazy-loading wrapper
- `DataFrameStorage(config)` — stores DataFrames based on ArtifactConfig, returns either df or LazyDataFrameRef

#### `src/feature_forge/artifacts/comparison.py`
Contains `compare_methods()` utility that:
- Runs all methods on same data
- Returns unified artifacts dict per method
- Auto-logs to tracker if provided

---

## 4. File-by-File Changes

### 4.1 `src/feature_forge/baselines/base.py`
- Import `ArtifactExporter`, `ArtifactConfig`, `DataFrameStorage`
- Make `Baseline` inherit from `ArtifactExporter`
- Add `artifact_config` parameter
- Add `self._storage` and `self._artifacts` attributes

### 4.2 `src/feature_forge/baselines/llmfe.py` — REWRITE
**Modes:** `single_shot` (default) and `iterative`

**single_shot flow:**
1. Build prompt from columns + task type
2. Call LLM
3. Strip markdown fences
4. Execute in sandbox
5. Store: prompt, raw_response, generated_code, generated_features_train/test, feature_metadata

**iterative flow:**
1. Evaluate baseline score
2. For i in range(n_features):
   a. Prompt LLM for ONE new feature given current cumulative state
   b. Execute code block in sandbox
   c. For each new column: evaluate via CVEvaluator, keep if gain > 0
   d. Append to cumulative features
   e. Store per-iteration: code, all_new_features, kept_features, gain
3. Store cumulative code + features + iterations list

**Artifacts:**
| Key | single_shot | iterative |
|-----|-------------|-----------|
| `prompt` | Yes | Yes (per-iteration) |
| `raw_response` | Yes | Yes (per-iteration) |
| `generated_code` | str | str (cumulative) |
| `generated_features_train` | DataFrame | DataFrame (cumulative) |
| `generated_features_test` | DataFrame | DataFrame (cumulative) |
| `iterations` | None | list[dict] |
| `feature_metadata` | list[dict] | list[dict] |

### 4.3 `src/feature_forge/baselines/caafe.py` — REWRITE
**Variants:** `unified` (default) and `fidelity`

**unified flow:**
- Reimplements CAAFE's iterative prompting using our infrastructure
- Build CAAFE-style prompt (dataset description, column samples, NaN freqs)
- Iterative loop: LLM call → code block → sandbox → CVEvaluator → keep/discard → update prompt with feedback
- Uses `CVEvaluator` and `SandboxedExecutor`

**fidelity flow:**
- Wraps original `caafe.CAAFEClassifier`
- Instantiates, calls `fit_pandas()` or `fit()`
- Extracts `.code` and `.mappings` after fit
- Executes code to get features

**Artifacts (both):**
| Key | Type |
|-----|------|
| `generated_code` | str |
| `generated_features_train` | DataFrame |
| `generated_features_test` | DataFrame |
| `iterations` | list[dict] (unified only) |
| `dataset_description` | str |
| `categorical_mappings` | dict (fidelity only) |
| `variant` | str |

### 4.4 `src/feature_forge/baselines/openfe.py` — ENHANCE
- Store reference to fitted `OpenFE` object (`self._ofe`)
- Implement `_extract_artifacts()`:
  - `selected_operators`: `self._features` (always available)
  - `candidate_operators`: `self._ofe.candidate_features_list` (best-effort, warn if missing)
  - `feature_importances`: `self._ofe.feature_importances_` (best-effort, warn if missing)
- `generated_scripts` returns `[]` (OpenFE doesn't generate code)

### 4.5 `src/feature_forge/pipeline/core.py`
- Return dict includes `all_features_train` and `all_features_test` (all generated, before top-k filter)
- Already returns `generated_code` from previous audit

### 4.6 `src/feature_forge/pipeline/iterative.py`
- Add `self.round_artifacts: list[dict] = []`
- After each round, append dict with:
  - `round`, `generated_code`, `all_features_train`, `all_features_test`
  - `selected_features_train`, `selected_features_test`
  - `specs`, `agent_gains`, `baseline_score`, `gains`, `agents`
- Return includes `round_artifacts`
- Add `get_artifacts()` method that flattens round artifacts with `round_{i}_` prefixes

### 4.7 `src/feature_forge/api.py`
- Make `MALMASFeatureEngineer` implement `ArtifactExporter`
- Implement `get_artifacts()`: flattens `pipeline_result["round_artifacts"]` with prefixes
- Implement `generated_scripts`: returns `feature_codes`
- Implement `feature_metadata`: aggregates all `specs` across rounds
- Implement `log_artifacts()`: delegates to ArtifactExporter base

### 4.8 `src/feature_forge/experiment/tracker.py`
- Add `log_artifacts_dict(artifacts, prefix)` — iterates dict, delegates to `_log_artifact_item`
- Add `_log_artifact_item(key, value)` — type dispatch
- Add abstract `_log_dataframe(key, df)`
- Add abstract `_log_code(key, code)`

### 4.9 `src/feature_forge/experiment/wandb_backend.py`
- `_log_dataframe`: Convert to `wandb.Table`, log
- `_log_code`:
  - If `log_code_to_wandb_artifact`: Save to temp .py file, create `wandb.Artifact(type="code")`, log
  - If `log_code_to_wandb_table`: Create Table with columns `["artifact_key", "code"]`, log

### 4.10 `src/feature_forge/experiment/mlflow_backend.py`
- `_log_dataframe`: Save to temp parquet, `mlflow.log_artifact()`
- `_log_code`: Save to temp .py file, `mlflow.log_artifact()`

---

## 5. Test Plan

### Unit Tests

**`tests/unit/test_artifact_storage.py`**
- Memory mode keeps DataFrames in memory
- Disk mode writes parquet and returns LazyDataFrameRef
- Hybrid mode spills large DataFrames
- LazyDataFrameRef loads correctly (parquet, csv, feather)
- CSV and feather roundtrips

**`tests/unit/test_artifact_exporter.py`**
- ArtifactConfig defaults
- generated_scripts returns list
- intermediate_dataframes filters correctly
- feature_metadata returns specs
- log_artifacts calls tracker methods

### Integration Tests

**`tests/integration/test_artifacts.py`**
- LLMFE single_shot: prompt, raw_response, generated_code, features exist
- LLMFE iterative: iterations list, per-iteration code, gains, kept flags
- CAAFE unified: iterative generation with CVEvaluator
- CAAFE fidelity: extraction from caafe library (skip if unavailable)
- OpenFE wrapper: selected_operators extracted; candidate_operators warns gracefully
- MALMAS per-round: round_1_generated_code, round_1_all_features, specs exist
- compare_methods: runs all methods, returns unified dicts
- compare_methods with tracker: verifies tracker methods called with prefixes
- ArtifactConfig disk mode: DataFrames written to disk, returned as LazyDataFrameRef
- WandB dual logging: both artifact and table logged

---

## 6. Documentation

**`docs/artifacts_guide.md`** (new)
- ArtifactExporter overview
- Per-method artifact reference table
- Storage mode comparison
- WandB logging examples
- compare_methods() usage

---

## 7. Known Risks

| Risk | Mitigation |
|------|-----------|
| OpenFE internal API changes | try/except extraction; fallback None + warning |
| CAAFE library unavailable | unified variant works without caafe installed |
| Large DataFrame memory | hybrid mode auto-spills; configurable threshold |
| WandB Table size limits | recommend disk mode for large datasets |
| LLMFE iterative expensive | default remains single_shot |
| CAAFE unified diverges from paper | document variant="fidelity" for published behavior |

---

## 8. Implementation Order

1. `artifacts/base.py` + `artifacts/storage.py`
2. Modify `baselines/base.py`
3. Rewrite `baselines/llmfe.py`
4. Rewrite `baselines/caafe.py`
5. Enhance `baselines/openfe.py`
6. Modify `pipeline/core.py` + `pipeline/iterative.py`
7. Modify `api.py`
8. Modify `experiment/tracker.py` + backends
9. Create `artifacts/comparison.py`
10. Write tests
11. Write docs
12. Run full test suite + ruff

---

## 9. Next Step for Execution Thread

1. Read this handoff document fully
2. Implement in order above
3. Run `uv run pytest tests/ -v` and `uv run ruff check src tests` after each major unit
4. Target: existing 127 tests pass + new artifact tests pass + zero ruff errors

---

## Context References

- **Project root**: `/Users/minghao/Desktop/personal/feature_forge`
- **Previous handoff**: `2026-05-02-feature-forge-phases-4-13-complete.md`
- **Current state**: 127 tests passing, 80% coverage, zero ruff errors
- **OpenFE source inspected**: `/tmp/pkg_inspect/openfe-0.0.12/`
- **CAAFE source inspected**: `/tmp/pkg_inspect/caafe-0.1.5/`

(End of handoff plan)
