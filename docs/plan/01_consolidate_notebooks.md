# Notebook Consolidation Plan

**Date:** 2026-05-13
**Status:** ✅ Completed

---

## 1. Current State

### 10 notebooks, each triple-authored (.ipynb + .qmd + .py)

| # | File | Title | Cells | Error-tolerant cells | Topic |
|---|------|-------|-------|---------------------|-------|
| 01 | `01_quick_start` | Quick Start: Sklearn API | 22 | 9 | End-to-end fit/transform/evaluate |
| 02 | `02_agents` | 6 Specialized Feature Generators | 17 | 2 | Agent registry + per-agent demo |
| 03 | `03_router` | Router & Pipeline Modes | 17 | 5 | Router strategies + ablation |
| 04 | `04_iterative_pipeline` | Iterative Pipeline & Memory | 19 | 6 | Multi-round pipeline + memory |
| 05 | `05_experiment_matrix` | Experiment Matrix & Tracking | 21 | 4 | Cartesian experiments + WandB/MLflow |
| 06 | `06_baselines` | Baselines Comparison | 23 | 8 | OpenFE/CAAFE/LLM-FE/Malmus |
| 07 | `07_evaluation` | Evaluation & Sandboxed Execution | 21 | 1 | CVEval, sandbox, metrics |
| 08 | `08_artifacts` | Artifacts & Dashboard | 27 | 7 | Artifact schema, diff, dashboard |
| 09 | `09_configuration` | Configuration & Tracking | 23 | 3 | Settings, YAML, env vars, trackers |
| 10 | `10_method_comparison` | Method Comparison | 19 | 6 | Head-to-head benchmark |

**Total:** 209 cells, 51 error-tolerant cells across 10 notebooks × 3 file formats = 30 files.

### Why most don't work

1. **LLM dependency** — `FeatureForge()` requires a valid API key (DeepSeek default). Every cell that calls `.fit()`, runs a pipeline, or uses baselines fails without credentials.
2. **`#| error: true` masking** — Quarto's error tolerance hides failures silently; cells fail but the notebook "renders."
3. **DeepSeek provider hardcoded** — Many notebooks explicitly `from feature_forge.llm.providers.deepseek import DeepSeekProvider` and construct it directly, which crashes if `DEEPSEEK_API_KEY` is unset.
4. **Optional dependencies** — WandB/MLflow trackers fail when packages aren't installed.
5. **Redundant data loading** — Each notebook independently loads `make_classification` data with slightly different parameters.

### Redundancy map

| Overlap | Notebooks |
|---------|-----------|
| Data loading (make_classification + train/test split) | ALL 10 |
| FeatureForge fit/transform demo | 01, 03, 04, 06, 08, 10 |
| Router + ablation modes | 01, 03 |
| Baseline method comparison | 06, 10 |
| Artifact diff/dashboard | 06, 08, 10 |
| Experiment tracking (WandB/MLflow) | 05, 09 |
| Configuration/Settings | 01, 03, 09 |
| LLM client setup | 02, 03, 04, 05, 06, 08, 10 |

---

## 2. Proposed Consolidation: 10 → 3 Notebooks

### A. `01_getting_started.ipynb` — Getting Started (replaces 01, 09)

**Goal:** A notebook that works **without an API key**. Demonstrates the full API surface end-to-end.

| Section | Content | Works offline? |
|---------|---------|---------------|
| Setup & Config | `Settings`, env vars, YAML, constructor overrides, validation | ✅ |
| Data Loading | Synthetic data + DatasetRegistry | ✅ |
| Evaluation Basics | CVEvaluator, ModelFactory, metrics, sandbox safety demo | ✅ |
| FeatureForge Overview | Architecture diagram (markdown), mode descriptions | ✅ |
| Quick Demo (LLM) | `FeatureForge.fit()` + transform + predict — single cell, error-tolerant | ⚠️ needs key |

**Key change:** Moves config (09) and evaluation (07) into context since they don't need an LLM. The single LLM-dependent cell at the end is clearly marked.

### B. `02_pipeline_deep_dive.ipynb` — Pipeline & Agents (replaces 02, 03, 04)

**Goal:** Deep exploration of the multi-agent pipeline. Requires API key.

| Section | Content |
|---------|---------|
| Agent Registry | 6 built-in agents, capabilities, custom agent |
| Router Strategies | data_driven, performance_driven, hybrid, llm |
| Iterative Pipeline | Multi-round generation, memory system |
| Ablation Modes | full vs no_memory vs no_router vs single-agent |

### C. `03_benchmarks_and_artifacts.ipynb` — Benchmarks & Artifacts (replaces 05, 06, 08, 10)

**Goal:** Experiment running, baseline comparison, artifact analysis. Requires API key.

| Section | Content |
|---------|---------|
| Experiment Matrix | Cartesian product definition, ExperimentRunner |
| Baselines | OpenFE, CAAFE, LLM-FE, Malmus |
| Method Comparison | Head-to-head benchmark with timing |
| Artifacts | Schema, diff, dashboard, storage backends |
| Tracking | WandB/MLflow (optional, graceful fallback) |

---

## 3. File Format Strategy

### Problem: 3 formats (.ipynb, .qmd, .py) are redundant

| Format | Current role | Recommendation |
|--------|-------------|----------------|
| `.ipynb` | Rendered output (notebooks) | **Primary source of truth** — edit directly |
| `.qmd` | Quarto input, generates .ipynb | **Delete** — not needed if we edit .ipynb directly |
| `.py` | Quarto python-format output | **Delete** — derivative of .qmd |

### Action
- Delete all `.qmd` files after consolidation
- Delete all `.py` notebook files after consolidation
- Delete `_quarto.yml`, `_freeze/`, `.quarto/` — Quarto infra not needed
- Keep only `.ipynb` as the canonical format

---

## 4. Implementation Phases

### Phase 1: Make notebooks work ✅ (do first)
- [ ] Extract shared setup into a `_utils.py` helper (data loading, LLM client with graceful fallback)
- [ ] Add `try/except` around LLM-dependent cells with clear error messages ("Set DEEPSEEK_API_KEY to run this cell")
- [ ] Make `FeatureForge.__init__` accept `llm_client=None` without crashing (defer error to `.fit()`)
- [ ] Add a mock/dry-run mode to `FeatureForge` for offline demo

### Phase 2: Consolidate 10 → 3
- [ ] Create `01_getting_started.ipynb` (merges old 01, 07, 09)
- [ ] Create `02_pipeline_deep_dive.ipynb` (merges old 02, 03, 04)
- [ ] Create `03_benchmarks_and_artifacts.ipynb` (merges old 05, 06, 08, 10)
- [ ] Verify each new notebook runs cell-by-cell

### Phase 3: Cleanup
- [ ] Delete old 10 `.ipynb` files
- [ ] Delete all `.qmd`, `.py`, `_quarto.yml`, `_freeze/`, `.quarto/`
- [ ] Update any references in docs/README

---

## 5. Key Design Decisions

1. **`.ipynb` as source of truth** — Most natural for interactive exploration. Quarto adds complexity without benefit here.
2. **Offline-first Notebook 1** — The getting-started notebook should work with zero configuration. LLM cells are bonus.
3. **Shared `_utils.py`** — Eliminates 10× repeated data loading and LLM setup boilerplate.
4. **Graceful degradation** — Cells that need LLM keys should print a helpful message, not crash silently.
5. **3 notebooks instead of fewer** — Each has a distinct audience (new user → pipeline developer → researcher).

---

## 6. Risk / Considerations

- **Quarto rendering pipeline** — If there's a CI/CD step that renders .qmd → docs, we need to replace it with `jupyter nbconvert` or similar.
- **`_freeze/` cache** — Contains rendered outputs; deleting it means notebooks lose cached outputs (fine, we'll re-run).
- **Links from other docs** — `docs/plan/*.md` files may reference old notebook paths; need to update.
