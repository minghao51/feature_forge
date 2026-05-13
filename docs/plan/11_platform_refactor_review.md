# Experimental Platform Refactor — Post-Implementation Review & Fixes

**Date:** May 2026
**Status:** Completed
**Parent:** `10_experimental_platform_refactor.md`

---

## Executive Summary

Phases 1–3 of the Experimental Platform Refactor are fully implemented and all unit tests pass. This document captures the issues, gaps, and optimizations identified during code review, and serves as the implementation plan for fixing them.

---

## Issues Catalog

### 1. Global vs. Local Registration Inconsistency

**Severity:** High
**File:** `src/feature_forge/platform.py`

- `register_baseline()` stores in `self._extra_baselines` (instance-local)
- `register_model()` and `register_metric()` mutate `ModelRegistry._builtin` and `MetricRegistry._builtin` (global class state)

**Fix:** Make all registrations instance-local. `list_models()` and `list_metrics()` should union built-in + instance-local extras.

---

### 2. Missing Integration Tests

**Severity:** High
**Files:** `tests/integration/`

The plan calls for:
- `tests/integration/test_platform_e2e.py` — end-to-end with real components
- `tests/integration/test_plugin_discovery.py` — verify entry-point discovery

Neither exists. Unit tests mock all external dependencies.

**Fix:** Create both integration test files.

---

### 3. CVEvaluator / ModelFactory Re-created Per Config

**Severity:** Medium
**File:** `src/feature_forge/platform.py` (lines 205–206)

`ModelFactory` and `CVEvaluator` are instantiated inside `experiment_fn`, which runs once per `(dataset, baseline, model, seed)` combination. These are stateless and should be created once.

**Fix:** Instantiate outside `experiment_fn` and close over them.

---

### 4. No Tracker / Parallel Execution Exposure

**Severity:** Medium
**File:** `src/feature_forge/platform.py`

`ExperimentalPlatform.run()` hardcodes `NoOpTracker` and only uses `ExperimentRunner.run()` (sequential). The underlying `ExperimentRunner` supports `run_parallel()` and arbitrary trackers.

**Fix:** Add `tracker: ExperimentTracker | None = None` and `parallel: bool = False` parameters.

---

### 5. DatasetRegistry Entry Point Stubbed Metadata

**Severity:** Medium
**File:** `src/feature_forge/data/registry.py` (lines 76–78)

Entry-point datasets are loaded with hardcoded `target: None, task: "classification"`. The registry doesn't call the loader to get metadata until `load()` is called.

**Fix:** When an entry point is discovered, call the loader once to extract metadata if the loader supports it (or document that metadata comes from the loader at `load()` time).

---

### 6. DatasetRegistry Scans Entry Points on Every Instantiation

**Severity:** Medium
**File:** `src/feature_forge/data/registry.py`

`_load_entry_point_datasets()` is called in `__init__`. Should be lazy like `MetricRegistry` / `ModelRegistry`.

**Fix:** Move to lazy initialization triggered by first `list()` / `info()` / `load()` call.

---

### 7. BaselineRegistry Has No Discovery Cache

**Severity:** Low
**File:** `src/feature_forge/baselines/base.py`

`MetricRegistry` and `ModelRegistry` cache discovered entry points in `_discovered`. `BaselineRegistry.get_all_baselines()` calls `discover()` every time, re-scanning entry points.

**Fix:** Add `_discovered: ClassVar[dict | None] = None` and lazy-load pattern.

---

### 8. No BaselineProtocol Validation at Discovery Time

**Severity:** Medium
**File:** `src/feature_forge/baselines/base.py`

`BaselineRegistry.discover()` loads entry points without checking `isinstance(cls, BaselineProtocol)`. Malformed entry points crash at instantiation time, not discovery time.

**Fix:** Add runtime protocol check with a clear warning on failure.

---

### 9. No Duplicate-Name Warnings

**Severity:** Low
**Files:** All registries

Plan specifies "warn on duplicate names" for collision handling. Registries silently override or skip.

**Fix:** Add `warnings.warn(...)` when a duplicate name is encountered.

---

### 10. Double Exception Handling

**Severity:** Medium
**File:** `src/feature_forge/platform.py`

`platform.run()` wraps baseline execution in `try/except` (line 189), but `ExperimentRunner.run()` also wraps `experiment_fn` in `try/except` (line 62). The inner catch prevents the outer catch from firing for baseline errors. Inner error dict lacks config context.

**Fix:** Remove the inner `try/except` from `experiment_fn` and let `ExperimentRunner` handle it. The runner already appends the full config to the error result.

---

### 11. No Progress Reporting

**Severity:** Medium
**Files:** `src/feature_forge/platform.py`, `src/feature_forge/experiment/runner.py`

Long-running `platform.run()` gives no user feedback.

**Fix:** Add `tqdm` progress bar to `ExperimentRunner.run()` (and `run_parallel()`).

---

### 12. Fragile Dataset Loader Functions

**Severity:** Low
**File:** `src/feature_forge/data/registry.py` (lines 13–20)

`titanic_loader()` and `house_prices_loader()` create a new `DatasetRegistry()` instance. Works today due to built-in shadowing, but is fragile.

**Fix:** Keep as-is but add a comment explaining the recursion guard, or refactor to use a module-level singleton.

---

### 13. Test Data Ignored

**Severity:** Low
**File:** `src/feature_forge/platform.py`

`platform.run()` loads `data["test"]` from the registry but never evaluates on it.

**Fix:** Document that test sets are loaded but not evaluated in `run()`. Optionally add a `evaluate_test: bool = False` parameter.

---

### 14. Missing API Stability Contract in Documentation

**Severity:** Low
**File:** `PLUGINS.md`

Plan notes "Need clear versioning/API stability contract". `PLUGINS.md` doesn't mention API versioning.

**Fix:** Add a "Plugin API Versioning" section to `PLUGINS.md`.

---

## Implementation Order

| Order | Issue | Effort | Files |
|-------|-------|--------|-------|
| 1 | Global vs. local registration | Small | `platform.py`, unit tests |
| 2 | BaselineRegistry cache + protocol validation | Small | `baselines/base.py`, unit tests |
| 3 | DatasetRegistry lazy loading + metadata | Small | `data/registry.py`, unit tests |
| 4 | Duplicate-name warnings | Small | All registry files |
| 5 | CVEvaluator cache + double exception fix | Small | `platform.py` |
| 6 | Tracker / parallel params + progress bar | Medium | `platform.py`, `runner.py` |
| 7 | Integration tests | Medium | New files |
| 8 | PLUGINS.md API contract | Small | `PLUGINS.md` |

---

## Verification Checklist

- [x] All existing unit tests pass
- [x] New integration tests pass
- [x] `ruff check src tests` clean
- [x] `mypy src` clean
- [x] `PLUGINS.md` updated
- [x] `docs/plan/00_index.md` references this document

---

## References

- Parent plan: `docs/plan/10_experimental_platform_refactor.md`
- Implementation: `src/feature_forge/platform.py`
- Tests: `tests/unit/test_*.py`
