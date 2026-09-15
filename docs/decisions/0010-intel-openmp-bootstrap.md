# ADR 0010: Intel Extension acceleration and OpenMP runtime isolation

- **Status:** accepted
- **Date:** 2025-08-31
- **Supersedes:** — (none)
- **Related:** `AGENTS.md` (architectural boundaries), `src/feature_forge/runtime/intel_bootstrap.py`, `src/feature_forge/evaluation/model_factory.py`, `src/feature_forge/methods/malmas/pipeline/core.py`, `REPORT_LOG.md`

## Context

The platform runs on Intel hardware and we want the speedup from Intel
Extension for Scikit-learn (`sklearnex`), which patches scikit-learn to execute
on the Intel oneAPI / DAAL backend. That backend loads the **Intel OpenMP
runtime (`libiomp5`)**.

The pip wheels of XGBoost and LightGBM (the project's default boosting models)
are built against the **GNU OpenMP runtime (`libgomp`)**. When two OpenMP
runtimes share one process and both try to drive threads simultaneously — for
example a booster fit with `n_jobs > 1` inside a `sklearnex`-patched CV loop —
they deadlock. The process pins at 100% CPU and never returns. This is the classic
`libiomp5` vs `libgomp` conflict reported against `scikit-learn-intelex` +
XGBoost / LightGBM (LightGBM FAQ #10/#11/#16; Intel `scikit-learn-intelex`
parallelism notes).

Measured state at decision time: the venv had **no `sklearnex` installed** and
xgboost/lightgbm/sklearn all linked `libgomp`, so there was no conflict. The
deadlock is a *latent* risk that materializes the moment Intel acceleration is
enabled. We chose to harden the setup now rather than debug a hang later.

## Decision

Enable `sklearnex` acceleration, but only through a single guarded bootstrap,
with the following concrete rules:

1. **Patch-first ordering.** `patch_sklearn()` runs as the very first
   import-time action in `feature_forge/__init__.py`, before any
   `sklearn` / `xgboost` / `lightgbm` submodule is imported.
2. **OpenMP safety env.** `KMP_DUPLICATE_LIB_OK=TRUE` is set before any OpenMP
   runtime loads, so a secondary libiomp5 load does not abort with `OMP: Error
   #15`. This is a safety net, not the deadlock cure.
3. **Same-process threading, no fork.** Boosting estimators stay at
   `n_jobs=1` (`evaluation/model_factory.py`) so libgomp never spawns threads
   alongside libiomp5. Evaluation uses the **threading** joblib backend in the
   *same* process whenever Intel acceleration is on
   (`methods/malmas/pipeline/core.py`). We deliberately avoid the `loky` /
   multiprocessing backends here: forking a child after OpenMP (libiomp5) has
   been initialized can deadlock in the child, which is the very hazard we are
   preventing. With `n_jobs=1` boosters, the threading backend runs fits in
   threads while libgomp stays idle, so the two runtimes never contend.
4. **Opt-in via config + extra.** Acceleration is gated by
   `Settings.intel_acceleration` (default `True`) and the `scikit-learn-intelex`
   package, installed through the `intel` optional dependency group. With the
   extra absent, the bootstrap logs `intel_bootstrap_unavailable` and proceeds
   with the stock libgomp stack (current behavior, no regression). Users can
   disable it with `FF_INTEL_ACCELERATION=false`.

## Consequences

- Intel hardware gets DAAL-accelerated sklearn for free once `uv add --optional
  intel` is run; enabling it cannot silently introduce the OpenMP deadlock
  because the `n_jobs=1` + threading (no fork) guards hold regardless of model
  choice.
- Results may differ slightly from the unpatched sklearn stack (DAAL algorithms
  are not bit-identical). This is the expected trade for acceleration and is
  controlled by a single config flag for reproducible A/B comparison.
- Remaining limitation: merely *loading* libiomp5 and libgomp in one process is
  still allowed. The only complete cure is a single OpenMP runtime, achieved by
  installing the Intel-optimized XGBoost/LightGBM builds (libiomp5-linked) via
  the `intel` conda channel. This is documented but out of scope for the pip/uv
  workflow; revisit if a true single-runtime build is required for an experiment.
- Follow-up: add a `threadpoolctl`-based diagnostic in CI to assert a single
  OpenMP runtime when the `intel` extra is active.

## Amendment (2026-09-06): opt-in parallel boosters when Intel is inactive

We investigated removing the `n_jobs=1` pin for the no-Intel case
(`booster_n_jobs = -1` when `is_intel_active()` is false). Synthetic-data
benchmarks on this repository's 14-core development host — which was carrying
a co-located 14-thread workload at the time, a realistic condition for how
this machine is actually used — falsified the blanket change:

| shape (5k×30, 3-fold CV, 200 trees) | xgb `n_jobs=1` | xgb `n_jobs=4` | xgb `n_jobs=-1` | lgbm `n_jobs=1` | lgbm `n_jobs=4` | lgbm `n_jobs=-1` |
|---|---|---|---|---|---|---|
| serial fit | 0.64 s | 0.53 s | **117.7 s** | 1.49 s | 0.62 s | **348.3 s** |
| nested (8 loky children) | 3.19 s | 0.66 s | — | 0.8–2.0 s | **32–38 s** | — |

`n_jobs=-1` collapses via the classic OpenMP spin-wait pathology under CPU
contention (185–234x serial, unmeasurable nested); even `n_jobs=4` is
bimodal — 5x win nested-xgb, 16–40x loss nested-lgbm — with the flip driven
by library-internal thread-pool behavior, not anything we control.

**Decision:** the pin stays the default. New opt-in knob
`Settings.evaluation.booster_n_jobs` (default `1`, validator permits `>= 1` or
`-1`) threads OpenMP-backed estimators (xgboost, lightgbm, random_forest)
when Intel acceleration is inactive. When Intel is active, `_booster_n_jobs()`
forces `1` regardless of configuration and logs
`booster_n_jobs_forced_single_thread` once — the deadlock invariant of this
ADR is absolute and cannot be opted out of. `ModelFactory` accepts
`booster_n_jobs` and plumbs it only to the three OpenMP-backed built-ins, so
entry-point and custom models are untouched.

Revive the auto-parallel default only with benchmarks on a quiet, dedicated
host showing consistent wins across both libraries and both shapes.
