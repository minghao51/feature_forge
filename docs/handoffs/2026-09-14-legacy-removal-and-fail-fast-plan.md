# Legacy-removal decision and fail-fast planning handoff

**Updated:** 2026-09-14  
**Workspace:** `/home/howt/work/feature_forge`  
**Authoritative plan:** `docs/plan/21_hamilton_default_execution_handoff.md`  
**Plan index:** `docs/plan/00_index.md`  
**Status:** PRs 1–6 and all six legacy-removal gates are complete; ADR 0016 authorizes removal, and plan 22 defines follow-on fail-fast/cancellation work. No commit created.

## Start here

Do not repeat PR 1–6 or the section 16 qualification. The next thread has two
ordered assignments:

1. Implement ADR 0016 legacy-engine removal as a dedicated change.
2. After removal validates independently, implement
   `docs/plan/22_fail_fast_cancellation_contract.md`, beginning with ADR 0017.

Do not combine the changes. The current checkout still contains `engine=legacy`;
the accepted decision authorizes removal but this documentation slice did not
perform it.

## Working-tree constraints

- The checkout contains substantial intentional pre-existing uncommitted work.
- Never reset, clean, broadly stash, or overwrite unrelated changes.
- Never commit unless explicitly requested.
- Treat `origin/feat/medallion-refactor` at `690a764` as read-only reference material.
- Never modify supplied files under `data/`; write experiment evidence under `experiments/`.
- Preserve the mandatory `LLMClient` DiskCache and sandbox execution boundary.
- Use fresh temporary Hamilton cache roots in tests.

## Implemented work

### PR 1–3: contracts, evidence, and stage DAGs

- ADRs 0012–0015 and typed Hamilton/dataflow/cache configuration.
- Strict Bronze/Silver/Gold/Platinum contracts, identities, manifests, verification,
  atomic package publication, resume planning, and exact ordered lineage.
- Separate stage drivers with tagged inventories, explicit recompute boundaries,
  provider-free Gold replay, and reconstructable Platinum evidence.

### PR 4: Hamilton public default

- Worker-local `HamiltonLayerExecutor` and `ExperimentalPlatform` Hamilton default.
- Verified incremental durable reuse, lifecycle journals, typed failures, retries,
  sequential/process parity, read-only planning, and explicit legacy rollback.

### PR 5: operations and documentation

- Bounded/redacted Hamilton telemetry and layer-aware cache events.
- WAL-backed concurrent metadata, atomic result publication, retention, inspection,
  guarded clear, artifact list/semantic verification CLI, and recovery guidance.
- Freshness-tested generated stage DAGs and strict MkDocs support.

### PR 6: correctness and dependency cleanup

- MALMAS baseline cache identity now includes complete X/y content and schema,
  folds, metric, task, seed, and model identity.
- Successful generated code is deduplicated, train-validated, and reused once per
  test partition. Name/index/dtype failures are retained per feature; failed test
  features cannot be selected or replayed.
- Metric minimization is direction-aware. `max_selected_features` replaces the
  misleading `min_effective` name; the old constructor/YAML key remains a migration alias.
- Unknown MALMAS modes/agents fail before provider construction. Normal fits reset
  memory/router state; `warm_start=True` explicitly opts into persisted learning.
- The core model default is `random_forest`. OpenFE, CAAFE fidelity, XGBoost,
  LightGBM, and CatBoost use named extras with precise installation errors.
- A clean-wheel package-contract CI lane verifies that standard installation has
  no heavyweight optional ML packages and that the default model fits/predicts.
- Concurrent SQLite WAL initialization now retries bounded lock contention.

## Important files

- Runtime/default path: `src/feature_forge/platform.py`,
  `src/feature_forge/experiment/hamilton_executor.py`
- MALMAS correctness: `src/feature_forge/api.py`,
  `src/feature_forge/methods/malmas/pipeline/core.py`,
  `src/feature_forge/methods/malmas/pipeline/iterative.py`,
  `src/feature_forge/methods/malmas/method.py`
- Evaluation/default model: `src/feature_forge/evaluation/cv.py`,
  `src/feature_forge/evaluation/model_factory.py`
- Cache/operations: `src/feature_forge/storage/hamilton_cache.py`,
  `src/feature_forge/observability/hamilton_adapter.py`, `src/feature_forge/cli.py`
- Package contract: `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`,
  `tests/unit/test_package_contract.py`
- Status/evidence: `REPORT_LOG.md`, `.planning/STATE.md`,
  `docs/plan/21_hamilton_default_execution_handoff.md`
- Accepted removal decision: `docs/decisions/0016-remove-legacy-execution-engine.md`
- Active scheduler plan: `docs/plan/22_fail_fast_cancellation_contract.md`

## Validation evidence

Required environment synchronization completed:

```bash
uv sync --all-groups --extra intel
```

Full suite completed with explicit native-thread limits to avoid the shared-host
BLAS/OpenMP worker exhaustion observed in earlier runs:

```bash
# Repeat with the Python 3.11, 3.12, and 3.13 project environments.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 uv run pytest
```

Results: Python 3.11, 3.12, and 3.13 each completed with 954 passed and
9 expected skips.

The nine skips are expected because XGBoost is now an optional extra and was not
installed by the required standard+Intel synchronization. The Intel/XGBoost CI
job explicitly installs `intel`, `xgboost`, and `lightgbm` together.

Also passing:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run mypy tests
uv run python scripts/check_repo_hygiene.py
uv run python scripts/check_docs_references.py
uv run python scripts/generate_stage_dag_docs.py --check
uv run mkdocs build --strict
uv lock --check
git diff --check
```

Additional evidence:

- `tests/integration/test_hamilton_execution_parity.py` proves real process
  recomputation parity under Linux-default and explicit spawn contexts and
  Hamilton/legacy metric parity within absolute tolerance `1e-9`.
- `tests/integration/test_hamilton_recovery.py` proves cache deletion/corrupt
  metadata recovery, tampered-Silver suffix recomputation, same-attempt resume,
  and provider-free Gold replay.
- `experiments/legacy_removal/2026-09-14/qualification_report.json` records a
  passed two-seed bundled breast-cancer matrix with verified packages, resolved
  secret-free settings, and parity deltas no larger than `3.71e-17`. Report
  SHA-256: `f6fa204c4a943690dd412e677577e313ae32073f856422568cfd893fed65b2f5`.
- The 100,000-row cache benchmark at
  `experiments/legacy_removal/2026-09-14/cache_benchmark/report.json` measured
  1,400.1 ms cold versus 327.0 ms warm with 10/10 eligible hits. Report SHA-256:
  `12821a0fbe22691c8fdd0798f17f4e8e0938b1cfa96ae917204a7dc2d60341b8`.
- A clean standard wheel was built and installed in an isolated venv; all of
  `caafe`, `openfe`, `xgboost`, `lightgbm`, and `catboost` were absent and the
  random-forest default fit/predict smoke passed.
- Hamilton cache qualification retained two source loads, zero provider calls,
  and zero sandbox executions. Performance remains workload-specific; do not
  generalize the representative result to every workload.
- Concurrent SQLite cache publication passed five repeated spawned-worker runs
  after adding bounded WAL initialization retry.

## Confirmed decisions and transitional state

- Gate 6 is complete: the maintainer accepted removal on 2026-09-14 and ADR 0016
  supersedes ADR 0012.
- `engine=legacy` remains temporarily present only because code removal is the
  next separately reviewable implementation change.
- Until removal lands, current-release behavior and documentation must remain
  truthful: the rollback flag still works and must never silently mix engines.
- Plan 22 is approved for implementation planning. Its first PR must record ADR
  0017 before changing the scheduler contract.
- Fail-fast defaults to `continue`; cancellation is cooperative at case
  boundaries; running work is not hard-killed; ordered result cardinality is
  preserved with typed cancelled rows.
- The generated artifact/package tree remains authoritative for completion;
  Hamilton cache hits are acceleration only.
- Cache GC must never touch medallion packages or the independent LLM response cache.
- Cache performance remains workload-specific despite the positive 100,000-row
  qualification; retain both the tiny and representative measurements.

## Suggested start for the next thread

Read these files first with the coding-agent file reader:

- `docs/README.md`
- `docs/plan/00_index.md`
- `docs/plan/21_hamilton_default_execution_handoff.md`
- `docs/decisions/0016-remove-legacy-execution-engine.md`
- `docs/plan/22_fail_fast_cancellation_contract.md`
- `.planning/STATE.md`
- `REPORT_LOG.md`

Then inspect and validate the workspace:

```bash
git status --short
uv run pytest tests/integration/test_platform_e2e.py -q
```

Review ADR 0016 and implement only the legacy-removal slice first. Validate and
update migration guidance before starting plan 22. The dormant case-retry fields
are outside plan 22 and must remain explicitly unimplemented.
