# PR 4 Handoff: Platinum Fold Evidence and Uncertainty

**Status:** Implemented 2026-07-14
**Depends on:** Verified Silver folds and Gold features
**Index:** [`2026-07-14-medallion-pr-index.md`](2026-07-14-medallion-pr-index.md)

## Objective

Make evaluation reconstructable from durable fold-level evidence. Baseline and
enhanced results must consume the exact persisted Silver fold assignments, and
the final `ExperimentResult` must point to a verified Platinum package.

PR 3 clarification: Gold `accepted_features` are validation-accepted, not final
metric/effect-selected. Platinum applies final selection using persisted Silver
folds and retains per-feature decision reasons.

## Scope

### Evaluation contract

- Refactor `CVEvaluator` or add a compatible evidence-producing evaluator.
- Accept explicit row IDs and fold assignments; do not regenerate splitters.
- Produce per-fold baseline/enhanced predictions and metrics.
- Record model name, distribution/version, resolved hyperparameters, task,
  metric, preprocessing policy, seed, and evaluation fingerprint.
- Keep preprocessing fitted only on each training fold.
- Preserve error context for fold, model, metric, case, run, and stage.

### Platinum contracts and package

Add strict, versioned contracts for:

- evaluation request/policy;
- fold assignment reference;
- fold metric and prediction records;
- aggregate metric and uncertainty summary;
- selection policy and decision summary;
- Platinum materialization/package;
- final result reference.

Persist at least:

- baseline fold predictions;
- enhanced fold predictions;
- fold metrics and gains;
- aggregate metrics;
- uncertainty/effect evidence;
- model/evaluation configuration;
- upstream Gold and Silver manifest references;
- checks and final manifest.

### Selection and uncertainty

- Keep a compatibility profile reproducing the legacy `gain > 0` rule.
- Add a recommended policy with practical effect threshold and paired evidence.
- Start with paired fold/seed summaries; add bootstrap only when sample/repeat
  count makes it defensible.
- Make threshold and uncertainty policy part of the appropriate fingerprint.
- Record the reason for every final feature selection outcome.

### `ExperimentResult` compatibility

- Preserve the first nine legacy dataclass fields and serialization order.
- Populate `run_id`, case/stage fingerprints, state, manifest URI, and
  uncertainty where available.
- Keep Reporter compatible with both old and new result shapes.
- Ensure sequential/process serialization has identical semantics.

## Optimization work

- Reuse persisted fold indices and preprocessed base fold data where safe.
- Avoid recomputing unchanged baseline evidence for each candidate policy.
- Bound model and BLAS threads explicitly.
- Measure the current per-feature `O(features x folds)` evaluation path before
  changing batching or screening behavior.
- Do not trade deterministic evidence for an opaque cache shortcut.

## Expected file areas

- `src/feature_forge/evaluation/cv.py`
- `src/feature_forge/evaluation/kit.py`
- `src/feature_forge/contracts/` Platinum models
- `src/feature_forge/dataflows/platinum.py`
- `src/feature_forge/experiment/execution.py`
- `src/feature_forge/experiment/reporter.py`
- focused tests and generated DAG/docs

## Tests to add

- Baseline/enhanced rows and folds match exactly.
- Aggregates reconstruct from stored fold records.
- Reordering input rows with preserved row IDs is detected or normalized by an
  explicit policy; it is never silently accepted.
- Preprocessing state is trained only on the fold training subset.
- Model-only, metric-only, and evaluation-policy changes alter Platinum but not
  Silver/Gold identity.
- Compatibility profile matches legacy metrics within stated tolerance.
- Reporter handles legacy and enriched results.
- Sequential and process result serialization agree.
- Corrupt or incomplete fold evidence fails verification.

## Acceptance criteria

- Every published aggregate is recomputable from stored fold evidence.
- Baseline and enhanced evaluations use identical persisted folds.
- The final result points to a verified Platinum manifest.
- Compatibility behavior is measured and documented, not assumed.
- Recommended selection does not rely solely on an unqualified positive mean.
- No provider call is needed to reevaluate a verified Gold package.

## Explicitly out of scope

- Scheduler failure isolation and resume.
- Catalog and CLI.
- Distributed evaluation.
- Automatic hyperparameter optimization.

## Verification

```bash
uv run ruff check .
uv run mypy src
uv run --extra pipeline pytest tests/unit/test_platinum_dataflow.py -q
uv run pytest tests/unit/test_evaluation.py tests/unit/test_experiment.py -q
uv run pytest -m "not llm" -q
uv run python scripts/run_pip_audit.py
uv run mkdocs build --strict
git diff --check
```

## Completion handoff requirements

Report evidence schemas, reconstruction proof, compatibility tolerances,
selection defaults, performance measurements, and the scheduler inputs PR 5
must treat as authoritative.

## Completion record

- Schema version 1 adds strict model, evaluation, uncertainty, selection,
  aggregate, request, package, materialization, and final-result contracts.
- Platinum consumes verified Silver `row_id`/fold assignments and verified
  validation-accepted Gold candidates. Final model-evidence selection is
  recorded in Platinum and never mutates Gold.
- Every fold uses a fresh estimator and train-fold-only preprocessing. Model
  class, distribution/version, resolved parameters, seed, and estimator/BLAS
  thread limits participate in identity.
- Fold assignments, per-arm metrics, row-keyed predictions, aggregates,
  uncertainty, decisions, checks, and report are atomically committed. Loading
  rechecks hashes, upstream refs, fold/prediction parity, and reconstruction.
- Compatibility selection preserves legacy raw gain (`enhanced - baseline`)
  and `gain > 0`. Recommended selection uses explicit metric direction plus a
  practical threshold and/or positive lower bound. RMSE, MAE, and NRMSE are
  explicitly lower-is-better.
- Paired fold uncertainty records count, mean directional gain, sample standard
  deviation, standard error, and t-interval bounds. Bootstrap is not part of
  the version 1 uncertainty contract.
- `ExperimentResult` preserves its first nine fields and appends Silver/Gold/
  Platinum fingerprints, manifest reference, directional gain, selection
  profile, and numeric uncertainty. Sequential/process `asdict` semantics and
  mixed legacy/enriched Reporter behavior are covered.
- The reproducible performance smoke compares legacy mean-only and explicit
  evidence on identical persisted folds, asserts exact aggregate parity and
  durable evidence counts, and enforces the recorded time budget.
- PR 5 must treat the verified Platinum manifest, its upstream refs,
  `platinum_input_fingerprint`, required semantic checks, and final enriched
  result as authoritative scheduler outputs. Missing required folds or a failed
  semantic check is failure, not partial success.

### Verification results

- The expanded Silver/Gold/Platinum/evaluation/result/benchmark lane passed:
  105 tests.
- `uv run pytest -m "not llm" -q`: 799 passed, 15 deselected, 88% coverage.
- `uv run ruff check .`: passed.
- `uv run mypy src`: passed across 100 source files.
- `uv lock --check`, `git diff --check`, repository hygiene, docs references,
  default dependency audit, and strict MkDocs all passed. The audit found no
  known vulnerabilities and ignored only the editable project distribution.
