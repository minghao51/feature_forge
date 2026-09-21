# PR 5 Handoff: Scheduler, Resume, Resources, and Failure Isolation

**Status:** Implemented and verified 2026-07-14
**Depends on:** Stable Bronze-Silver-Gold-Platinum contracts
**Index:** [`2026-07-14-medallion-pr-index.md`](2026-07-14-medallion-pr-index.md)

## Objective

Make multi-case execution failure-aware, resumable, and resource-bounded while
preserving equivalent sequential and process behavior.

## Confirmed PR 4 inputs

- A reusable result requires a verified Platinum manifest and `_SUCCESS`, valid
  Silver and Gold upstream references, matching `platinum_input_fingerprint`,
  and passing semantic reconstruction checks.
- Missing required folds, mismatched baseline/enhanced prediction keys, or
  unreconstructable aggregates are failures, not partial successes.
- The enriched `ExperimentResult` preserves its first nine fields and supplies
  stage fingerprints, Platinum manifest URI, numeric uncertainty, directional
  gain, and selection profile with identical sequential/process serialization.
- Effective model and BLAS thread limits are already captured in the Platinum
  request. PR 5 owns outer/nested concurrency coordination and resume policy.

## Scope

### Scheduler contract

- Implement a concrete scheduler behind the existing outer control plane.
- Return one structured outcome per case; one failure must not cancel unrelated
  cases by default.
- Add explicit `fail_fast` behavior.
- Preserve input ordering or document a stable result-order contract.
- Keep original exception context through a structured failure record rather
  than replacing every worker failure with a generic `RuntimeError`.
- Support cancellation state and tracker finalization.

### Parallel registration/tracker correctness

- Ensure instance-local datasets, models, metrics, and supported methods are
  serialized to workers or rejected before scheduling.
- Do not rely on class-level registry mutation leaking through fork semantics.
- Define tracker ownership: worker-local run tracking or a central event sink.
- Make configured tracker behavior equivalent in sequential and process modes.
- Avoid constructing an unused tracker in the parent process.

### Resume and reusable-layer lookup

- Find reusable artifacts by layer fingerprint and verified manifest.
- Resume from the last committed valid layer.
- Reject incomplete, corrupt, wrong-version, or policy-incompatible packages.
- Never broaden recovery to upstream regeneration without explicit policy.
- Record reused/skipped/executed stages in the run manifest.
- Produce a full dry-run plan without network, provider, catalog mutation, or
  artifact writes.

### Resource policy

Add a typed resource plan covering:

- outer process workers;
- LLM concurrency;
- sandbox processes/time/memory;
- candidate execution batches;
- CV workers/backend;
- model threads;
- BLAS/OpenMP threads;
- retry/backoff limits.

Default to one heavy parallel layer at a time. Resolve and log the effective
plan for every run without exposing secrets.

### Retry/failure classification

- Use `FailureClass` consistently: transient, deterministic, policy, resource,
  and cancelled.
- Retry only explicitly transient failures by default.
- Resource retries must use a safer plan, not repeat the same allocation.
- Deterministic validation/code/data failures must not auto-retry.

### Local artifact crash recovery

- Detect and safely handle stale commit locks.
- Garbage-collect abandoned staging directories under an explicit policy.
- Never delete a committed namespace during cleanup.
- Add tests simulating interruption before/after manifest and `_SUCCESS`.

## Expected file areas

- `src/feature_forge/experiment/execution.py`
- `src/feature_forge/experiment/case_executor.py`
- `src/feature_forge/platform.py`
- scheduler/run repository implementations
- `src/feature_forge/config.py`
- `src/feature_forge/storage/atomic.py`
- `src/feature_forge/storage/local.py`
- focused process/resume/resource tests

## Tests to add

- One failed case does not cancel independent cases by default.
- `fail_fast=True` cancels or skips remaining work predictably.
- Sequential and process modes produce compatible result/failure shapes.
- Custom model/metric/dataset behavior is portable or rejected clearly.
- Trackers initialize/log/finalize once per case in both modes.
- Resume skips verified upstream layers and executes only invalid downstream
  layers.
- Fingerprint changes invalidate exactly the intended downstream layers.
- Dry-run creates no files and makes no network/provider calls.
- Effective nested concurrency never exceeds the resource plan.
- Stale locks/staging are recovered safely; live locks are not stolen.

## Acceptance criteria

- All cases receive a terminal structured outcome.
- Independent work survives another case's failure by default.
- Resume begins at the last verified compatible layer.
- Resource planning is deterministic, logged, and directly testable.
- Tracker and registration semantics no longer depend on operating-system fork
  behavior.
- Cleanup cannot remove or overwrite a committed package.

## Explicitly out of scope

- Distributed or remote scheduler.
- Kubernetes/Ray/Celery integration.
- DuckDB catalog implementation.
- Automatic resource benchmarking across arbitrary hardware.

## Verification

```bash
uv run ruff check .
uv run mypy src
uv run pytest tests/unit/test_execution_seams.py tests/unit/test_platform.py -q
uv run pytest tests/integration/test_platform_e2e.py -q
uv run --extra pipeline pytest -m "not llm" -q
uv run python scripts/run_pip_audit.py
git diff --check
```

## Completion handoff requirements

Report scheduler semantics, serialized worker payload, tracker ownership,
failure taxonomy, resource defaults, resume decision algorithm, crash-recovery
policy, and catalog events PR 6 can consume.

## Completion record

- `ExperimentResult` retains its first nine fields in their original order and
  appends a versioned structured failure record, attempt number, effective
  resource plan, and dry-run execution plan. Failures preserve category, type,
  bounded cause types, stage, retryability, and a redacted message.
- Sequential and spawn-based process adapters return exactly one terminal
  outcome per input position. Process submission is bounded by
  `experiment_workers`; independent failures are isolated by default.
  `fail_fast=True` stops further submission, retains already-running outcomes,
  and marks never-started cases cancelled.
- Worker payloads explicitly carry settings, dataset/method/model/metric
  overrides, tracker configuration, and the effective resource plan. Portable
  top-level registrations work under spawn; lambdas/closures become terminal
  serialization failures before worker execution. Explicit tracker instances
  remain sequential-only.
- Configured trackers are worker-owned in process mode and are initialized,
  logged, and finalized per case. Tracker init/log/finalize degradation marks
  an otherwise valid computation `PARTIAL`; it does not replace durable
  evidence or mask the computation failure.
- `ResourceConfig` resolves to one immutable `EffectiveResourcePlan`. Only one
  of outer workers, LLM concurrency, or CV workers may be heavy. The plan also
  constrains sandbox concurrency, candidate batch size, model threads,
  BLAS/OpenMP threads, memory, retry attempts, and backoff. Resource retries
  require a strictly smaller allocation; explicitly transient LLM failures are
  the only LLM failures retried.
- Resume lookup scans committed local packages by layer fingerprint, verifies
  `_SUCCESS`, manifest/schema/state, required artifact hashes, and exact
  upstream references, then selects only the longest valid contiguous
  Bronze-Silver-Gold-Platinum prefix. A corrupt preferred run fails closed
  unless `recompute_invalid` is explicit. `execute_resume_plan` executes only
  the missing suffix and re-verifies fingerprint and lineage after each stage.
- Dry-run validates configuration and registries, expands cases, resolves
  resources, validates metric existence and dataset-task compatibility, and
  returns the actual read-only resume decisions without tracker/provider/model/
  sandbox initialization, catalog mutation, lifecycle writes, or artifact
  writes. A non-legacy plan fails closed unless the caller supplies resolved
  per-run layer fingerprints.
- `LocalRunRepository` writes append-only versioned JSONL events plus an atomic
  current-state snapshot. PR 6 can consume `case_planned`, `case_running`,
  typed `stage_terminal`, and `case_terminal` events immediately. Stage,
  disposition, fingerprint, manifest reference, typed failure, attempt, and
  effective-resource details are populated fields rather than opaque details.
- Artifact publication uses token-owned commit locks, manifest-anchored success
  markers, file/directory fsync boundaries, and write-once namespace rename.
  Staging areas carry owner metadata. Cleanup removes only aged legacy state or
  provably dead same-host owners, retains live/foreign/symlinked state, and
  never removes a committed namespace.

### Verification results

- Adversarial scheduler/resume/resource/retry/lifecycle/platform lane: 82 passed.
- Full non-LLM lane after adversarial remediation: 830 passed at 88% coverage.
- Ruff and strict mypy pass across 104 source files.
- Strict MkDocs, lockfile, diff, repository hygiene, docs-reference, and
  dependency-audit gates pass. The dependency audit reports no known
  vulnerabilities and skips only the editable local distribution.

### Adversarial remediation (2026-07-15)

- Serialization preflight catches `PicklingError` and arbitrary reducer
  failures only around `ForkingPickler.dumps`, while interrupts still escape.
  Immediate submit rejection participates in fail-fast and never-started cases
  become typed cancellations.
- `layer_boundaries` no longer falls through to legacy computation. Platform
  execution requires a portable layer executor and per-run fingerprints,
  executes only the missing suffix, and semantically reloads Silver, Gold, and
  Platinum. The executor receives the complete verified upstream map.
- Resume runs the layer-specific semantic loaders and requires the exact
  Bronze-Silver-Gold-Platinum request/manifest chain; generic hash verification
  alone cannot authorize reuse.
- Outer or CV parallelism forces model and BLAS threads to one. Effective plans
  reject multiplicative model/BLAS and sandbox/batch products; deterministic
  resolution serializes nested products.
- Generic `OSError` and `ConnectionError` are deterministic unless a provider
  supplies an explicit transient subtype/status. Retry results synchronize the
  terminal attempt in both the result and failure record and expose the actual
  safer resource plan used by that attempt.
- Semantic loader failures, including `DatasetError`, invalidate and clear the
  candidate manifest reference before `recompute_invalid` selects execution;
  an invalid semantic package cannot remain marked reused.
- A mid-suffix layer failure raises `ResumeExecutionError` with the durable
  partial plan. Prior reused/executed stages retain successful states, the
  failed stage is recorded with `FAILED`, and lifecycle output emits those
  typed stage events before the failed case terminal event.
