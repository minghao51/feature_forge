# Fail-Fast and Cooperative Cancellation Contract

**Date:** 2026-09-14  
**Status:** Approved for implementation planning; PR 1 must accept an ADR before runtime changes  
**Decision owner:** Maintainer  
**Prerequisite:** implement ADR 0016 legacy-engine removal first  
**Related:** ADR 0006, ADR 0016,
`docs/plan/21_hamilton_default_execution_handoff.md`, `.planning/STATE.md`

## 0. Continuation checkpoint

Plan 21 technical legacy-removal gates 1–5 pass, and ADR 0016 accepts removal
of the compatibility engine. The remaining plan 21 definition-of-complete gap
is an explicit fail-fast and cancellation contract at the outer experiment
scheduler. Current behavior always evaluates the complete case list; workers
normalize failures into `ExperimentResult.error`, and the process adapter eagerly
submits every case. There is no public cancellation token, no cancellation
result, and no focused cancellation lifecycle coverage.

The next implementation thread should remove legacy execution under ADR 0016,
then implement this plan in separate reviewable changes. Do not combine legacy
removal with scheduler behavior changes.

## 1. Goals

1. Preserve current continue-on-error behavior by default.
2. Allow an experiment to stop scheduling new cases after the first terminal
   failed case.
3. Allow callers to request cooperative cancellation without signals, globals,
   provider construction, or cross-process runtime objects in case payloads.
4. Return one ordered, typed result per requested case, including cases that
   never started.
5. Record cancellation truthfully without deleting or pretending to complete
   durable packages.
6. Apply identical semantics to sequential and process execution.
7. Keep tracker effects parent-owned and all Hamilton drivers worker-local.

## 2. Normative behavior

### 2.1 Failure policy

Add a string enum:

```python
class FailurePolicy(StrEnum):
    CONTINUE = "continue"
    FAIL_FAST = "fail_fast"
```

Add `ExecutionPolicyConfig` to `Settings`:

```python
class ExecutionPolicyConfig(BaseModel):
    failure_policy: FailurePolicy = FailurePolicy.CONTINUE
```

The environment form is `FF_EXECUTION__FAILURE_POLICY=fail_fast`. Add an
optional `failure_policy=` override to `ExperimentalPlatform.run()`. The run
override wins only for that invocation and must be reflected in resolved
configuration/provenance; it must not mutate cached global settings.

`continue` remains the default for backward compatibility. `fail_fast` triggers
only on a terminal case result whose typed state is `failed`. Feature-level
rejections, warnings, cache misses, recovered retries, and successfully handled
partial candidates do not trigger it.

### 2.2 Result state and cardinality

Add an explicit `state: RunState` to `ExperimentResult` and its public serialized
row. Every requested case produces exactly one result in original matrix order:

- `succeeded`: the case completed successfully;
- `failed`: the case completed with a terminal failure;
- `cancelled`: the case never started because of fail-fast or caller
  cancellation.

Existing `error` remains for compatibility. A cancelled result carries a typed
`FailureRecord` with `failure_class=cancelled`, a stable non-sensitive
`error_type`, the case/attempt identity allocated by the parent, no stage
packages, and no fabricated score. Do not classify an already-running case as
cancelled after it has produced a real success or failure.

### 2.3 Cancellation token

Add a small parent-process `CancellationToken` with thread-safe:

```python
cancel(reason: str = "operator_request") -> None
is_cancelled() -> bool
reason: str | None
```

`ExperimentalPlatform.run(cancellation_token=...)` checks the token before any
case starts and between case completions. The token is not serialized into
`CaseComputationInput` and never enters a Hamilton DAG or cache key. The first
implementation is cooperative at case boundaries; it does not kill a running
model, sandbox, provider call, or stage publication.

### 2.4 Sequential semantics

- Under `continue`, preserve current behavior exactly.
- Under `fail_fast`, execute in matrix order until one terminal failed result,
  retain that failure, and emit cancelled results for every later case without
  constructing a method, provider, driver, or tracker run.
- If the token is already cancelled, execute no cases and return only cancelled
  results.
- If cancellation arrives during a case, allow that case to finish safely, then
  cancel the remaining cases.

### 2.5 Process semantics

Replace eager submission of the complete matrix with a bounded submission
window of at most `max_workers` futures.

- Keep refilling the window while policy/token permits.
- After a terminal failure under `fail_fast`, submit nothing else.
- Attempt `Future.cancel()` for submitted work that has not started.
- Allow already-running workers to finish; retain their real results and
  lifecycle evidence.
- Convert never-submitted and successfully cancelled-before-start cases to typed
  cancelled results.
- Preserve original result order and cardinality regardless of completion order.
- Close/join the executor before returning; do not leak worker processes.

The exact set already running at the stop point may contain up to `max_workers`
cases. This is bounded, documented behavior—not a promise of immediate hard
termination.

### 2.6 Keyboard interruption

Library code must not swallow `KeyboardInterrupt` or convert it into an ordinary
failed experiment. On interruption:

1. stop new submission;
2. request cancellation of pending futures;
3. close scheduler resources safely;
4. journal cancellation for cases proven not to have started where identities
   are available;
5. re-raise `KeyboardInterrupt` to the caller.

Running external calls remain governed by their existing timeout mechanisms.

## 3. Lifecycle, tracker, cache, and artifact rules

- Parent-owned lifecycle events use `event_type="case_cancelled"` and
  `state=cancelled`, with a typed cancellation `FailureRecord`.
- Record only cancellation category and the triggering case fingerprint; never
  copy exception text, prompts, feature values, inputs, or secrets into bounded
  telemetry.
- Never initialize a tracker run for a case that never started. In-flight cases
  that actually complete keep exactly-once tracker handling.
- Cancellation does not delete Hamilton cache entries, LLM cache entries,
  staging directories, or verified packages.
- A later invocation may resume from any verified contiguous prefix produced by
  a case that started before cancellation.
- Fail-fast is evaluated only after any implemented case-level retry policy has
  produced a terminal result. Retry implementation itself is outside this plan;
  the existing `ResourceConfig.max_attempts` gap must not be silently claimed as
  solved here.

## 4. Public compatibility

- Default calls remain continue-on-error and return the same number/order of
  rows as today, with the additive `state` field.
- `parallel=False` and `parallel=True` share the same policy vocabulary.
- Invalid failure-policy values fail before dataset loading or provider
  construction.
- Once ADR 0016 removal lands, stale `engine=legacy` configuration fails with
  migration guidance independently of this policy.
- No provider abstraction, distributed queue, signal-handler installation, or
  remote cancellation service is introduced.

## 5. Implementation sequence

### PR 1 — ADR and typed contracts

**Before code:** add and accept ADR 0017 for the failure/cancellation contract.

Owned scope:

- `docs/decisions/0017-*.md` and ADR index;
- `src/feature_forge/config.py`, `config/settings.yaml`;
- `src/feature_forge/contracts/orchestration.py`;
- `src/feature_forge/experiment/execution.py` result/token contracts;
- focused config/contract tests and migration documentation.

Acceptance:

- strict enum validation and environment override;
- explicit result state for success/failure/cancellation;
- thread-safe, idempotent token with first-reason wins;
- no provider/source/cache side effects on invalid policy.

### PR 2 — Sequential fail-fast and lifecycle

Owned scope:

- `SequentialExecutionAdapter` scheduling control;
- `ExperimentalPlatform` policy resolution and cancelled-result factory;
- parent lifecycle helper for never-started cases;
- tracker behavior and focused unit/integration tests.

Acceptance:

- default policy executes all cases despite failures;
- fail-fast stops immediately after the first terminal failure;
- pre-cancelled token invokes zero workers;
- result order/cardinality and typed states are exact;
- cancelled cases initialize no tracker/provider/driver;
- later rerun can reuse valid prefixes from completed cases.

### PR 3 — Bounded process scheduling

Owned scope:

- `ProcessPoolExecutionAdapter` bounded submission;
- pending-future cancellation and executor shutdown;
- Linux-default and explicit `spawn` tests;
- process lifecycle/tracker assertions.

Acceptance:

- no more than `max_workers` cases are in flight;
- no new cases are submitted after the stop condition;
- already-running outcomes remain truthful;
- never-started rows are cancelled and ordered correctly;
- no worker leak after normal, fail-fast, token-cancelled, or interrupted runs;
- sequential/process semantic parity holds for deterministic schedules.

### PR 4 — Operator documentation and completion audit

Owned scope:

- README/API/operations/migration documentation;
- plan 21 and plan index status;
- generated/public examples if affected;
- `REPORT_LOG.md` and continuation handoff.

Acceptance:

- documentation distinguishes fail-fast from hard termination;
- examples show settings, environment, and per-run override;
- lifecycle and result-state fields are documented;
- plan 21 dry-run, resume, replay, retry, cancellation, sequential, and process
  claims are audited literally; any retry gap remains explicit rather than
  being marked complete.

## 6. Required tests

At minimum:

1. Continue policy: success/failure/success all execute in both adapters.
2. Sequential fail-fast: success/failure/cancelled/cancelled.
3. Token pre-cancel: all cancelled; worker/provider/source call counts are zero.
4. Token during run: current case completes; later cases cancel.
5. Process fail-fast with `max_workers=2`: bounded in-flight completion plus
   cancelled tail, under default and `spawn` contexts.
6. A pending future successfully cancelled before worker start.
7. `KeyboardInterrupt`: cleanup runs and interruption is re-raised.
8. Tracker: exactly once for actual completions, never for unstarted cases.
9. Lifecycle: one redacted `case_cancelled` event per unstarted Hamilton case.
10. Resume: verified packages from completed/in-flight cases remain reusable.
11. Serialization: process payloads contain no token or live service.
12. Backward compatibility: omitted policy matches current continue behavior.

Tests must use top-level pickleable workers, fresh temporary artifact/cache
roots, no providers/network, explicit process cleanup, and controlled native
thread limits where sandbox/model work is involved.

## 7. Non-goals and deferred escalation

- Hard-killing a running process, model fit, sandbox, or provider request.
- Distributed cancellation across machines.
- Installing global SIGINT/SIGTERM handlers in library code.
- Treating cache cleanup or artifact deletion as cancellation.
- Implementing the currently dormant `ResourceConfig.max_attempts` retry fields.
- Adding a remote execution backend or tracking platform.

Revisit in-case cooperative cancellation only after measurements show that
case-boundary cancellation is operationally inadequate. Such a change would
require cancellation checks at stage/provider/sandbox boundaries and a new ADR.

## 8. Validation

Run affected focused suites first, then:

```bash
uv sync --all-groups --extra intel
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 uv run pytest
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

Process scheduling changes must also run on Python 3.11, 3.12, and 3.13 and
exercise both the Linux-default and explicit `spawn` contexts.

## 9. Definition of complete

This plan is complete when the default continue behavior is preserved,
fail-fast and token cancellation have typed deterministic result semantics,
sequential/process tests pass, never-started cases have redacted lifecycle
evidence, no live service crosses the process boundary, public documentation is
accurate, and the plan 21 completion audit no longer overstates cancellation or
retry coverage.
