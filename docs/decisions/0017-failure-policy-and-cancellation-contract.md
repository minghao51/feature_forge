# ADR 0017: Failure policy and cooperative cancellation contract

- **Status:** Accepted (maintainer accepted on 2026-09-14)
- **Date:** 2026-09-14
- **Related:** ADR 0006, ADR 0016,
  `docs/plan/22_fail_fast_cancellation_contract.md`, `REPORT_LOG.md`

## Context

ADR 0016 removed the legacy execution engine, leaving Hamilton as the sole
case dataflow path behind the experiment execution seam (ADR 0006). The
remaining plan 21 completion gap is an explicit failure and cancellation
contract at the outer experiment scheduler. Current behavior always evaluates
the complete case list: the scheduler has no notion of stopping after a failed
case, results carry only a nullable `error` string with no typed terminal
state, and there is no public way to request cancellation. The process adapter
eagerly submits every case before the first result is observed, so an
unbounded matrix is fully in flight from the start.

Plan 22 (`docs/plan/22_fail_fast_cancellation_contract.md`) specifies the
required behavior. This ADR records the accepted design before any scheduler
code changes, per the decision order in `AGENTS.md`.

On 2026-09-14 the maintainer accepted this ADR, unlocking the runtime
scheduling changes (plan 22 PRs 2–3).

## Decision

The outer experiment scheduler gains the following contract. The typed
configuration/result vocabulary ships first (plan 22 PR 1); the runtime
scheduling behavior follows in separate changes after this ADR is accepted.

1. **Failure policy.** A `FailurePolicy` string enum (`continue`,
   `fail_fast`) is added to `Settings` as `execution.failure_policy`.
   `continue` is the default and preserves today's behavior exactly.
   `fail_fast` is evaluated only on a terminal failed case result:
   feature-level rejections, warnings, cache misses, recovered retries, and
   successfully handled partial candidates never trigger it.
2. **Typed result state and exact cardinality.** `ExperimentResult` gains an
   explicit terminal `state` (`succeeded`, `failed`, `cancelled`) that is also
   serialized into the public result row. Every requested case produces
   exactly one result in original matrix order, including cases that never
   started. Results without an explicit state derive `failed` when an error or
   failure record is present and `succeeded` otherwise; `cancelled` is only
   ever set explicitly. Cancelled results carry a typed `FailureRecord` with
   `failure_class=cancelled`, a stable non-sensitive error type, no stage
   packages, and no fabricated score. An already-running case that produced a
   real success or failure is never reclassified as cancelled.
3. **Cooperative cancellation token.** A small parent-process
   `CancellationToken` with thread-safe, idempotent `cancel(reason)`
   (first reason wins), `is_cancelled()`, and a read-only `reason` property.
   The token is checked at case boundaries only, is never serialized into
   `CaseComputationInput`, and never crosses the process seam into a Hamilton
   DAG, sandbox, provider call, or cache key. It does not kill running work;
   a case in flight when cancellation arrives is allowed to finish safely.
4. **Bounded process submission.** The process adapter replaces eager
   submission of the complete matrix with a bounded submission window of at
   most `max_workers` futures, refilled only while the policy and token
   permit. Submitted-but-not-started futures are cancelled with
   `Future.cancel()`; already-running workers finish and keep their real
   results and lifecycle evidence. The exact set already running at the stop
   point may contain up to `max_workers` cases — bounded, documented behavior,
   not a promise of hard termination.
5. **Interruption hygiene.** `KeyboardInterrupt` is never swallowed or
   converted into an ordinary failed experiment: the scheduler stops new
   submission, cancels pending futures, closes resources safely, journals
   cancellation for cases proven not to have started, and re-raises.

Sequential and process execution share the same policy vocabulary and must
hold semantic parity for deterministic schedules, preserving original result
order and cardinality regardless of completion order. Parent-owned lifecycle
evidence for unstarted cases uses the redacted `case_cancelled` event shape;
tracker runs are initialized only for cases that actually started, and
cancellation never deletes Hamilton cache entries, LLM cache entries, staging
directories, or verified packages.

**Non-goals** (plan 22 §7): no hard-killing a running process, model fit,
sandbox, or provider request; no distributed cancellation across machines; no
global SIGINT/SIGTERM handler installation in library code; no implementation
of the dormant `ResourceConfig.max_attempts` retry fields (fail-fast is
evaluated only after any implemented retry policy yields a terminal result,
and that retry gap remains explicitly open); and cache cleanup or artifact
deletion is not cancellation. In-case cooperative cancellation is revisited
only if measurements show case-boundary cancellation is operationally
inadequate, and would require a new ADR.

## Consequences

- Operators can stop paying for a matrix after its first terminal failure and
  can request cancellation without signals, globals, or provider
  construction, while default calls remain byte-for-byte compatible apart
  from the additive `state` field.
- Result consumers get an exact, typed, ordered one-result-per-case contract,
  making downstream reporting and resume logic trustworthy; the nullable
  `error` field remains for compatibility.
- Cancellation is cooperative: up to `max_workers` already-running cases may
  still complete (and be billed) after a stop condition. Documentation must
  distinguish fail-fast from hard termination.
- The token is parent-process-only, so it cannot be lost inside serialized
  payloads; the cost is that in-flight cases observe cancellation only at the
  next case boundary.
- This ADR was accepted by the maintainer on 2026-09-14, after PR 1 landed
  the ADR, typed contracts (`FailurePolicy`, `ExecutionPolicyConfig`, result
  `state`, `CancellationToken`, cancelled `FailureRecord` factory), and
  focused tests. PRs 2–3 (sequential fail-fast and lifecycle, bounded process
  scheduling) proceed under this acceptance; PR 4 closes documentation and
  audits plan 21/22 completion claims.
