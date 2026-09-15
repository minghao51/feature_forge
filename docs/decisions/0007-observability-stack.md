# ADR 0007: Observability stack — structlog + OpenTelemetry + Langfuse

- **Status:** Accepted
- **Date:** 2026-08-29
- **Related:** `docs/plan/01_architecture.md` (observability layer);
  ADR 0001, ADR 0006

## Context

Multi-agent LLM runs are expensive and hard to debug: a failed
experiment needs structured logs, a way to correlate log lines with
traces, and per-call LLM visibility (prompts, tokens, cost) without
forcing every user to run a tracing platform. Per ADR 0001's
smallest-architecture preference, the stack had to be local-first with
optional cloud pieces, and no always-on APM dependency in the core run
path.

This record was written retroactively on 2026-08-29, after
implementation; see Related for the originating plan documents.

## Decision

1. **structlog for all logging** (`observability/structlog_config.py`):
   pretty console rendering in a TTY, JSON otherwise (production/CI);
   ISO-UTC timestamps, contextvars merging, level filtering via
   `FF_LOG_LEVEL`.
2. **OpenTelemetry context injection, not an SDK.** A structlog
   processor injects the current OTel span's `trace_id`/`span_id` into
   every event so logs correlate with any OTel-instrumented process; no
   exporter is configured by the package itself.
3. **Langfuse for LLM tracing, lazily and optionally**
   (`observability/langfuse_tracer.py`): `@observe`-decorated LLM call
   paths import Langfuse on first use and degrade with a clear error if
   the extra is not installed — the core run path has no hard Langfuse
   dependency.
4. **Experiment tracking stays separate.** WandB/MLflow record
   experiment-level metrics and artifacts via ADR 0006's opt-in tracker
   configuration; they are not part of the logging/tracing stack.

## Consequences

- Logs are machine-parseable in CI and human-readable in development
  with zero configuration; LLM calls can be traced end-to-end (with
  costs) by users who install and configure Langfuse.
- `cache_logger_on_first_use=True` means logging configuration is fixed
  at first use per process — changing it at runtime requires a restart
  (accepted trade-off).
- Trace/span correlation only materializes when an OTel context exists;
  without one, logs simply lack those fields.
- Replacing any leg of the stack (e.g. Langfuse → another tracer) is a
  superseding-ADR-sized change because call sites assume the
  decorator/processor seams above.
