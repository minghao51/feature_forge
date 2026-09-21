# Deferred Design and Research

Parking lot for ideas deliberately not pursued yet. Deferral is a decision
with a trigger, not a rejection: each entry records what the idea is, why
now is not the time, and the measured condition that revives it.
Referenced by `AGENTS.md` (decision order 3).

## Entry format

```
### D-N Short title
- Source: plan doc / issue / experiment that produced the idea
- Summary: one or two sentences
- Deferred because: what current evidence says
- Revival trigger: the measurable condition that reopens it
```

## Entries

### D-1 SAGE-inspired memory evolution

- Source: `docs/plan/18_sage_inspired_memory_evolution.md`
- Summary: graph-based agent memory, a self-evolution loop, and
  cross-dataset memory transfer inspired by the SAGE paper. The one
  cheap idea worth taking — top-K relevance retrieval — is already
  implemented (`src/feature_forge/methods/malmas/memory/retrieval.py`);
  what remains deferred is the graph/evolution machinery.
- Deferred because: the current 3-tier memory (procedural / feedback /
  conceptual) plus top-K retrieval is sufficient for running the memory
  ablations (`experiments/memory_ablations/`); at ~50-300 entries per
  run a scored list is functionally equivalent to a graph, so evolution
  machinery adds cost before the ablations tell us where memory actually
  helps.
- Revival trigger: `experiments/memory_ablations/` results showing
  retrieval quality or memory adaptation is the bottleneck, or the
  plan-doc scale thresholds hit: memory beyond ~500 entries, datasets
  beyond ~100 columns, 20+ datasets run, or memory prompt sections
  consistently over 2K tokens.

### D-2 UX improvements (CLI, config wizard, dashboard, feature explorer)

- Source: `docs/plan/19_ux_improvements.md`
- Summary: 10 documented UX gaps (G1-G10) of the library-only package —
  CLI tool, config wizard, progress display, HTML dashboard, feature
  explorer, quick-CSV mode — addressed by 9 proposed improvements
  (U1-U9).
- Deferred because: current users iterate via scripts, notebooks, and the
  sklearn API (the sync `fit()` wrapper exists, but `async_fit` is still
  the exposed async surface and `fit_from_csv`/`suggest_features` are
  unbuilt); UX work does not support a measured experiment and the
  smallest-architecture rule (ADR 0001) applies.
- Revival trigger: a recorded onboarding attempt where time-to-first-
  result exceeds the plan's 30-second `forge init && forge run` target
  (current baseline ~15 minutes), or documented friction from external
  users logged in `REPORT_LOG.md`.

### D-3 MLflow tracking parity

- Source: `docs/plan/08_experiment_tracking.md`
- Summary: WandB is the recommended tracker; MLflow is optional and not
  at feature parity (no native table logging, no sweep equivalent).
  Both are opt-in — the config default backend is `none`.
- Deferred because: no current experiment requires MLflow; WandB covers
  tracking needs. Both backends are reachable from the platform via
  `create_tracker_from_config`, so the remaining gap is parity features,
  not wiring.
- Revival trigger: a collaboration or environment constraint that blocks
  WandB access (e.g., air-gapped or data-sovereignty requirement), or a
  tracking feature WandB cannot provide.

### D-4 Multi-table data ingestion

- Source: `docs/plan/06_data_strategy.md`
- Summary: scale ingestion beyond single-table Kaggle datasets to
  multi-table joins (plan Phase 2: Home Credit 7 tables, IEEE-CIS Fraud
  4 tables, Recruit Restaurant 5 tables with time-series).
- Deferred because: current benchmarks are single-table; multi-table
  ingestion adds schema complexity no experiment measures yet.
- Revival trigger: a benchmark requiring multi-table datasets (Home
  Credit / IEEE-CIS / Recruit are the designed candidates), or
  cross-dataset memory transfer (see D-1) becoming active.

### D-5 Dataset registry growth and ingestion automation

- Source: `docs/plan/06_data_strategy.md` (Phase 1 table, ingestion
  architecture)
- Summary: the plan designs a 5-dataset Phase-1 registry (adding Porto
  Seguro 595K×57, Santander 200K×200, California Housing) plus an
  `OpenMLFetcher`; today only titanic and house_prices are built in
  (`DatasetRegistry.SAMPLE_DATASETS`) and `KaggleFetcher` is the only
  fetcher in `src/feature_forge/data/ingestion.py`.
- Deferred because: the running ablations (`experiments/memory_ablations/`,
  `router_strategies/`, `agent_isolation/`) operate on the two default
  datasets; no experiment has needed broader coverage, and each new
  dataset adds Kaggle-credential and caching surface for zero measured
  benefit.
- Revival trigger: an experiment matrix designed for at least 3 datasets
  (method-ranking claims needing more than 2 datasets for statistical
  power), or a scale claim that requires the large Phase-1 datasets
  (Porto Seguro / Santander row and column counts).

### D-6 Langfuse integration depth

- Source: `docs/plan/07_observability.md` (Layer 2)
- Summary: the plan designs hierarchical traces (experiment → round →
  agent → generation → sandbox tool), versioned prompt management via
  `langfuse.get_prompt`, and per-call cost tracking; implemented is only
  a flat per-completion `trace_generation` decorator (defined in
  `src/feature_forge/observability/langfuse_tracer.py`, applied in
  `src/feature_forge/llm/base.py`) — no round/agent/tool spans, no
  prompt fetch, and no cost tracking anywhere in `src/`.
- Deferred because: enforced DiskCache plus structlog events cover
  current debugging needs; no experiment so far has needed per-agent
  cost attribution or treated prompts as a versioned experimental
  variable.
- Revival trigger: a `REPORT_LOG.md` entry that cannot attribute cost or
  latency to a specific agent/round from existing logs, or a designed
  prompt-version ablation (A/B prompts as an experimental factor).

### D-7 OpenTelemetry trace export and log correlation

- Source: `docs/plan/07_observability.md` (Layer 3)
- Summary: vendor-neutral OTel export linking structlog events, Langfuse
  traces, and tracker metrics. Only `opentelemetry-api` is installed for
  the `add_open_telemetry_spans` processor; no tracer provider or
  exporter is configured by `feature_forge`, so log events carry no
  usable trace/span ids and cannot be correlated with a trace view.
- Deferred because: runs are effectively single-process experiments and
  Langfuse (when enabled) already holds the trace view; wiring an OTel
  SDK/exporter adds configuration surface no current debugging session
  has needed.
- Revival trigger: a debugging session that needs to jump from a trace
  (Langfuse or any OTel backend) to the exact structured log lines of
  the same run, or a self-hosted/air-gapped tracing requirement (pairs
  with D-3).

### D-8 Tracker-native hyperparameter sweeps

- Source: `docs/plan/08_experiment_tracking.md` ("WandB Sweeps for
  Hyperparameter Search")
- Summary: Bayesian/grid sweeps with hyperband early termination over
  `n_rounds`, `llm_temperature`, and `router_strategy`; the platform
  only executes full Cartesian experiment matrices and no sweep code
  exists in `src/`.
- Deferred because: the enforced DiskCache makes repeated matrix cells
  cheap, and no tuning question has required search beyond the explicit
  matrix; sweep infrastructure would be unused until a tuning study is
  actually designed.
- Revival trigger: a tuning study whose full Cartesian grid exceeds the
  LLM-call or wall-clock budget (e.g., more than ~100 uncached
  combinations × rounds of LLM calls), where Bayesian search with early
  termination would fit the budget.

### D-9 LLM response cache garbage collection (RESOLVED 2026-09-06)

- Source: 2026-09-05 security/ops audit — "old entries are never deleted;
  no TTL configured" for `memory_files/llm_cache`.
- Resolution: implemented rather than deferred. Two optional `LLMConfig`
  knobs (both default `None` = historical behavior, preserving the cache
  as a reproducibility artifact):
  - `cache_ttl_days` — per-entry TTL; expired entries are ignored on read
    and physically removed by `DiskCache.maintain()` /
    `scripts/cache_gc.py`. diskcache itself drops writes whose expiry is
    already past.
  - `cache_size_limit_mb` — total-volume cap, enforced **eagerly by
    diskcache on every write** (least-recently-stored eviction; no
    background thread), and re-enforced by `maintain()` when the cap is
    tightened after the fact.
- Policy: GC is manual/CI-invocable (`uv run python scripts/cache_gc.py`,
  supports `--dry-run`, `--ttl-days`, `--size-limit-mb`, `--cache-dir`,
  falling back to `Settings.llm` when flags are omitted). No scheduler,
  no daemons — runs stay single-process experiments.
- Why not always-on: default-forever entries keep exact-run replayability
  (cache hit == the response the experiment actually consumed); operators
  opt into expiry/caps when disk pressure is real.
- Revival trigger (for richer policy): multi-host experiment fleets sharing
  a cache directory, or cache directories exceeding ~10 GiB.

### D-10 Optional browser/interactive catalog (Astro or equivalent)

- Source: `docs/handoffs/2026-07-14-pr8-optional-astro-catalog.md` and
  `docs/handoffs/2026-07-14-pr8-decision-readiness.md` (recovered 2026-09-21
  from `archive/pr-1-medallion-refactor` at `690a764`); branch-only PR8
  evidence corpus remains accessible under that tag
  (`docs/generated/pr8/`, `scripts/*pr8*`, `docs/decisions/pr8-*` paths do
  not exist on main and must not be copied)
- Summary: an optional rebuildable browser/interactive catalog (Astro UI or
  equivalent) over medallion packages, beyond the static MkDocs site and
  generated stage-DAG/reference docs. PR8 gathered decision evidence and a
  synthetic snapshot-schema proposal only; implementation was never
  authorized.
- Deferred because: current static MkDocs site plus generated references are
  adequate for recorded needs; a second serving surface adds hosting,
  security-review, and maintenance obligations with no measured experiment
  demanding interactivity; ADR 0013 keeps the warehouse-free authoritative
  package store as the only required path.
- Revival trigger: at least two demonstrated interactive requirements (e.g.,
  cross-run package comparison with filtering that static docs cannot
  serve), evidence at representative scale, a stable versioned + allowlisted
  public-snapshot schema, and named security, hosting, and maintenance
  owners recorded in a new accepted decision record.
