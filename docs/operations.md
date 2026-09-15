# Operations

This guide covers runtime operations for the Hamilton-default execution engine:
where evidence and caches live, how to inspect and clear caches, how to list and
verify artifact packages, and how to recover from cache or package loss.

## Engine and evidence model

- **Hamilton is the default execution engine** (`FF_DATAFLOW__ENGINE=hamilton`,
  ADR 0012). Each case runs through `HamiltonLayerExecutor`, which executes four
  separate stage DAGs — Bronze, Silver, Gold, and Platinum — and publishes
  *verified* medallion packages under the configured artifact root (default
  `experiments/artifacts`).
- **Hamilton is the sole execution engine.** The legacy imperative engine and
  the `legacy` artifact-policy branch were removed per ADR 0016. Stale
  `engine=legacy` configuration (env `FF_DATAFLOW__ENGINE` or YAML
  `dataflow.engine`) fails fast with actionable migration guidance — see
  [Migration Guide](migration_guide.md). The operational rollback path is a
  package/version downgrade to the last compatibility release, not a runtime
  engine switch.
- **Verified packages are the authoritative experiment evidence** (plan §2,
  decision 5). A cache hit can accelerate a computation but can never establish
  successful stage completion — only a verified package does.

## Configuration

| Variable | Meaning | Default |
|----------|---------|---------|
| `FF_DATAFLOW__ENGINE` | `hamilton` is the only supported value; any other value (including `legacy`) fails with migration guidance | `hamilton` |
| `FF_DATAFLOW__ARTIFACT_ROOT` | Root for committed medallion packages + control data | `experiments/artifacts` |
| `FF_DATAFLOW__CACHE__ENABLED` | Hamilton node-cache on/off (diagnostic override) | `true` |
| `FF_DATAFLOW__CACHE__PATH` | Explicit Hamilton cache directory | `<artifact_root>/control/cache/hamilton` |
| `FF_DATAFLOW__CACHE__MAX_AGE_DAYS` | Optional automatic maximum cache-entry age | unset |
| `FF_DATAFLOW__CACHE__MAX_SIZE_MB` | Optional automatic serialized-result size cap | unset |
| `FF_DATAFLOW__CACHE__TELEMETRY_MAX_EVENTS` | Maximum node/cache telemetry events retained per Hamilton execution | `10000` |
| `FF_EXECUTION__FAILURE_POLICY` | Outer scheduler failure policy: `continue` / `fail_fast` (ADR 0017) | `continue` |
| `FF_TRACKER__BACKEND` | `none` (default) / `wandb` / `mlflow` | `none` |
| `FF_TRACKER__PROJECT` | Tracker project name | — |
| `FF_TRACKER__ENTITY` | Tracker entity/team | — |

Cache-path precedence is: explicit run override → environment/YAML →
`<resolved artifact_root>/control/cache/hamilton`. The persistent cache is **never
partitioned by attempt ID**, so warm reuse works across attempts.

## Execution failure policy and cancellation

The outer experiment scheduler has a failure policy and a cooperative
cancellation token (ADR 0017, plan 22). All three forms below select the same
behavior; precedence is per-run override → environment/YAML → default.

### Selecting a failure policy

`continue` is the default and preserves historical behavior: every requested
case is scheduled and executed even when earlier cases failed. `fail_fast`
stops scheduling new cases after the first **terminal failed case result**
(feature-level rejections, warnings, cache misses, recovered retries, and
successfully handled partial candidates never trigger it).

```yaml
# config/settings.yaml
execution:
  failure_policy: "fail_fast"   # default: "continue"
```

```bash
# Environment form (wins over YAML)
export FF_EXECUTION__FAILURE_POLICY=fail_fast
```

```python
# Per-run override (wins for this invocation only; recorded in tracker
# provenance; never mutates cached Settings)
from feature_forge import ExperimentalPlatform

platform = ExperimentalPlatform()
results = platform.run(
    datasets=["titanic"], methods=["malmus"], failure_policy="fail_fast"
)
```

Invalid values fail at configuration/argument validation, before any dataset
loading or provider construction.

### Cooperative cancellation token

For operator-driven cancellation (no signals, no globals), construct a
`CancellationToken`, pass it to `run(...)`, and call `cancel()` from another
thread or operator control path:

```python
import threading

from feature_forge.experiment.execution import CancellationToken

token = CancellationToken()
results: list[dict] = []
# reuse the `platform` constructed above

run_thread = threading.Thread(
    target=lambda: results.extend(
        platform.run(datasets=["titanic"], methods=["malmus"],
                     cancellation_token=token)
    )
)
run_thread.start()

token.cancel(reason="operator_request")  # idempotent; first reason wins
run_thread.join()
```

The token is checked before each case starts and between case completions.
It lives in the parent process only: it is never serialized into case
payloads and never enters a Hamilton DAG, sandbox, provider call, or cache
key.

### Fail-fast and cancellation are not hard termination

Both mechanisms are **cooperative at case boundaries**:

- A case already in flight when the stop condition arrives finishes safely
  and keeps its real result. It is never relabelled as cancelled.
- On the process path, submission uses a bounded window of at most
  `max_workers` futures. At the stop point, up to `max_workers` in-flight
  cases may still complete (and be billed). Queued futures that have not
  started are cancelled; the never-submitted tail becomes typed cancelled
  results. There is no hard kill of a running model, sandbox, provider
  request, or stage publication.
- `KeyboardInterrupt` is never swallowed or converted into a failed
  experiment: the scheduler stops new submission, cancels pending futures,
  closes resources safely, journals cancellation for cases proven not to have
  started, and re-raises the interrupt to the caller.
- Cancellation never deletes Hamilton cache entries, LLM cache entries,
  staging directories, or verified packages. A later invocation can resume
  from any verified contiguous prefix produced by cases that started before
  the cancellation.

### Result `state` field

Every requested case yields exactly one result row, in original matrix order,
with an additive `state` field (`succeeded` / `failed` / `cancelled`):

- `succeeded`: the case completed successfully;
- `failed`: the case completed with a terminal failure;
- `cancelled`: the case never started (fail-fast tail, queued-future
  cancellation, pre-cancelled token, or post-interrupt remainder).

Cancelled rows carry no scores (`cv_score`/`gain`/`baseline_score` are
`None`) and no stage packages; their `error` holds a redacted cancellation
record (category, stable error type, and case identity only — no exception
text, prompts, inputs, or secrets). One redacted `case_cancelled` lifecycle
event is journaled per never-started case.

Note for report consumers: cancelled rows populate `error` with the redacted
cancellation message for compatibility, so `Reporter.summary_stats`-style
aggregations that count non-null `error` as failures will include them under
failed runs. Prefer the explicit `state` field when classifying outcomes.

### Retry gap (explicit)

Case-level retry is **not implemented**: the `ResourceConfig.max_attempts`
fields remain dormant, and fail-fast is evaluated only after a terminal
result. A failed case counts as terminal immediately; do not configure
`max_attempts` expecting fail-fast to wait for retries.

## Two independent caches

| Cache | Purpose | Policy | Cleared via |
|-------|---------|--------|-------------|
| LLM `DiskCache` | Mandatory response cache, SHA-256 keys (ADR 0001) | Always on; never bypassed | Provider/LLM client (`DiskCache.clear()`) |
| Hamilton node cache | Disposable acceleration for pure DAG nodes (ADR 0015) | On by default; `source`/`method`/`sandbox`/materialization nodes recompute | `feature-forge cache clear` |

The Hamilton cache is **independent** of the mandatory LLM response cache.
Disabling or clearing one never affects the other.

## Cache operations

The commands preserve the machine-output contract: with `--format json`, stdout
contains exactly one JSON document; diagnostics remain on stderr.

### Status / inspection

```bash
uv run feature-forge cache status
```

Reports cache location, entry count, total/serialized size, and oldest/newest
entry timestamps. Use `cache inspect` for bounded, layer-tagged entry metadata:

```bash
uv run feature-forge cache inspect --layer silver --limit 50
```

Lifecycle cache telemetry records hit/miss/error, duration, serialized bytes,
node, and layer. It is **metadata-only** — it never contains node inputs,
results, prompts, feature data, or secrets (ADR 0015, plan §11).

### Garbage collection

```bash
uv run feature-forge cache gc --max-age-days 30 --max-size-mb 20480
```

Age/size GC of the Hamilton node cache. Configured automatic retention runs in
the parent after all case workers close. Entry age is measured from first cache
creation, not last access. GC is bounded and **never deletes medallion packages
or LLM `DiskCache` entries** — only disposable Hamilton cache nodes. Use it to
reclaim disk without touching experiment evidence.

### Clear

```bash
uv run feature-forge cache clear          # Hamilton node cache only
uv run feature-forge cache clear --force  # required destructive-operation guard
```

**Hamilton cache deletion is safe.** Removing or corrupting the Hamilton cache
falls back to recomputation; verified medallion packages are unaffected and
cases re-run from them (ADR 0015, plan §9.4). Clearing the cache does **not**
delete any experiment evidence.

To clear the separate, mandatory LLM response cache, use the LLM client's
`DiskCache.clear()` — it is a different store and is not touched by the commands
above.

## Artifact operations

### List

```bash
uv run feature-forge artifacts list                 # all committed packages
uv run feature-forge artifacts list --layer gold     # filter by layer
uv run feature-forge artifacts list --case <case_key>
```

Lists committed Bronze/Silver/Gold/Platinum packages with their layer
fingerprint, reuse fingerprint, completion marker, and upstream lineage.

### Verify

```bash
uv run feature-forge artifacts verify
uv run feature-forge artifacts verify --case <case_key>
```

Verifies each committed package's manifest schema version, completion marker,
per-artifact SHA-256 hashes, and declared layer/lineage. Silver, Gold, and
Platinum packages also pass their semantic package loaders. A layer is reusable
only when all required checks verify
(plan §4). Corrupt or incompatible packages stop reuse at that layer; the
remaining suffix recomputes when policy permits rather than skipping to a later
layer.

## Lifecycle journal

Each attempt writes a lifecycle journal under
`<artifact_root>/control/lifecycle` (keyed by its unique `attempt_id`). The
journal records case, stage, node, cache, and terminal events used during
operational inspection. Each worker writes only its own attempt namespace;
state snapshots are atomic.

## Recovery guidance

### Cache loss — safe

Deleting or corrupting the Hamilton cache is non-destructive. The next run
recomputes pure nodes and reuses any still-valid verified packages; materialization
verification still executes even when every upstream value hits (plan §9.4). No
experiment evidence is lost.

### Package deletion — separate and governed

Deleting a verified medallion package is a **separate, governed operator action**,
not a side effect of cache clearing (plan §10). Rules:

- Never silently copy, move, merge, or delete a legacy artifact root.
- Migration of an old `.feature_forge_artifacts` root to the new layout is an
  explicit, operator-requested action that produces its own verification report.
- After deletion, the affected layer (and downstream layers) recompute on the
  next run; upstream verified layers are reused where their lineage still matches.

Prefer `feature-forge artifacts verify` after any manual change to
`<artifact_root>` before re-running experiments.

## Cache timing benchmark

The legacy imperative engine was removed per ADR 0016, so no dual-engine
qualification runs are required or possible. The historical Hamilton/legacy
parity record is preserved under `experiments/legacy_removal/2026-09-14/`
(resolved secret-free settings, per-case metrics, parity deltas, Gold sandbox
decisions, package verification, and cache status); treat it as read-only
evidence and write new benchmark output elsewhere.

For Hamilton cache timing at a representative row count:

```bash
uv run python scripts/qualify_hamilton_cache.py \
  --rows 100000 \
  --cache-dir experiments/cache_benchmark/<date>/cache \
  --output experiments/cache_benchmark/<date>/report.json
```

## Experiment tracking (opt-in)

Tracking defaults to `none` (no external tracker; `NoOpTracker`) so default runs
never require tracker credentials (ADR 0006, R17). Opt in explicitly:

```python
from feature_forge.config import TrackerConfig
from feature_forge.experiment import create_tracker_from_config, WandBTracker

config = TrackerConfig(backend="wandb", project="feature-forge")
tracker = create_tracker_from_config(config)  # or WandBTracker(project="feature-forge")
```

`ExperimentalPlatform.run(tracker=...)` accepts the same instance to override the
configured backend. Tracker side effects are applied exactly once, by the parent
process; workers never initialize a tracker.

## Telemetry hygiene

- Hamilton cache events preserve stage/layer tags and are bound/cleared after
  every run (plan §11).
- Telemetry and the lifecycle journal contain no node values, generated data,
  prompt bodies, or secrets.
