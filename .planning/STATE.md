# Feature Forge — Current State

Last updated: 2026-09-18

## What's Implemented

| Feature | Backend | Frontend |
|---------|---------|----------|
| sklearn-compatible API (`FeatureForge.fit/transform`) | Full | N/A |
| Multi-agent MALMAS pipeline (6 agents + router + memory) | Full | N/A |
| Agent router (data_driven, performance_driven, hybrid, llm) | Full | N/A |
| 3-tier agent memory (procedural, feedback, conceptual) | Full | N/A |
| Iterative N-round pipeline with feature accumulation | Full | N/A |
| Ablation pipelines (NoMemory, NoRouter, SingleAgent, NoMemoryStaticRouter) | Full | N/A |
| Baseline methods: CAAFE (unified + fidelity) | Full | N/A |
| Baseline methods: LLMFE (single_shot + iterative) | Full | N/A |
| Baseline methods: Malmus (single_shot + iterative) | Full | N/A |
| Baseline methods: OpenFE wrapper | Full | N/A |
| Sandboxed code execution (AST validation + process isolation) | Full | N/A |
| Cross-validation feature evaluation (CVEvaluator) | Full | N/A |
| LLM provider abstraction (OpenAI, DeepSeek, Anthropic, LiteLLM) | Full | N/A |
| Disk-backed LLM response cache (DiskCache) | Full | N/A |
| Retry with exponential backoff | Full | N/A |
| Experiment tracking (WandB, MLflow, NoOp) | Full | N/A |
| Experiment platform (ExperimentalPlatform facade) | Full | N/A |
| Pydantic-settings config (YAML + env + local .env) | Full | N/A |
| Artifact export (ArtifactExporter ABC + DataFrameStorage memory/disk/hybrid) | Full | N/A |
| Artifact schema validation (Pydantic) | Removed — `ArtifactBundle`/`FeatureMetadata`/`ProvenanceRecord`/`IterationRecord`/`ArtifactConfigSchema` were dead (0 production callers) and have been deleted. | N/A |
| Structured logging (structlog + OpenTelemetry) | Full | N/A |
| Langfuse tracing integration | Full | N/A |
| Dataset registry (Kaggle, local, entry points) | Full | N/A |
| Plugin discovery via entry points (methods, agents, datasets, metrics) | Full | N/A |
| Parquet-based IPC for sandbox worker | Full | N/A |
| Async-to-sync bridge (`run_coro_sync`) | Full | N/A |

## Stubbed / Unimplemented

All files below raise `NotImplementedError` or return 501:

- `src/feature_forge/methods/malmas/pipeline/iterative.py:62` — `BaseIterativePipeline._select_agents()` base class (intentional ABC; `IterativePipeline` override at line 274)
- `src/feature_forge/llm/base.py:123` — `LLMClient._call_api()` (intentional hook; all providers override)
- `src/feature_forge/llm/base.py:127` — `LLMClient._extract_content()` (intentional hook; all providers override)
- `src/feature_forge/llm/base.py:135` — `LLMClient._extract_usage()` (intentional hook; all providers override)

## Known Bugs

The 2026-09-14 independent audit opened plan 23. Confirmed gaps are: disabled
discovery/evaluation holdout in the Hamilton executor, whole-frame Platinum
preprocessing, selection/reporting on shared evidence, dormant selection policy
fields, incorrect configurable/directional uncertainty, host-filesystem access
through allowed sandbox libraries, unbounded sandbox timeout cleanup, and an
incomplete LLM cache identity. See
`docs/plan/23_evaluation_integrity_security_hardening.md`. **PR 1 landed
2026-09-15**: every finding now has a deterministic characterization test
(37 `xfail(strict=True)` + 3 pins across the three `tests/unit/test_plan23_*`
modules), the OS-isolation spike proved Landlock ABI 7 containment on this
Linux host (WSL2, kernel 6.18 — see
`docs/spikes/2026-09-15-sandbox-os-isolation.md`), and ADRs 0018–0020 were
**accepted by the maintainer on 2026-09-15**, unlocking runtime fixes (PRs
2–7). **PR 2 landed 2026-09-16** (partition-aware discovery per ADR 0018
decisions 1–3): typed evaluation-protocol settings, fail-closed holdout
partitioning, discovery-only method fitting, the row-local scope contract,
and protocol-aware reuse fingerprints. **PR 3 landed 2026-09-16** (fold-local
preprocessing and selection per ADR 0018 decisions 4–6): per-fold train-only
imputation/encoding with the documented -1 unknown-category sentinel
(`evaluation/preprocessing.py`), discovery-fold candidate and selection
evidence, greedy forward selection enforcing `minimum_practical_gain`,
`require_positive_lower_bound` (1.96 normal margin until PR 4 swaps in
Student-t), `max_selected_features`, and the stable name tie-break, two-arm
evaluation-fold reporting where rejected winners mirror baseline, and
`PlatinumRequest` selection-partition/profile validation. The remaining
Platinum-side gap is directional Student-t intervals plus evidence schema v2
(PR 4); sandbox containment is PR 5 and LLM cache identity is PR 6. **PR 4
landed 2026-09-18** (Platinum v2 evidence and uncertainty per ADR 0018
decisions 7–8): directional Student-t intervals with scipy as a direct
dependency (`_t_critical` owns the margin in `uncertainty_summary` and the
greedy gate), the 12-artifact evidence schema v2 (`evidence.json`
`PlatinumEvidenceIndex` marker, `discovery_fold_metrics.parquet`,
`selection_steps.json`, `preprocessing.json`) with offline reconstruction
and tamper detection in `load_platinum_package`, Platinum-only
`PLATINUM_IDENTITY_SCHEMA_VERSION="2"` reuse rejection (v1 packages stay
readable and integrity-verified but never reusable), and directional
gain/bounds + `evaluation_protocol`/`selection_biased` labeling in results
and tracker output. PR 5 landed 2026-09-18 (sandbox containment and bounded
worker lifecycle per ADR 0019): expanded AST I/O policy blocking direct and
aliased NumPy/Pandas file APIs, raw-ctypes Landlock strict containment
(exact inode grants, empty-allow TCP at ABI >= 4, seccomp fallback,
fail-closed when unavailable) with an explicit recorded
`degraded_development` profile, and a bounded sync/async worker lifecycle
(event-loop-owned spawn, single monotonic deadline, process-group kill,
leak-free cleanup); reviewer pass incorporated. Next change: PR 6 (LLM
cache identity v2, ADR 0020).

| Ref | Issue | Resolution |
|-----|-------|------------|
| R12 | `NoMemoryStaticRouterPipeline._post_round` was flagged as a possible bug (ellipsis body skipping `router.update_performance()`) | **Not a bug — intentional ablation.** Body is `pass` (not `...`); class docstring documents "keeps router performance state fixed across rounds"; `test_no_memory_static_router_does_not_update_router_performance` (`tests/integration/test_pipeline.py:500`) asserts router performance is unchanged across rounds |
| R13 | The per-column keep/gain loop and the iterative-fit tail (set state + cache `X_train_enhanced`) were verbatim-duplicated across `caafe`/`llmfe`/`malmus` (~25 LOC × 3), so a fix in one could drift from the others | Extracted `_evaluate_and_select` and `_finalize_iterative_fit` onto `BaseMethod` (alongside the existing `_transform_via_iteration_codes`); the three iterative methods now call them, removing ~62 LOC of duplication (`src/feature_forge/methods/base.py`) |
| R14 | `core.py` was a 666-LOC god-class mixing code-gen, AST validation, and orchestration; the import denylist (`_BANNED_IMPORTS`, local to `_validate_code_ast`) could drift from the sandbox's `ALLOWED_IMPORTS` allowlist | Moved `CodeGenerator` + `_validate_code_ast` to new `methods/malmas/pipeline/codegen.py` (core.py now 524 LOC); promoted `BANNED_IMPORTS` to a module-level constant in `evaluation/sandbox.py` as the single source, imported by `codegen.py` (`src/feature_forge/methods/malmas/pipeline/codegen.py`) |
| R15 | Only 2 of 5 registries (`Metric`, `Model`) used the shared `discover_entry_points` helper; `AgentRegistry.discover` had **no** error handling (a broken entry point would crash the registry), and `Method`/`Dataset` reinvented the load+warn+dedup loop inline | All 5 registries now route loading through `discover_entry_points`: `AgentRegistry` gains error handling + dedup warnings, `MethodRegistry` keeps its type + `MethodProtocol` validation as a post-pass, `DatasetRegistry` keeps its loader/metadata bookkeeping; public method names left as-is (renaming would be API-breaking) (`src/feature_forge/{methods,methods/malmas/agents,data}/...`) |
| R16 | `DeepSeekProvider._extract_content`/`_extract_usage` were byte-identical to the inherited `OpenAIProvider` versions — pure noise that could mask the real divergence point (`_call_api` thinking mode) | Deleted both overrides; `DeepSeekProvider` now genuinely differs from `OpenAIProvider` only in `_call_api` (thinking mode), `provider_name`, and constructor defaults (`src/feature_forge/llm/providers/deepseek.py`) |
| R17 | After R11 wired `create_tracker_from_config`, the default `TrackerConfig.backend = "wandb"` (also in `settings.yaml`) became live — any user on default config would hit `wandb.init` at run time and fail if wandb is unconfigured, where runs previously no-op'd silently | Changed default backend to `"none"` in both `TrackerConfig` (`config.py`) and `settings.yaml`; library is now safe-by-default and users opt into wandb/mlflow explicitly. The real factory path is now exercised by e2e tests (the previous monkeypatch stub was removed) |

## Fixed Issues

| Ref | Issue | Fix |
|-----|-------|-----|
| R1 | `complete_json` had no cache/retry support — JSON-mode calls bypassed caching and retry logic, causing redundant API calls on repeated prompts | `LLMClient.complete_json()` now shares cache/retry path with `complete()` via `_do_complete_json` hook in providers (`src/feature_forge/llm/base.py`) |
| R2 | `BaseMethod.fit_transform()` re-ran `transform` on the training set instead of reusing the enhanced frame, and iterative methods (CAAFE/LLMFE/Malmus) leaked discarded candidate features into transform output | `fit_transform()` now returns cached `pipeline_result["X_train_enhanced"]`; methods track `_kept_features` and `_transform_via_iteration_codes` filters to selected columns only |
| R3 | `MalmusMethod.transform()` reconstructed code from `self._feature_defs` on every call, ignoring which features were selected during fit | Malmus now regenerates per-iteration code from kept `FeatureDefinition`s only; iterative transform uses the stored code pipeline |
| R4 | `LLMClient.api_key` property returned plaintext `SecretStr` value, enabling accidental logging | Replaced with explicit `get_api_key()` method; providers + tests updated |
| R5 | `_sandbox_worker_main` caught only `Exception`, so `MemoryError`/`SystemExit` escaped containment | Widened to `except BaseException` in the worker |
| R6 | `get_settings()` re-read YAML + env on every call (no caching) | Now backed by `functools.lru_cache(maxsize=1)` via `_get_base_settings()` |
| R7 | `_column_fingerprint` and per-column stats recomputed on every agent `generate()` call | Added `_fingerprint_cache` and `_column_stats_cache` ClassVars on `BaseFeatureAgent` |
| R8 | `_baseline_cache_key` previously used an incomplete/object-identity cache key | Uses a canonical fingerprint of complete X/y content and schema plus folds, metric, task, random state, and model identity; in-place mutation invalidates the baseline |
| R9 | Loky large-matrix guard warned but did not switch backends, risking OOM | Now falls back to the `threading` backend when the matrix is too large |
| R10 | `DiskCache` was fully implemented but never instantiated — `create_llm_client` always passed `cache=None`, so the cache hit/miss branches in `LLMClient` were unreachable and repeated LLM calls re-hit the API | `create_llm_client` now auto-instantiates `DiskCache()` when `cache` is unset and `LLMConfig.cache_responses` is true (default ON); explicit `cache=` still overrides (`src/feature_forge/llm/factory.py`) |
| R11 | `TrackerConfig.backend` (`wandb`/`mlflow`/`none`) was never consulted — `ExperimentalPlatform.run()` hardcoded `NoOpTracker`, so MLflow and WandB were unreachable from the platform despite being advertised | New `create_tracker_from_config` maps config → backend (`src/feature_forge/experiment/factory.py`); `ExperimentalPlatform.run()` now consults it when no explicit `tracker=` is passed (`src/feature_forge/platform.py`) |

## Security Concerns

| Severity | Issue | Location |
|----------|-------|----------|
| Medium | Sandbox uses `exec()` in worker process (`src/feature_forge/evaluation/sandbox.py:412`). While AST validation + restricted builtins + process isolation provide defense-in-depth, `exec` of LLM-generated code is inherently risky. Memory limit is best-effort only on macOS. | `src/feature_forge/evaluation/sandbox.py:412` |
| Low | ~~`api_key` property on `LLMClient` returns plaintext `str` from `SecretStr`, allowing accidental logging.~~ **Fixed (R4):** replaced with `get_api_key()` method. | `src/feature_forge/llm/base.py` |
| Low | Agent memory JSON files persisted to disk unencrypted. Agent memories contain feature strategies and dataset column info that could leak proprietary feature engineering logic. | `src/feature_forge/methods/malmas/memory/persistence.py:23-36` |
| Low | ~~`_sandbox_worker_main` catches `BaseException` at line 430 via `except Exception`...~~ **Fixed (R5):** worker now catches `BaseException`. Timeout remains the only reliable containment for runaway allocations on macOS. | `src/feature_forge/evaluation/sandbox.py:430` |

## Performance Issues

| Issue | Location |
|-------|----------|
| ~~Hamilton's tiny-fixture warm run was slower than cold~~ **Qualified:** a 100,000-row direct Bronze/Silver run measured 1,400.1 ms cold vs 327.0 ms warm with 10/10 eligible hits. Continue treating performance as workload-specific rather than a universal speedup claim. | `experiments/legacy_removal/2026-09-14/cache_benchmark/report.json` |
| Serial per-column CV evaluation in iterative methods (CAAFE, LLMFE, Malmus) — each new feature is evaluated one at a time against baseline, causing O(n) full CV passes per iteration | `src/feature_forge/methods/caafe/method.py:148-157`, `src/feature_forge/methods/llmfe/method.py:128-137`, `src/feature_forge/methods/malmus/method.py:222-232` |
| ~~`_column_fingerprint` hashes entire head(256) as strings on every agent `generate()` call~~ **Fixed (R7):** fingerprint + column stats now cached on `BaseFeatureAgent`. | `src/feature_forge/methods/malmas/agents/base.py` |
| `CorePipeline._evaluate_and_select` uses joblib `Parallel` for feature evaluation but copies full `X_train` to each worker — high memory overhead for wide datasets | `src/feature_forge/methods/malmas/pipeline/core.py:554-559` |
| `_baseline_cache_key` fingerprints complete `X_train`/`y_train` content and evaluation identity | Required for mutation-safe correctness; benchmark representative iterative workloads before introducing any safe memoization. | `src/feature_forge/methods/malmas/pipeline/core.py` |

## Maintenance Issues

| Issue | Detail |
|-------|--------|
| 36 broad `except Exception` handlers across source | Many catch-and-log-all patterns in `src/feature_forge/methods/malmas/pipeline/core.py:73`, `src/feature_forge/evaluation/sandbox.py:381,430`, `src/feature_forge/methods/base.py:152,174`, `src/feature_forge/api.py:74,206`, `src/feature_forge/methods/malmas/agents/base.py:250`, etc. These swallow unexpected errors (e.g., `KeyboardInterrupt` is `BaseException` not `Exception`, but bugs like `KeyError` in internal dicts would be silently logged). |
| ~~Memory unbounded growth in `AgentMemory`~~ | **Fixed:** `_enforce_limits()` / `_trim_sequence` cap each tier at `MemoryConfig.max_size` on every record, load, and save (`src/feature_forge/methods/malmas/memory/base.py`); wired from `config.memory.max_size` in the iterative pipeline. Recorded in [ADR 0002](../docs/decisions/0002-three-tier-agent-memory.md). |
| `structlog` configured globally on first `get_logger()` call | `src/feature_forge/observability/structlog_config.py:95-100` — `cache_logger_on_first_use=True` means logging config cannot be changed at runtime without process restart |
| `MetricRegistry._builtin` is a class var shared across all instances | `src/feature_forge/evaluation/metrics.py:102` — `register()` modifies class-level dict, affecting all users of the registry globally. `MetricRegistry.reset()` now restores builtin defaults for test isolation. |
| `ProcessPoolExecutionAdapter.run` re-raises as `RuntimeError` on worker failure | `src/feature_forge/experiment/execution.py:106` — loses original exception type, making error handling harder for callers |
| Duplicated `DatasetRegistry` instances in `titanic_loader` / `house_prices_loader` | `src/feature_forge/data/registry.py:156-162` — each call creates a new registry instance and re-discovers entry points |
| Hamilton cancellation/fail-fast implementation | **Implemented (plan 22 PRs 1–4, ADR 0017, 2026-09-14):** continue/fail-fast failure policy (`FF_EXECUTION__FAILURE_POLICY`), per-run overrides, cooperative `CancellationToken` at case boundaries, typed `state` results, bounded process submission, and redacted `case_cancelled` lifecycle events. Operator docs: `docs/operations.md` ("Execution failure policy and cancellation"). Case-level retry (`ResourceConfig.max_attempts`) remains dormant/out of scope. |

## Hamilton operations and documentation (PR 5)

The post-PR 4 platform now includes:

- `ExperimentMatrix` / `ExperimentRunner` examples removed from `README.md`,
  `docs/index.md`, `docs/quick_start.md`, and `docs/api_reference.md`; all point
  to `ExperimentalPlatform.run()` (cartesian expansion is built in — ADR 0006).
- Tracker default corrected to `none`/opt-in across `README.md`, `docs/index.md`,
  `.planning/OVERVIEW.md`, and the `tracker.py` module docstring (ADR 0006; R17).
- Hamilton documented as the execution engine (`FF_DATAFLOW__ENGINE`); the
  `legacy` compatibility-release rollback path documented at the time was later
  removed by the ADR 0016 removal change.
- Bounded redacted Hamilton node/cache telemetry with layer, outcome, duration,
  and serialized-byte metadata; concurrent WAL/atomic cache stores; configurable
  age/size retention; and safe cache status / inspect / GC / clear commands.
- Medallion artifact list / semantic verify commands; Hamilton GC never touches
  artifact packages or the independent mandatory LLM DiskCache.
- New `docs/operations.md` states that Hamilton cache deletion is safe while
  verified package deletion is a separate, governed operator action (plan §10).
- Freshness-tested generated stage DAGs are published via `mkdocs.yml`; strict
  MkDocs builds pass.

## Legacy correctness and dependency cleanup (PR 6)

- MALMAS baseline caching now fingerprints complete X/y content plus fold,
  metric, task, seed, and model identity; mutation cannot reuse a stale score.
- Unique successful code batches execute once per train/test partition. Feature
  output names, indexes, and dtypes are checked, with per-feature failures
  retained in artifacts; minimization metrics use direction-aware selection.
- `max_selected_features` replaces the misleading `min_effective` setting name
  (the old constructor input remains a migration alias). Unknown modes/agents
  fail before provider construction. Normal fits clear router/memory learning;
  `warm_start=True` is the explicit persistence opt-in.
- `random_forest` is the core default. OpenFE, CAAFE fidelity, XGBoost,
  LightGBM, and CatBoost are named extras with precise install guidance. A CI
  lane builds and exercises a clean standard wheel.
- Full standard+Intel suite: 922 passed, 9 optional-dependency skips. A newly
  exposed concurrent SQLite WAL initialization race was fixed and passed five
  repeated spawned-worker runs.

## Legacy-removal technical qualification

- Plan 21 gates 1–5 are evidenced. Python 3.11, 3.12, and 3.13 each completed
  the full suite with 954 passed and 9 expected optional-XGBoost skips.
- Real-process recomputation matches sequential execution under Linux-default
  and explicit spawn contexts. Hamilton/legacy parity uses absolute tolerance
  `1e-9` and passed on the two-seed bundled breast-cancer matrix.
- Cache deletion/corrupt-metadata recovery, tampered-package suffix recompute,
  same-attempt resume, and provider-free Gold replay pass end to end.
- Persistent evidence: `experiments/legacy_removal/2026-09-14/`.
- Gate 6 is accepted: ADR 0016 supersedes ADR 0012, and the dedicated removal
  change has landed (2026-09-14). Hamilton is the sole execution engine; stale
  legacy configuration fails fast with migration guidance, and the parity
  evidence remains preserved at `experiments/legacy_removal/2026-09-14/`.

## Active execution-runtime sequence

1. ✅ Implement ADR 0016 legacy removal, including stale-config migration
   errors — landed 2026-09-14. Hamilton is the sole engine; evidence reports
   are preserved and rollback is a release downgrade, not a fallback.
2. ✅ Implement `docs/plan/22_fail_fast_cancellation_contract.md` — PRs 1–4
   landed (typed contracts, sequential fail-fast, bounded process scheduling,
   operator documentation + completion audit). Final post-audit validation:
   full suite 1000 passed / 9 skipped (expected optional-XGBoost skips),
   `mypy src`+`mypy tests`, ruff, format, hygiene, docs references, mkdocs
   `git diff --check` all green, repeated on Python 3.11, 3.12, and 3.13
   (1000 passed / 9 expected skips each). Default behavior remains
   continue-on-error; fail-fast and cancellation are cooperative at case
   boundaries. Operator docs:
   `docs/operations.md`.
3. Keep engine-removal and scheduling changes separate. The dormant case-retry
   fields (`ResourceConfig.max_attempts`) remain an explicit out-of-scope gap
   and must not be reported as implemented.
