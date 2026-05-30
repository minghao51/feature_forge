# Feature Forge — Current State

Last updated: 2026-05-25

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
| Pydantic-settings config (YAML + env + dotenvx) | Full | N/A |
| Artifact export (memory/disk/hybrid storage) | Full | N/A |
| Artifact schema validation (Pydantic) | Full | N/A |
| Structured logging (structlog + OpenTelemetry) | Full | N/A |
| Langfuse tracing integration | Full | N/A |
| Dataset registry (Kaggle, local, entry points) | Full | N/A |
| Plugin discovery via entry points (methods, agents, datasets, metrics) | Full | N/A |
| Parquet-based IPC for sandbox worker | Full | N/A |
| Async-to-sync bridge (`run_coro_sync`) | Full | N/A |

## Stubbed / Unimplemented

All files below raise `NotImplementedError` or return 501:

- `src/feature_forge/experiment/execution.py:58` — `ExecutionBackend.run()` base class (intentional ABC; concrete `SequentialExecutionAdapter` and `ProcessPoolExecutionAdapter` exist at lines 61–106)
- `src/feature_forge/methods/malmas/pipeline/iterative.py:62` — `BaseIterativePipeline._select_agents()` base class (intentional ABC; `IterativePipeline` override at line 274)
- `src/feature_forge/llm/base.py:123` — `LLMClient._call_api()` (intentional hook; all providers override)
- `src/feature_forge/llm/base.py:127` — `LLMClient._extract_content()` (intentional hook; all providers override)
- `src/feature_forge/llm/base.py:135` — `LLMClient._extract_usage()` (intentional hook; all providers override)

## Known Bugs

| Severity | Issue | Location |
|----------|-------|----------|
| Medium | `NoMemoryStaticRouterPipeline._post_round` is an ellipsis body (`...`), silently skipping all post-round logic including `router.update_performance()` — may cause stale router state in ablation experiments | `src/feature_forge/methods/malmas/pipeline/ablations.py:55` |
| Medium | `MalmusMethod.transform()` reconstructs code from `self._feature_defs` on every call, ignoring the iterative code pipeline used in `fit()` — iterative-mode transform may produce different features than those selected during fit | `src/feature_forge/methods/malmus/method.py:273-280` |
| Low | `BaseMethod.fit_transform()` calls `self.transform(X_train)` instead of returning `pipeline_result["X_train_enhanced"]` — may not include all accumulated features for iterative methods | `src/feature_forge/methods/base.py:78-81` |
| Low | `wandb` and `memory_files/` dirs committed to repo root (contain run artifacts from local experiments) | Repo root: `wandb/`, `memory_files/` |
| Low | `notebooks/memory_files/` duplicates root `memory_files/` — stale copy | `notebooks/memory_files/` |

## Fixed Issues

| Ref | Issue | Fix |
|-----|-------|-----|
| R1 | `complete_json` had no cache/retry support — JSON-mode calls bypassed caching and retry logic, causing redundant API calls on repeated prompts | `LLMClient.complete_json()` now shares cache/retry path with `complete()` via `_do_complete_json` hook in providers (`src/feature_forge/llm/base.py`) |

## Security Concerns

| Severity | Issue | Location |
|----------|-------|----------|
| Medium | Sandbox uses `exec()` in worker process (`src/feature_forge/evaluation/sandbox.py:412`). While AST validation + restricted builtins + process isolation provide defense-in-depth, `exec` of LLM-generated code is inherently risky. Memory limit is best-effort only on macOS. | `src/feature_forge/evaluation/sandbox.py:412` |
| Low | `api_key` property on `LLMClient` returns plaintext `str` from `SecretStr`, allowing accidental logging. | `src/feature_forge/llm/base.py:89-90` |
| Low | Agent memory JSON files persisted to disk unencrypted. Agent memories contain feature strategies and dataset column info that could leak proprietary feature engineering logic. | `src/feature_forge/methods/malmas/memory/persistence.py:23-36` |
| Low | `_sandbox_worker_main` catches `BaseException` at line 430 via `except Exception`, but `MemoryError` and `SystemExit` would still propagate in the parent — timeout is the only reliable containment for runaway allocations on macOS | `src/feature_forge/evaluation/sandbox.py:430` |

## Performance Issues

| Issue | Location |
|-------|----------|
| Serial per-column CV evaluation in iterative methods (CAAFE, LLMFE, Malmus) — each new feature is evaluated one at a time against baseline, causing O(n) full CV passes per iteration | `src/feature_forge/methods/caafe/method.py:148-157`, `src/feature_forge/methods/llmfe/method.py:128-137`, `src/feature_forge/methods/malmus/method.py:222-232` |
| `_column_fingerprint` hashes entire head(256) as strings on every agent `generate()` call — redundant if columns don't change across rounds | `src/feature_forge/methods/malmas/agents/base.py:169-176` |
| `CorePipeline._evaluate_and_select` uses joblib `Parallel` for feature evaluation but copies full `X_train` to each worker — high memory overhead for wide datasets | `src/feature_forge/methods/malmas/pipeline/core.py:554-559` |
| `_baseline_cache_key` recomputes `blake2b` hash of entire `X_train` + `y_train` on every round, even though only column additions change — full content hash is unnecessary | `src/feature_forge/methods/malmas/pipeline/core.py:253-263` |

## Maintenance Issues

| Issue | Detail |
|-------|--------|
| 36 broad `except Exception` handlers across source | Many catch-and-log-all patterns in `src/feature_forge/methods/malmas/pipeline/core.py:73`, `src/feature_forge/evaluation/sandbox.py:381,430`, `src/feature_forge/methods/base.py:152,174`, `src/feature_forge/api.py:74,206`, `src/feature_forge/methods/malmas/agents/base.py:250`, etc. These swallow unexpected errors (e.g., `KeyboardInterrupt` is `BaseException` not `Exception`, but bugs like `KeyError` in internal dicts would be silently logged). |
| `MALMASFeatureEngineer` backward-compat alias in api.py | `src/feature_forge/api.py:316` — kept for backward compat but undocumented and may confuse new users |
| Memory unbounded growth in `AgentMemory` | `src/feature_forge/methods/malmas/memory/base.py` — no `max_size` enforcement despite `MemoryConfig.max_size` existing. Procedural, feedback, and conceptual lists grow without limit across rounds. |
| `Settings` class re-instantiated on every `get_settings()` call | `src/feature_forge/config.py:318-323` — no caching; each call re-reads YAML + env vars. Multiple calls per pipeline run. |
| `structlog` configured globally on first `get_logger()` call | `src/feature_forge/observability/structlog_config.py:95-100` — `cache_logger_on_first_use=True` means logging config cannot be changed at runtime without process restart |
| `MetricRegistry._builtin` is a class var shared across all instances | `src/feature_forge/evaluation/metrics.py:102` — `register()` modifies class-level dict, affecting all users of the registry globally |
| `ProcessPoolExecutionAdapter.run` re-raises as `RuntimeError` on worker failure | `src/feature_forge/experiment/execution.py:106` — loses original exception type, making error handling harder for callers |
| Duplicated `DatasetRegistry` instances in `titanic_loader` / `house_prices_loader` | `src/feature_forge/data/registry.py:156-162` — each call creates a new registry instance and re-discovers entry points |
