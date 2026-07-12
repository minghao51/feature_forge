# Feature Forge — Current State

Last updated: 2026-07-11

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

- `src/feature_forge/experiment/execution.py` — `ExecutionBackend` is now an `ABC` with `@abstractmethod run()` (no longer raises `NotImplementedError`). Concrete `SequentialExecutionAdapter` and `ProcessPoolExecutionAdapter` implement it.
- `src/feature_forge/methods/malmas/pipeline/iterative.py:62` — `BaseIterativePipeline._select_agents()` base class (intentional ABC; `IterativePipeline` override at line 274)
- `src/feature_forge/llm/base.py:123` — `LLMClient._call_api()` (intentional hook; all providers override)
- `src/feature_forge/llm/base.py:127` — `LLMClient._extract_content()` (intentional hook; all providers override)
- `src/feature_forge/llm/base.py:135` — `LLMClient._extract_usage()` (intentional hook; all providers override)

## Known Bugs

None currently open.

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
| R8 | `_baseline_cache_key` hashed full `y_train` values each round | Uses `id(y_train)` instead |
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
| Serial per-column CV evaluation in iterative methods (CAAFE, LLMFE, Malmus) — each new feature is evaluated one at a time against baseline, causing O(n) full CV passes per iteration | `src/feature_forge/methods/caafe/method.py:148-157`, `src/feature_forge/methods/llmfe/method.py:128-137`, `src/feature_forge/methods/malmus/method.py:222-232` |
| ~~`_column_fingerprint` hashes entire head(256) as strings on every agent `generate()` call~~ **Fixed (R7):** fingerprint + column stats now cached on `BaseFeatureAgent`. | `src/feature_forge/methods/malmas/agents/base.py` |
| `CorePipeline._evaluate_and_select` uses joblib `Parallel` for feature evaluation but copies full `X_train` to each worker — high memory overhead for wide datasets | `src/feature_forge/methods/malmas/pipeline/core.py:554-559` |
| ~~`_baseline_cache_key` recomputes `blake2b` hash of entire `X_train` + `y_train` on every round~~ **Fixed (R8):** uses `id(y_train)`. | `src/feature_forge/methods/malmas/pipeline/core.py` |

## Maintenance Issues

| Issue | Detail |
|-------|--------|
| 36 broad `except Exception` handlers across source | Many catch-and-log-all patterns in `src/feature_forge/methods/malmas/pipeline/core.py:73`, `src/feature_forge/evaluation/sandbox.py:381,430`, `src/feature_forge/methods/base.py:152,174`, `src/feature_forge/api.py:74,206`, `src/feature_forge/methods/malmas/agents/base.py:250`, etc. These swallow unexpected errors (e.g., `KeyboardInterrupt` is `BaseException` not `Exception`, but bugs like `KeyError` in internal dicts would be silently logged). |
| Memory unbounded growth in `AgentMemory` | `src/feature_forge/methods/malmas/memory/base.py` — no `max_size` enforcement despite `MemoryConfig.max_size` existing. Procedural, feedback, and conceptual lists grow without limit across rounds. |
| `structlog` configured globally on first `get_logger()` call | `src/feature_forge/observability/structlog_config.py:95-100` — `cache_logger_on_first_use=True` means logging config cannot be changed at runtime without process restart |
| `MetricRegistry._builtin` is a class var shared across all instances | `src/feature_forge/evaluation/metrics.py:102` — `register()` modifies class-level dict, affecting all users of the registry globally. `MetricRegistry.reset()` now restores builtin defaults for test isolation. |
| `ProcessPoolExecutionAdapter.run` re-raises as `RuntimeError` on worker failure | `src/feature_forge/experiment/execution.py:106` — loses original exception type, making error handling harder for callers |
| Duplicated `DatasetRegistry` instances in `titanic_loader` / `house_prices_loader` | `src/feature_forge/data/registry.py:156-162` — each call creates a new registry instance and re-discovers entry points |
