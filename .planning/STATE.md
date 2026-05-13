# Feature Forge — Current State

Last updated: 2026-05-12

## What's Implemented

| Module / Area | Status | Notes |
|---|---|---|
| **Core Pipeline** (`pipeline/core.py`) | Full | Single-round: agent specs → code gen → sandbox → CV eval → top-k selection |
| **Iterative Pipeline** (`pipeline/iterative.py`) | Full | Multi-round with memory + router integration |
| **Ablation Pipelines** (`pipeline/ablations.py`) | Full | `NoMemoryPipeline`, `NoRouterPipeline`, `SingleAgentPipeline` |
| **Sklearn API** (`api.py`) | Full | `FeatureForge` (fit/transform/get_feature_names_out), `MALMASFeatureEngineer` alias |
| **Agent Framework** (`agents/base.py`) | Full | ABC + `BaseFeatureAgent` with LLM interaction pattern, `AgentRegistry` with entry points |
| **6 Feature Agents** (`agents/*.py`) | Full | unary, cross_compositional, aggregation, temporal, local_transform, local_pattern |
| **Router Agent** (`agents/router.py`) | Full | data_driven, performance_driven, hybrid, LLM-based selection strategies |
| **LLM Client ABC** (`llm/base.py`) | Full | Retry, JSON mode injection, timing, structured `LLMResponse` |
| **OpenAI Provider** (`llm/providers/openai.py`) | Full | Async OpenAI client with JSON mode |
| **DeepSeek Provider** (`llm/providers/deepseek.py`) | Full | Inherits OpenAI provider, default base_url |
| **Anthropic Provider** (`llm/providers/anthropic.py`) | Full | System message extraction, usage mapping |
| **LiteLLM Provider** (`llm/providers/litellm_provider.py`) | Full | 100+ provider unified interface |
| **LLM Factory** (`llm/factory.py`) | Full | Auto-inference from model name, lazy imports |
| **LLM Cache** (`llm/cache.py`) | Full | Disk-backed (diskcache/SQLite), SHA-256 keys |
| **LLM Retry** (`llm/retry.py`) | Full | Tenacity-based exponential backoff with jitter |
| **Langfuse Wrapper** (`llm/langfuse_wrapper.py`) | Full | Cache + tracing decorator on any LLMClient |
| **Sandbox Executor** (`evaluation/sandbox.py`) | Full | AST validation, spawn-isolated worker, timeout + memory limits |
| **CV Evaluator** (`evaluation/cv.py`) | Full | K-fold with StratifiedKFold for classification, preprocessing |
| **Metrics** (`evaluation/metrics.py`) | Full | AUC, ACC, F1, RMSE, MAE, R2, NRMSE |
| **Model Factory** (`evaluation/model_factory.py`) | Full | XGBoost, LightGBM, CatBoost, RandomForest, MLP |
| **Memory System** (`memory/base.py`, `persistence.py`) | Full | Procedural + feedback + conceptual tiers, JSON persistence |
| **Conceptual Memory** (`memory/conceptual.py`) | Full | LLM-summarized rules per-agent and global |
| **Config** (`config.py`) | Full | pydantic-settings with YAML + env + constructor priority |
| **Experiment Tracker** (`experiment/tracker.py`) | Full | ABC + `NoOpTracker` |
| **WandB Backend** (`experiment/wandb_backend.py`) | Full | Dual code logging (artifact + table) |
| **MLflow Backend** (`experiment/mlflow_backend.py`) | Full | Full implementation with temp file handling |
| **Experiment Matrix** (`experiment/matrix.py`) | Full | Cartesian product parameter grid builder |
| **Experiment Runner** (`experiment/runner.py`) | Full | Sequential + parallel (ProcessPoolExecutor) |
| **Reporter** (`experiment/reporter.py`) | Full | Markdown/HTML export, best-per-group selection |
| **Artifact System** (`artifacts/base.py`, `storage.py`, `schema.py`) | Full | ABC, lazy refs, memory/disk/hybrid modes, Pydantic validation |
| **Artifact Diff** (`artifacts/diff.py`) | Full | Cross-method overlap matrix, gain comparison |
| **Artifact Dashboard** (`artifacts/dashboard.py`) | Full | Self-contained HTML report generation |
| **Method Comparison** (`artifacts/comparison.py`) | Full | Unified runner for all methods |
| **CAAFE Baseline** (`baselines/caafe.py`) | Full | unified + fidelity variants, iterative with CV feedback |
| **OpenFE Baseline** (`baselines/openfe.py`) | Full | Wrapper with best-effort artifact extraction |
| **LLMFE Baseline** (`baselines/llmfe.py`) | Full | single_shot + iterative modes |
| **Malmus Baseline** (`baselines/malmus.py`) | Full | Structured JSON mode, single_shot + iterative |
| **Data Ingestion** (`data/ingestion.py`) | Full | Kaggle fetcher with caching |
| **Dataset Registry** (`data/registry.py`) | Full | Local + Kaggle, auto-discovery |
| **Observability** (`observability/`) | Full | structlog (pretty/JSON), OpenTelemetry span injection, Langfuse tracing |
| **Exception Hierarchy** (`exceptions.py`) | Full | 12 domain-specific exception types |
| **Types** (`types.py`) | Full | NewTypes for domain strings, FeatureSpec, type variables |
| **Utils** (`utils.py`) | Full | `run_coro_sync` (async bridge), `strip_markdown_fences` |
| **Prompt Templates** (`prompts/*.txt`) | Full | 7 prompt files (unary, cross_compositional, aggregation, temporal, local_transform, local_pattern, router, code_generation) |
| **Notebooks** (`notebooks/01-10`) | Full | 10 marimo notebooks covering quickstart through method comparison |

## Stubbed / Unimplemented

Abstract methods intentionally left for subclass override (not bugs):

- `src/feature_forge/llm/base.py:96` — `LLMClient._call_api()` raises `NotImplementedError` (must override in providers)
- `src/feature_forge/llm/base.py:100` — `LLMClient._extract_content()` raises `NotImplementedError`
- `src/feature_forge/llm/base.py:104` — `LLMClient._extract_usage()` raises `NotImplementedError`
- `src/feature_forge/pipeline/iterative.py:61` — `BaseIterativePipeline._select_agents()` raises `NotImplementedError` (must override in subclasses)

No-op pass statements in `NoOpTracker` (by design):

- `src/feature_forge/experiment/tracker.py:95-113` — All 7 methods are `pass` (intentional no-op tracker)

Silent pass in error handlers:

- `src/feature_forge/evaluation/sandbox.py:172-174` — `FileNotFoundError`/`OSError` silently passed during temp file cleanup (acceptable — cleanup is best-effort)
- `src/feature_forge/agents/router.py:208` — LLM JSON parse failure silently falls back to hybrid (acceptable — designed fallback)
- `src/feature_forge/experiment/wandb_backend.py:82,113` — `ImportError` silently passed (acceptable — graceful no-op when wandb unavailable)

## Known Bugs

| Severity | Issue | Location |
|---|---|---|
| Medium | `FeatureSpec` is `dict[str, Any]` — no schema validation on agent outputs. Malformed specs silently pass through with `name="unknown"` defaults | `src/feature_forge/types.py:23` |
| Medium | `BaseFeatureAgent.generate()` at `agents/base.py:161` calls `self._parse_response()` which strips markdown fences then tries `json.loads()` — but the LLM is prompted with `complete()` not `complete_json()`, so parsing is fragile | `src/feature_forge/agents/base.py:190` |
| Medium | `cv.py:_cv_score` catches `Exception` at line 135 but re-raises as `EvaluationError` — however it loses the original traceback's frame information | `src/feature_forge/evaluation/cv.py:135` |
| Low | `ExperimentRunner.run_parallel()` at `runner.py:70` does not initialize/finish tracker per-run (tracker is only used in sequential `run()`) | `src/feature_forge/experiment/runner.py:70` |
| Low | `ArtifactConfig.__post_init__` only creates `storage_dir` for non-memory modes — if config is later changed to disk mode, directory may not exist | `src/feature_forge/artifacts/base.py:48` |

## Security Concerns

| Severity | Issue | Location |
|---|---|---|
| **High** | `LiteLLMProvider._setup_env()` sets `os.environ["API_KEY"]` (generic key name) and arbitrary `provider_env_vars` into process environment. This pollutes the global env and could leak to child processes | `src/feature_forge/llm/providers/litellm_provider.py:55-61` |
| **High** | Sandbox allows `Exception` and `NotImplementedError` in `ALLOWED_BUILTINS` — while blocked by AST for direct calls, `Exception.__init__` could theoretically be used to construct objects that bypass checks | `src/feature_forge/evaluation/sandbox.py:96-97` |
| Medium | `DeepSeekProvider.__init__()` reads `DEEPSEEK_API_KEY` from env directly instead of using `SecretStr` — key may appear in logs/tracebacks | `src/feature_forge/llm/providers/deepseek.py:40` |
| Medium | 26 broad `except Exception` clauses across the codebase swallow error context. Most re-raise or log, but some silently continue (see Stubbed section) | Multiple files (see grep results) |
| Medium | `MemoryPersistence.save()` writes JSON without atomic rename — crash mid-write corrupts the file | `src/feature_forge/memory/persistence.py:22` |
| Low | `_sandbox_worker_main` uses `exec()` on AST-validated code — AST validation is good but not a complete sandbox (e.g., attribute traversal `"".__class__.__mro__` could bypass if not caught by dunder prefix check) | `src/feature_forge/evaluation/sandbox.py:258` |
| Low | `LangfuseLLMWrapper` stores raw `api_key` from inner client — accessible via `client.api_key` | `src/feature_forge/llm/langfuse_wrapper.py:35` |

## Performance Issues

| Issue | Location |
|---|---|
| Sequential per-feature CV evaluation — each candidate column is evaluated independently with a full K-fold CV, meaning O(n_candidates × n_folds) model fits per round | `src/feature_forge/pipeline/core.py:204-217` |
| `_infer_column_descriptions()` computes statistics (mean, std, min, max, mode) for every column on every agent call — no caching across agents in the same round | `src/feature_forge/agents/base.py:139-159` |
| Sandbox spawns a new process per code execution via `mp.Process` — process creation overhead is ~50-100ms per call. Multiple sandbox calls per round (train + test) | `src/feature_forge/evaluation/sandbox.py:125-137` |
| `evaluate_feature()` in CV evaluates one feature at a time — could batch features for efficiency | `src/feature_forge/evaluation/cv.py:64-99` |
| `LangfuseLLMWrapper.complete()` computes SHA-256 cache key on every call even when cache is disabled | `src/feature_forge/llm/langfuse_wrapper.py:76` |
| Memory JSON persistence is synchronous file I/O — `save()` blocks on every round for every agent | `src/feature_forge/memory/persistence.py:22` |
| `build_cache_key()` serializes full message list to JSON + SHA-256 on every LLM call in base client (duplicate logic with `DiskCache.get_key`) | `src/feature_forge/llm/base.py:290-308`, `src/feature_forge/llm/cache.py:50-69` |

## Maintenance Issues

| Issue | Detail |
|---|---|
| **Duplicate cache key logic** | `LLMClient.build_cache_key()` at `llm/base.py:290` and `DiskCache.get_key()` at `llm/cache.py:50` compute identical SHA-256 hashes with the same payload structure. Changes to one must be mirrored to the other. |
| **`FeatureSpec` is untyped** | `types.py:23` defines `FeatureSpec = dict[str, Any]`. Used in 10+ locations but never validated. Should be a TypedDict or Pydantic model for safety. |
| **`TaskType` / `MetricType` are bare `str`** | `types.py:35-45` defines `TaskType = str`, `MetricType = str`, etc. These are documented as Literal types but not enforced. The actual enforcement is in `config.py` validators. |
| **Missing test files** | No unit tests for: `pipeline/core.py`, `pipeline/iterative.py`, `pipeline/ablations.py`, `agents/router.py`, `agents/unary.py`, `agents/cross_compositional.py`, `agents/aggregation.py`, `agents/temporal.py`, `agents/local_transform.py`, `agents/local_pattern.py`, `llm/factory.py`, `llm/providers/*.py`, `llm/langfuse_wrapper.py`, `evaluation/sandbox.py`, `evaluation/cv.py`, `evaluation/model_factory.py`, `memory/conceptual.py`, `memory/persistence.py`, `artifacts/comparison.py`, `artifacts/diff.py`, `artifacts/dashboard.py`, `data/ingestion.py`, `data/registry.py`, `experiment/matrix.py`, `experiment/runner.py`, `experiment/reporter.py`, `baselines/llmfe.py`, `baselines/openfe.py`, `baselines/malmus.py` |
| **Test files that exist** | `tests/unit/test_api.py`, `test_agents.py`, `test_artifact_schema.py`, `test_artifact_exporter.py`, `test_artifact_storage.py`, `test_config.py`, `test_evaluation.py`, `test_experiment.py`, `test_llm_cache.py`, `test_malmus.py`, `test_memory.py`, `test_observability.py`, `test_retry.py`, `test_utils.py`; `tests/integration/test_artifacts.py`, `test_baselines.py`, `test_data_ingestion.py`, `test_dotenvx_flow.py`, `test_pipeline.py`; `tests/benchmarks/test_performance_smoke.py` |
| **Broad exception handlers** | 26 `except Exception` clauses across codebase. Most are justified (wrapping external API calls), but some should narrow to specific exceptions: `pipeline/core.py:189`, `artifacts/comparison.py:50,55,62`, `baselines/openfe.py:123` |
| **Hardcoded model defaults** | `LLMFEBaseline` and `MalmusBaseline` default to `DeepSeekProvider()` without API key — will raise `LLMError` at instantiation if `DEEPSEEK_API_KEY` env var is unset | `baselines/llmfe.py:48`, `baselines/malmus.py:118` |
| **`nrmse` metric not in config validator** | `METRIC_REGISTRY` at `evaluation/metrics.py:80` includes `nrmse` but `Settings._validate_metric()` at `config.py:266` only allows `{auc, acc, f1, rmse, mae, r2}` — `nrmse` is unreachable via standard config |
| **`ExperimentRunner.run_parallel` tracker gap** | Parallel execution at `experiment/runner.py:70-87` does not call `tracker.init_run()` / `tracker.finish()` / `tracker.log_metrics()` — only sequential `run()` tracks experiments |
| **Sandbox `Exception` in allowed builtins** | `sandbox.py:96` includes `Exception` in `ALLOWED_BUILTINS` — this is listed under both Security and Maintenance because it could enable unexpected behavior in generated code |
