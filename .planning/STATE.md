# Feature Forge — Current State

Last updated: 2026-05-15

## What's Implemented

| Feature | Backend | Frontend |
|---------|---------|----------|
| FeatureForge sklearn API (`api.py`) | Full | N/A |
| ExperimentalPlatform facade (`platform.py`) | Full | N/A |
| LLM providers (DeepSeek, OpenAI, Anthropic, LiteLLM) | Full | N/A |
| LLM client with retry, JSON mode, disk caching | Full | N/A |
| Configuration (pydantic-settings + YAML + env vars) | Full | N/A |
| Sandbox: AST validation, process isolation, resource limits | Full | N/A |
| CV evaluator + 7 metrics (AUC, ACC, F1, RMSE, MAE, R2, NRMSE) | Full | N/A |
| Model factory (XGBoost, LightGBM, CatBoost, RF, MLP) | Full | N/A |
| CAAFE method (unified + fidelity variants) | Full | N/A |
| LLMFE method (single_shot + iterative) | Full | N/A |
| Malmus method (JSON-mode structured output) | Full | N/A |
| MALMAS method (multi-agent pipeline) | Full | N/A |
| OpenFE method (non-LLM baseline wrapper) | Full | N/A |
| MALMAS 6 specialized agents | Full | N/A |
| Router agent (hybrid/data_driven/performance_driven/llm) | Full | N/A |
| Agent memory (procedural, feedback, conceptual) | Full | N/A |
| MALMAS ablation pipelines (no_memory, no_router, single_agent) | Full | N/A |
| Experiment tracking (WandB, MLflow, NoOp) | Full | N/A |
| Artifact storage (memory, disk, hybrid) | Full | N/A |
| Langfuse observability integration | Full | N/A |
| Dataset registry (Kaggle, local, entry points) | Full | N/A |
| Plugin system (methods, agents, models, metrics, datasets via entry points) | Full | N/A |
| MkDocs documentation site | Full | N/A |

## Stubbed / Unimplemented

Abstract hooks with concrete implementations in subclasses (by design):

- `src/feature_forge/llm/base.py:123` — `_call_api()` raises `NotImplementedError`; overridden by DeepSeek, OpenAI, Anthropic, LiteLLM providers
- `src/feature_forge/llm/base.py:127` — `_extract_content()` raises `NotImplementedError`; overridden by all providers
- `src/feature_forge/llm/base.py:135` — `_extract_usage()` raises `NotImplementedError`; overridden by all providers
- `src/feature_forge/methods/malmas/pipeline/iterative.py:61` — `_select_agents()` raises `NotImplementedError` in `BaseIterativePipeline`; overridden by `IterativePipeline`, `SingleAgentPipeline`, `NoRouterPipeline`

No genuine stubs found — all abstract methods have working implementations.

## Known Bugs

| Severity | Issue | Location |
|----------|-------|----------|
| Medium | Silent exception swallow when LLM client creation fails — defers error to `fit()` with no logged warning | `src/feature_forge/api.py:74` |
| Medium | `except Exception:` with no logging silently ignores dataset registry entry-point load failures | `src/feature_forge/data/registry.py:69` |
| Medium | `except Exception:` with no logging silently ignores OpenFE importance extraction failures | `src/feature_forge/methods/openfe/method.py:123` |
| Low | `except Exception:` with no logging in experiment tracker artifact fallback | `src/feature_forge/experiment/tracker.py:65` |
| Low | Sandbox `_to_parquet_safe` has two `except Exception:` blocks with no logging | `src/feature_forge/evaluation/sandbox.py:57`, `src/feature_forge/evaluation/sandbox.py:65` |

## Security Concerns

| Severity | Issue | Location |
|----------|-------|----------|
| Medium | `exec()` in sandbox worker runs LLM-generated code (mitigated by AST validation + process isolation + restricted builtins, but `exec` is still used) | `src/feature_forge/evaluation/sandbox.py:376` |
| Medium | `subprocess.run` invoked with generated code piped via stdin for ruff linting — ruff path not validated | `src/feature_forge/methods/malmas/pipeline/core.py:152` |
| Low | Hardcoded API key placeholder `"sk-..."` in LangfuseLLMWrapper docstring | `src/feature_forge/llm/langfuse_wrapper.py:23` |
| Low | `.env` file committed to git (encrypted via dotenvx, but risk if decryption keys are exposed) | `.env` |
| Low | `socket.create_connection` monkey-patched only inside worker process; main process is unmodified | `src/feature_forge/evaluation/sandbox.py:350` |

## Performance Issues

| Issue | Location |
|-------|----------|
| Per-feature CV evaluation spawns up to `min(os.cpu_count(), 8)` threads per round — can overwhelm on high-core machines with many candidate features | `src/feature_forge/methods/malmas/pipeline/core.py:482` |
| Sandbox creates a new `spawn` subprocess per `execute()` call — no process reuse; high overhead for iterative pipelines with many features | `src/feature_forge/evaluation/sandbox.py:216` |
| `CorePipeline.run()` is ~410 lines handling code gen, sandbox exec, test exec, and evaluation — latency is dominated by sequential LLM calls within the semaphore (max 3 concurrent by default) | `src/feature_forge/methods/malmas/pipeline/core.py:200` |
| Baseline CV score re-evaluated per `CorePipeline.run()` invocation (not cached across rounds); each baseline call trains N-fold models | `src/feature_forge/methods/malmas/pipeline/core.py:474` |
| `_column_desc_cache` in `BaseFeatureAgent` uses FIFO eviction (not LRU) — frequently-used entries may be evicted prematurely | `src/feature_forge/methods/malmas/agents/base.py:187` |

## Maintenance Issues

| Issue | Detail |
|-------|--------|
| Broad `except Exception` with no logging | 6 instances silently swallow errors: `src/feature_forge/api.py:74`, `src/feature_forge/data/registry.py:69`, `src/feature_forge/evaluation/sandbox.py:57`, `src/feature_forge/evaluation/sandbox.py:65`, `src/feature_forge/experiment/tracker.py:65`, `src/feature_forge/methods/openfe/method.py:123` |
| Broad `except Exception` with logging but no re-raise | 22 instances catch and log but don't propagate — makes debugging pipeline failures harder (scattered across `core.py`, `iterative.py`, `method.py` files in methods/) |
| `MethodRegistry._discovered` is a class-level mutable cache that never invalidates | `src/feature_forge/methods/base.py:92` — new entry points installed at runtime are invisible until process restart |
| `MetricRegistry._discovered` same class-level mutable cache issue | `src/feature_forge/evaluation/metrics.py:104` |
| `ModelRegistry._discovered` same class-level mutable cache issue | `src/feature_forge/evaluation/model_factory.py:100` |
| Generated artifacts directory not cleaned up on failure | `src/feature_forge/evaluation/sandbox.py:218-265` — temporary parquet files cleaned in `finally`, but `feature_forge_artifacts/` dir accumulates across runs |
| `wandb/` and `htmlcov/` directories present in repo despite being listed in `.gitignore` | `wandb/`, `htmlcov/` — already tracked before gitignore rules were added |
| `site/` directory (built MkDocs site) committed to repo despite `/site` in `.gitignore` | `site/` — should be generated in CI, not tracked |
| `_column_desc_cache` is a `ClassVar` dict shared across all `BaseFeatureAgent` instances — not thread-safe for concurrent access | `src/feature_forge/methods/malmas/agents/base.py:138` |
