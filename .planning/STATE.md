# Feature Forge — Current State

Last updated: 2026-07-20

## Current Working Tree

The medallion refactor and the R12–R19 remediation series are present in the
working tree but **not yet committed** (~75 files changed/added on `main`). The
last commit on `main` is `74f33fd`; everything described below is uncommitted.
This STATE section captures the verified state of the working tree.

Verification performed on 2026-07-20 (Python 3.13, default env + `pipeline`
extra for the medallion lanes):

- `uv run ruff check src tests` — pass
- `uv run mypy src` — pass, 112 source files
- `uv run pytest tests/ -m "not slow and not llm"` — pass, 913 tests collected,
  15 deselected (slow/llm). (A `_FastEvaluator` stub was missing the
  `metric_direction` attribute added by R18 and broke two benchmark tests; the
  stub was updated and the suite is green again — see R20.)
- `uv run python scripts/run_pip_audit.py` — no known vulnerabilities in the
  default environment (torch risk in the optional `all` extra remains, see
  Security Concerns).
- `check_repo_hygiene.py`, `check_docs_references.py` — pass.

Deferred refactors (tracked, not done in this pass):

- `num_features_generated` is overloaded: in `case_executor` and the Platinum
  dataflow it means "accepted output columns", in MALMAS `iterative.py` it means
  "proposed specs". Renaming or splitting it touches 6 test files and the
  reporter column header, so it is deferred to its own change.
- `CaseComputation` previously called `fit()` then `transform(X)` on the
  training frame, re-executing generated code. It now calls
  `method.fit_transform(X, y)`; `MALMASMethod.fit_transform` was updated to
  delegate to `FeatureForge.fit_transform`, which reuses the cached
  `X_train_enhanced` rather than re-running the sandbox.

## Medallion Refactor Progress

PR 0 through PR 7 from `.claude/handoffs/2026-07-13-medallion-refactor.md` are implemented
within their recorded boundaries. PR 7 does not make the library a turnkey durable materializer:
full Bronze-to-Platinum execution requires an application-owned, module-level `LayerExecutor`.

- ADRs added under `docs/adr/` for the Hamilton boundary, medallion-lite semantics, artifact formats/atomic commit, and the MkDocs decision.
- Versioned Pydantic contracts added under `src/feature_forge/contracts/`.
- Secret-free dataset/case fingerprint helpers and a run-scoped atomic local artifact store added under `src/feature_forge/storage/`.
- Existing `DataFrameStorage` remains backward-compatible and supports optional run-scoped paths via `ArtifactConfig.run_id`.
- `ExperimentResult` retains its legacy fields and now has optional manifest/run identity fields for forward compatibility.
- Optional `pipeline` dependencies are pinned in `uv.lock`: `sf-hamilton==1.90.0`, `apache-hamilton==1.90.0`, and `duckdb==1.5.4`.
- Profile-aware Bronze/Silver Hamilton dataflow, deterministic folds/checks, durable Bronze snapshots/references, verified offline Silver loading, DAG generation, and an optional pipeline CI lane are implemented.
- Review hardening completed: persisted contracts reject unknown/future fields, identifiers and artifact paths are cross-platform safe, fingerprints redact only actual secrets and reject ambiguous normalization, package publication is atomic and idempotent, and manifests enforce layer/schema/artifact integrity.
- PR 2.5 adds a per-case execution context. Dataset-owned task metadata, a compatible metric, seed, folds, evaluation/resource settings, run identity, artifact policy, and execution profile are resolved before method/provider construction. Built-in adapters receive the same settings/evaluator as outer evaluation, including mixed classification/regression matrices.
- Bronze, Silver, Gold-input, and Platinum-input identities are versioned separately. Silver reuse compares the Silver fingerprint rather than the aggregate case fingerprint.
- Manifests now bind `layer`, `package_kind`, and `layer_fingerprint`; safe resolution requires a declared artifact and rechecks size/hash. The Silver loader enforces exact package filenames, media types, schema versions, and layers.
- Adversarial PR 2.5 follow-up: regression defaults to higher-is-better R², `_SUCCESS` anchors the commit-time manifest SHA-256, and existing Bronze/Silver reuse is controlled by verified namespace/layer fingerprint rather than full request equality.
- Optional dependencies are explicit: `pipeline` owns Hamilton/DuckDB, `observability` does not, and aggregate `all` includes `pipeline`. The lock resolves `setuptools==83.0.0`.
- Security audit: isolated default and pipeline environments report no known vulnerabilities. The aggregate `all` environment retains the accepted optional Torch risk (`torch==2.10.0`: `PYSEC-2026-139` and `CVE-2025-3000`, no published fix in the resolved lane). The existing temporary DiskCache advisory allowlist remains scoped in `security/pip_audit_allowlist.txt`.
- PR 2.5 verification passed before PR 3. After the PR 5 adversarial remediation, the combined lane passes 830 non-LLM tests at 88% coverage. Ruff and strict mypy (104 source files), dependency audit, lockfile, diff, repository hygiene, docs references, and strict MkDocs gates pass.
- PR 6 focused catalog/CLI/telemetry plus PR 3-5 regression lanes pass. Ruff and
  strict mypy now cover 110 source files; lock, diff, repository hygiene,
  docs-reference, strict MkDocs, and dependency-audit gates pass. After the
  delta remediation, one clean full lane passed all 857 non-LLM tests at 88%
  coverage.
- PR 6 adversarial remediation closes artifact-reference authorization,
  semantic lineage authorization, complete dependent-row reconciliation,
  case-terminal stale detection, production Hamilton/journal/cache integration,
  expanded telemetry redaction, complete typed lifecycle indexing, and shared
  catalog-writer durability. Delta review additionally closes malformed-journal
  reconciliation crashes and standalone provider-key telemetry leakage.
- PR 7 makes MkDocs canonical for pipeline and operations, adds deterministic
  Silver/Gold/Platinum/case topology references plus canonical portable Mermaid DAGs,
  unchecked PNG previews, and generated
  contract/check/catalog/CLI/package references, freshness enforcement, offline
  documentation tests, a socket-denied/scrubbed MkDocs wrapper, and a non-allow-failure CI
  docs job. Branch-rule required-check
  enforcement has not been inspected. All Markdown pages now have an explicit navigation
  location.
- PR 7 removes stale Quarto automation and invalid dependency examples, documents
  legacy-to-medallion migration/rollback, and keeps PR 8 deferred because no
  interaction-scale requirement or owner/security/hosting approval exists.
- PR 7 operability is verified within the library boundary: planning, resume inspection,
  existing-package replay, recovery, verification, and catalog rebuild are documented and
  tested. The original turnkey new-contributor criterion is explicitly not met; application
  integration must supply the durable `LayerExecutor` that materializes valid packages.
- PR 7 final verification passes all 877 pipeline-enabled non-LLM tests at 88%
  coverage, Ruff, strict mypy across 112 source files plus the runnable operations example and
  offline MkDocs wrapper,
  deterministic generated
  freshness, strict MkDocs, hygiene, documentation references, dependency audit,
  lockfile consistency, and diff whitespace.
- PR 3 adds strict Gold candidate/provenance/decision/request/package contracts, separate input and materialization fingerprints, ordered sandbox replay, row-ID-keyed candidate and validation-accepted matrices, exact package loading, provider construction/completion guards, and actual output-column feature accounting. Final metric/effect selection remains assigned to Platinum.

PR 4 adds row-keyed Platinum fold evidence, paired uncertainty, final
evidence-based selection, and append-only enriched results. PR 5 adds stable
failure-aware sequential/spawn scheduling, fail-fast cancellation, portable
worker registrations, worker-owned tracking, deterministic nested-resource
plans, safer classified retries, append-only lifecycle events, verified
contiguous resume, side-effect-free dry-run, and conservative artifact crash
recovery. PR 6 adds a rebuildable versioned DuckDB catalog, transactional
manifest indexing, deterministic reconciliation, read-only verification and
planning CLI commands, redacted Hamilton lifecycle/cache telemetry, and
metrics/reference-only tracker publication with optional/required failure
policy. The default legacy execution result shape remains compatible. PR 8 remains
deferred; no next implementation phase is approved. Continue with canonical MkDocs and
generated static references until its interaction and governance gates are met.
The evidence-only readiness sequence is tracked in
`.claude/handoffs/2026-07-14-pr8-decision-readiness.md`; it permits measurement,
requirements, schema proposals, and synthetic-data spikes, but does not authorize
Astro implementation or deployment.
PR 8 readiness now includes a deterministic synthetic scale generator with small
and expected measurements, a deny-by-default public snapshot schema proposal,
and a synthetic-only snapshot fixture. Owners, interaction tasks, security
approval, hosting, and deployment evidence remain outstanding; PR 8 is still
deferred.
Non-legacy platform execution now fails closed without per-run layer
fingerprints and a portable layer executor; dry-run reports the same read-only
resume decisions used by execution.

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
| R18 | Metric direction was assumed "higher is better" throughout the legacy feature-selection path — `CVEvaluator.evaluate_feature` returned `new_score - baseline_score` and `BaseMethod._evaluate_and_select` / `CorePipeline._evaluate_and_select` kept features with `gain > 0`, so RMSE/MAE/NRMSE improvements (negative raw gains) were discarded. `MetricDirection` existed but was only wired into the Platinum path and reporter | `CVEvaluator` now resolves `metric_direction` in `__init__` and exposes `_directional()` / `evaluate_feature_directional` / `evaluate_features_batch_directional` mirroring the Platinum sign convention. `BaseMethod._evaluate_and_select` uses the directional batch; `CorePipeline._evaluate_and_select` selects/sorts on directional gains with a direction-aware failure sentinel; `IterativePipeline._record_feature_in_memory` takes an explicit `maximize` flag. Raw gains are preserved on `PipelineResult.gains` and in artifacts for transparency (`src/feature_forge/{evaluation/cv.py,methods/base.py,methods/malmas/pipeline/{core,iterative}.py}`) |
| R19 | `CaseComputation._method_kwargs` built method adapters via `inspect.signature` reflection, which silently dropped `llm_client` (CAAFE unified raised through the platform had `llm_client=None` → `EvaluationError("llm_client required...")`) and dropped `mode` for MALMAS (its `__init__(config, **kwargs)` didn't declare it) | Each built-in method now exposes an explicit `from_run_context(cls, ctx)` classmethod that receives a single `CaseExecutionContext`; `CaseComputation._construct_method` dispatches through it, falling back to `_method_kwargs` only for third-party plugins. `CaseExecutionContext` gained `llm_client`, `sandbox`, and `eval_kit` fields built once in `resolve_case_context` (best-effort client — None without an API key). `MALMASMethod.__init__` now declares `mode` and `llm_client` explicitly so they reach `FeatureForge`. LLMFE/Malmus/CAAFE reuse the context-owned client instead of self-building (`src/feature_forge/{experiment/{context,case_executor}.py,methods/{caafe,llmfe,malmus,openfe,malmas}/method.py}`) |
| R20 | The R18 direction-aware-selection work added `CVEvaluator.metric_direction` and `CorePipeline._evaluate_and_select` reads `self.evaluator.metric_direction`, but the `_FastEvaluator` test stub in `tests/benchmarks/test_performance_smoke.py` was not updated — `test_candidate_eval_backend_smoke_budget[threading\|loky]` raised `AttributeError: '_FastEvaluator' object has no attribute 'metric_direction'` and would have failed CI | `_FastEvaluator` now exposes `metric_direction = MetricDirection.MAXIMIZE` matching the `CVEvaluator` surface read by the pipeline (`tests/benchmarks/test_performance_smoke.py`) |
| R21 | `CaseComputation` (legacy case path) called `method.fit(X, y)` followed by `method.transform(X)`, re-executing every generated code block on the training frame even though `fit` had already produced an enhanced frame. `MALMASMethod.fit_transform` had the same shape — `fit` then `transform` — so it never reused `FeatureForge`'s cached `X_train_enhanced` | `CaseComputation` now calls `method.fit_transform(X, y)` (the canonical path documented by R2). `MALMASMethod.fit_transform` delegates to `self._forge.fit_transform`, which returns the cached enhanced frame. The legacy executor no longer pays for a second sandbox pass over the training data (`src/feature_forge/{experiment/case_executor.py,methods/malmas/method.py}`, `tests/unit/test_malmas_method.py`) |

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
| Medium | The optional `all`/CAAFE dependency set installs `torch==2.10.0`, which `pip-audit` reports as affected by PYSEC-2026-139 and CVE-2025-3000 with no published fix version in this environment. The default and pipeline environments do not install this dependency. | `pyproject.toml`, `uv.lock` |
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
| `structlog` configured globally on first `get_logger()` call | `src/feature_forge/observability/structlog_config.py:95-100` — `cache_logger_on_first_use=True` means logging config cannot be changed at runtime without process restart |
| `MetricRegistry._builtin` is a class var shared across all instances | `src/feature_forge/evaluation/metrics.py:102` — `register()` modifies class-level dict, affecting all users of the registry globally. `MetricRegistry.reset()` now restores builtin defaults for test isolation. |
| `ProcessPoolExecutionAdapter.run` re-raises as `RuntimeError` on worker failure | `src/feature_forge/experiment/execution.py:106` — loses original exception type, making error handling harder for callers |
| `num_features_generated` is overloaded | In `case_executor.py` / `dataflows/platinum.py` it means "accepted output columns"; in `methods/malmas/pipeline/iterative.py` it means "proposed specs". The reporter surfaces the legacy-case value as a column. Renaming or splitting it touches 6 test files and the reporter; tracked here so it is not lost. |
