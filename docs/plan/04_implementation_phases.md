# Implementation Phases

## Phase 1: Skeleton & Tooling (Days 1-2)

**Goal:** Establish project foundation with modern Python tooling.

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | `uv init --src feature-forge`, create directory structure | `pyproject.toml`, `src/`, `tests/`, `config/` |
| 1 | Configure `pyproject.toml` with metadata, dependencies, tool configs | Installable package |
| 2 | Set up GitHub Actions CI (`ci.yml`) | Automated lint/test on PR |
| 2 | Set up pre-commit hooks (ruff, conventional commits) | `.pre-commit-config.yaml` |
| 2 | Configure dotenvx for secrets | `.env` (encrypted), `.env.keys` (gitignored) |

**Success Criteria:**
- `uv sync` completes without errors
- `uv run pytest` runs (even if no tests yet)
- `uv run ruff check .` passes
- `pre-commit install` succeeds
- `import feature_forge` works after `uv pip install -e .`

---

## Phase 2: Config & Types (Days 3-4)

**Goal:** Immutable, validated, env-var-overridable configuration.

| Day | Task | Deliverable |
|-----|------|-------------|
| 3 | Implement `Settings` with pydantic-settings | `src/feature_forge/config.py` |
| 3 | Create `config/settings.yaml` with defaults | `config/settings.yaml` |
| 3 | Write `exceptions.py` with full hierarchy | `src/feature_forge/exceptions.py` |
| 4 | Write `types.py` with shared aliases | `src/feature_forge/types.py` |
| 4 | Add unit tests for config validation | `tests/unit/test_config.py` |

**Success Criteria:**
- `Settings()` loads from YAML
- `Settings(temperature=0.5)` overrides YAML
- `FF_TASK=regression` env var overrides
- Invalid config raises `ConfigurationError`
- `settings.llm.api_key` returns `SecretStr`

---

## Phase 3: Observability (Days 5-6)

**Goal:** Structured logging + LLM tracing from day one.

| Day | Task | Deliverable |
|-----|------|-------------|
| 5 | Configure structlog (JSON prod / pretty dev) | `src/feature_forge/observability/structlog_config.py` |
| 5 | Add OpenTelemetry processor for trace correlation | `add_open_telemetry_spans` processor |
| 6 | Integrate Langfuse `@observe` decorators | `src/feature_forge/observability/langfuse_tracer.py` |
| 6 | Add unit tests for logging | `tests/unit/test_observability.py` |

**Success Criteria:**
- `structlog.get_logger().info("test", x=1)` outputs JSON in CI
- `bind_contextvars(experiment_id="e1")` propagates to all logs
- `@observe()` decorator captures function latency
- Langfuse traces show in cloud dashboard

---

## Phase 4: LLM Layer (Days 7-9)

**Goal:** Provider-agnostic LLM client with enforced caching.

| Day | Task | Deliverable |
|-----|------|-------------|
| 7 | Implement `LLMClient` ABC | `src/feature_forge/llm/base.py` |
| 7 | Implement `DiskCache` with SHA-256 keys | `src/feature_forge/llm/cache.py` |
| 8 | Implement OpenAI provider | `src/feature_forge/llm/providers/openai.py` |
| 8 | Implement DeepSeek provider | `src/feature_forge/llm/providers/deepseek.py` |
| 9 | Implement Anthropic provider | `src/feature_forge/llm/providers/anthropic.py` |
| 9 | Add Langfuse wrapper (auto-tracing) | `src/feature_forge/llm/langfuse_wrapper.py` |

**Success Criteria:**
- `LLMClient.complete(messages)` returns response
- Same prompt returns cached response (no API call)
- Langfuse shows generation spans with token usage
- Invalid API key raises `LLMError`

---

## Phase 5: Agent System (Days 10-13)

**Goal:** All 6 MALMAS agents + Router + Registry.

| Day | Task | Deliverable |
|-----|------|-------------|
| 10 | Implement `Agent` ABC + `AgentRegistry` | `src/feature_forge/agents/base.py` |
| 10 | Port prompt templates to `src/feature_forge/prompts/` | All 6 prompt files |
| 11 | Implement UnaryFeatureAgent | `src/feature_forge/agents/unary.py` |
| 11 | Implement CrossCompositionalAgent | `src/feature_forge/agents/cross_compositional.py` |
| 12 | Implement AggregationConstructAgent | `src/feature_forge/agents/aggregation.py` |
| 12 | Implement TemporalFeatureAgent | `src/feature_forge/agents/temporal.py` |
| 13 | Implement LocalTransformAgent + LocalPatternAgent | `src/feature_forge/agents/local_transform.py`, `local_pattern.py` |
| 13 | Implement RouterAgent | `src/feature_forge/agents/router.py` |

**Success Criteria:**
- `AgentRegistry.discover()` finds all 6 agents
- Each agent generates a `FeatureSpec` from a prompt
- Router selects agents based on data characteristics
- All agents are entry-point registered

---

## Phase 6: Memory System (Days 14-16)

**Goal:** Procedural, feedback, and conceptual memory.

| Day | Task | Deliverable |
|-----|------|-------------|
| 14 | Implement `MemoryEntry` dataclass + persistence | `src/feature_forge/memory/base.py`, `persistence.py` |
| 14 | Implement `ProceduralMemory` | `src/feature_forge/memory/procedural.py` |
| 15 | Implement `FeedbackMemory` | `src/feature_forge/memory/feedback.py` |
| 15 | Implement `ConceptualMemory` (with LLM summarization) | `src/feature_forge/memory/conceptual.py` |
| 16 | Integrate memory into agent base class | `Agent.memory` attribute |
| 16 | Add memory tests | `tests/unit/test_memory.py` |

**Success Criteria:**
- Memory persists across rounds
- Conceptual memory generates bullet-point rules
- Top-k features retrievable by score
- Memory serializes to JSON and loads back

---

## Phase 7: Evaluation (Days 17-19)

**Goal:** Feature evaluation, model factory, sandboxed execution.

| Day | Task | Deliverable |
|-----|------|-------------|
| 17 | Implement metrics (AUC, ACC, NRMSE) | `src/feature_forge/evaluation/metrics.py` |
| 17 | Implement k-fold CV evaluator | `src/feature_forge/evaluation/cv.py` |
| 18 | Implement ModelFactory | `src/feature_forge/evaluation/model_factory.py` |
| 18 | Implement `SandboxedExecutor` | `src/feature_forge/evaluation/sandbox.py` |
| 19 | Add evaluation tests | `tests/unit/test_evaluation.py`, `test_sandbox.py` |

**Success Criteria:**
- `cv.evaluate_feature(X, y, feature_code)` returns gain
- `SandboxedExecutor` blocks `eval()`, `open()`, imports
- ModelFactory creates XGB/LGB/CatBoost/RF/MLP
- Sandbox allows `pandas`, `numpy`, `math` operations

---

## Phase 8: Pipeline & API (Days 20-23)

**Goal:** Core pipeline, iterative pipeline, ablations, sklearn API.

| Day | Task | Deliverable |
|-----|------|-------------|
| 20 | Implement `CorePipeline` (single round) | `src/feature_forge/pipeline/core.py` |
| 21 | Implement `IterativePipeline` (N-round) | `src/feature_forge/pipeline/iterative.py` |
| 22 | Implement ablation pipelines | `src/feature_forge/pipeline/ablations.py` |
| 22 | Implement `MALMASFeatureEngineer` (sklearn) | `src/feature_forge/api.py` |
| 23 | Add pipeline integration tests | `tests/integration/test_pipeline.py` |

**Success Criteria:**
- `fe.fit(X_train, y_train)` runs full pipeline
- `fe.transform(X_test)` applies generated features
- `Pipeline([("fe", fe), ("clf", XGBClassifier())])` works
- `cross_val_score(pipeline, X, y)` works

---

## Phase 9: Baselines (Days 24-27)

**Goal:** OpenFE, CAAFE, LLM-FE baseline implementations.

| Day | Task | Deliverable |
|-----|------|-------------|
| 24 | Implement `Baseline` ABC + `BaselineRegistry` | `src/feature_forge/baselines/base.py` |
| 24 | Implement OpenFE baseline | `src/feature_forge/baselines/openfe.py` |
| 25 | Implement CAAFE baseline | `src/feature_forge/baselines/caafe.py` |
| 26 | Implement LLM-FE baseline | `src/feature_forge/baselines/llmfe.py` |
| 27 | Add baseline tests | `tests/integration/test_baselines.py` |

**Success Criteria:**
- Each baseline implements `fit(X_train, y_train)` / `transform(X_test)`
- `BaselineRegistry.discover()` finds all baselines
- OpenFE baseline matches reference implementation

---

## Phase 10: Experiment Harness (Days 28-31)

**Goal:** Unified tracking, experiment matrices, auto-reporting.

| Day | Task | Deliverable |
|-----|------|-------------|
| 28 | Implement `ExperimentTracker` ABC | `src/feature_forge/experiment/tracker.py` |
| 28 | Implement `WandBTracker` | `src/feature_forge/experiment/wandb_backend.py` |
| 29 | Implement `MLflowTracker` | `src/feature_forge/experiment/mlflow_backend.py` |
| 29 | Implement `ExperimentMatrix` | `src/feature_forge/experiment/matrix.py` |
| 30 | Implement `ExperimentRunner` | `src/feature_forge/experiment/runner.py` |
| 31 | Implement `Reporter` | `src/feature_forge/experiment/reporter.py` |

**Success Criteria:**
- `ExperimentMatrix` generates all combinations
- `ExperimentRunner` executes in parallel
- WandB shows all metrics, parameters, artifacts
- Reporter generates markdown comparison tables

---

## Phase 11: Data Layer (Days 32-33)

**Goal:** Kaggle-focused data ingestion with sample datasets.

| Day | Task | Deliverable |
|-----|------|-------------|
| 32 | Implement `KaggleFetcher` | `src/feature_forge/data/ingestion.py` |
| 32 | Implement `DatasetRegistry` | `src/feature_forge/data/registry.py` |
| 33 | Add sample datasets + ingestion tests | `data/samples/`, `tests/integration/test_data_ingestion.py` |

**Success Criteria:**
- `KaggleFetcher.fetch("titanic")` downloads dataset
- `DatasetRegistry.list()` shows available datasets
- Sample datasets load without internet
- Ingestion handles CSV + metadata JSON

---

## Phase 12: Tests & Documentation (Days 34-38)

**Goal:** Comprehensive test coverage and interactive notebooks.

| Day | Task | Deliverable |
|-----|------|-------------|
| 34-35 | Unit tests for all core modules | `tests/unit/` — target 80%+ coverage |
| 36 | Integration tests | `tests/integration/` |
| 37 | Marimo notebooks | `notebooks/01_agent_comparison.py`, etc. |
| 38 | API reference docs | `docs/api_reference.md` |

**Success Criteria:**
- `pytest --cov=feature_forge` shows >80% coverage
- All integration tests pass
- Notebooks run end-to-end

---

## Phase 13: Benchmarks & Release Prep (Days 39-42)

**Goal:** Full benchmark suite and release readiness.

| Day | Task | Deliverable |
|-----|------|-------------|
| 39-40 | Run full benchmark suite | `.github/workflows/benchmark.yml` |
| 41 | Write README with quick start | `README.md` |
| 42 | Write migration guide | `docs/migration_guide.md` |

**Success Criteria:**
- Benchmark workflow runs on schedule
- README has working code examples
- Package installable via `uv pip install -e .`

---

## Total Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1-2 (Foundation) | 4 days | Day 4 |
| 3-5 (Core Engine) | 9 days | Day 13 |
| 6-8 (Pipeline) | 9 days | Day 22 |
| 9-11 (Methods + Data) | 9 days | Day 31 |
| 12-13 (Quality + Release) | 9 days | Day 40 |

**Total: ~6 weeks** (assuming 1 developer, full-time)
