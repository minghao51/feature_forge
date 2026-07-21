# PR 3 Handoff: Gold Method Adapter and Offline Replay

**Status:** Implemented 2026-07-14; verification recorded below
**Depends on:** Correct case context, layer fingerprints, and manifest integrity
**Index:** [`2026-07-14-medallion-pr-index.md`](2026-07-14-medallion-pr-index.md)

## Objective

Create the Gold boundary around feature generation, execution, verification,
and selection. A committed Gold package must reproduce deterministic feature
values offline without initializing or calling an LLM provider.

## Required design

### Generic method adapter

- Wrap `MethodProtocol` without requiring third-party methods to import
  Hamilton.
- Support a coarse adapter for existing methods and an optional richer staged
  protocol for methods that expose generation/execution/selection separately.
- Consume the PR 2.5 `CaseExecutionContext` and verified `SilverPackage`.
- Preserve existing direct method and sklearn-compatible entrypoints.
- Keep MALMAS-specific agents, routing, memory, and prompts behind its adapter.

### Gold contracts

Add strict, versioned contracts for at least:

- `FeatureCandidate` and its stable candidate ID;
- `FeatureProvenance` including method, agent/round where available, prompt
  bundle, code hash, dependencies, and source columns;
- `FeatureDecision` with accepted/rejected/error state and reason;
- `GoldRequest` and Gold fingerprint inputs;
- `GoldManifest`/materialization reference;
- `GoldPackage` for verified offline loading;
- validation/check records for schema, alignment, values, leakage, determinism,
  and train/inference parity.

### Physical package

Persist separately:

- candidate feature matrix;
- accepted feature matrix;
- feature specifications/provenance;
- decisions and validation checks;
- exact generated code in human-diffable files;
- prompt/template fingerprints, not secret provider configuration;
- dependency/environment snapshot;
- upstream Silver manifest reference.

Candidate persistence should default on for expensive/LLM methods and remain
configurable for cheap deterministic methods. The manifest must always retain
enough evidence to explain every accepted/rejected feature.

### Replay guard

- Add an explicit replay provider guard at the provider factory/boundary.
- Fail if replay code attempts provider construction or completion.
- Load Silver and Gold only through verified manifests.
- Execute persisted code/specifications in the existing sandbox.
- Compare schema, row IDs, dtypes, values/tolerance, and failure outcomes.
- Do not fall back to regeneration when replay evidence is missing or corrupt.

### Correct feature accounting

Replace `len(method.generated_scripts)` as feature count. Record at least:

- candidates proposed;
- candidates successfully executed;
- candidates accepted;
- accepted output columns;
- failed/rejected candidates by reason.

Keep the legacy `num_features_generated` field compatible, with one documented
meaning, and add richer optional fields/contracts rather than overloading it.

### Baseline cache intersection

- Remove `id(y_train)` from any cache contract used by the new flow.
- Use verified Silver identity/folds for durable or cross-process reuse.
- Keep disposable in-process caches clearly separate from Gold evidence.

## Hamilton node boundaries

Recommended batched nodes:

1. `method_generation_request`
2. `candidate_feature_specs`
3. `candidate_execution_batches`
4. `candidate_verification`
5. `candidate_selection`
6. `gold_manifest`
7. `gold_materialization`

Do not create one Hamilton node per feature.

## Expected file areas

- `src/feature_forge/contracts/` Gold models
- `src/feature_forge/dataflows/gold.py`
- `src/feature_forge/dataflows/driver.py`
- `src/feature_forge/methods/base.py`
- `src/feature_forge/methods/malmas/`
- `src/feature_forge/llm/factory.py` or a dedicated replay guard
- `src/feature_forge/evaluation/sandbox.py`
- `src/feature_forge/storage/`
- tests and generated Gold DAG/docs

## Required fixtures/tests

- Deterministic no-LLM method creates and replays a Gold package.
- Recorded MALMAS fixture creates a committed Gold package offline.
- Replay test fails on any provider construction/completion attempt.
- Replay reproduces schema, row IDs, dtypes, and deterministic values.
- Corrupt code/spec/matrix/decision artifacts fail verified loading.
- Candidate and accepted artifacts remain distinct.
- Every candidate has provenance and a decision reason.
- Same Silver + same method inputs yields the same Gold fingerprint.
- Model-only or metric-only changes reuse Gold.
- Prompt/method/config/code changes invalidate Gold.
- Legacy direct method invocation works without importing Hamilton.

## Acceptance criteria

- A recorded MALMAS Gold package replays with zero provider calls.
- Each accepted feature is traceable to code/spec/provenance and a decision.
- Rejected and failed candidates are reviewable without rerunning generation.
- Replay never silently regenerates missing artifacts.
- Feature counts report actual feature records/columns, not code-block count.
- Gold loading verifies upstream Silver identity and package integrity.
- Default platform execution remains unchanged unless the new profile is chosen.

## Performance measurements

Record:

- generation calls and tokens avoided by replay;
- candidate matrix size and package size;
- replay vs regeneration wall time;
- sandbox execution batches and peak concurrency;
- impact of candidate persistence configuration.

## Explicitly out of scope

- Final model evaluation and fold evidence.
- Resume/scheduler behavior.
- DuckDB catalog.
- External orchestration or distributed execution.

## Verification

```bash
uv run ruff check .
uv run mypy src
uv run --extra pipeline pytest tests/unit/test_gold_dataflow.py -q
uv run pytest -m "not llm" -q
uv run python scripts/run_pip_audit.py
uv run mkdocs build --strict
git diff --check
```

## Completion handoff requirements

Report contract versions, Gold package layout, provider-guard enforcement point,
feature-count semantics, recorded fixture provenance, performance measurements,
and exact inputs PR 4 should consume.

## Completion record

- Schema version 1 covers candidates, provenance, decisions, requests,
  materializations, and verified packages.
- Gold input identity supports lookup before generation; materialization
  identity additionally binds specifications, code hashes, provenance, and decisions.
- Gold `accepted` means structural/validation acceptance; final model-, metric-,
  and effect-based selection is deferred to PR 4.
- Packages contain request/candidate/provenance/decision records, optional
  candidate Parquet, required accepted Parquet, checks, dependencies, and ordered Python batches.
- Matrices use authoritative Silver `row_id`; generated code sees feature inputs only.
- Replay guards provider construction and both completion APIs, verifies Gold
  and loads upstream Silver as the authoritative input, executes batches
  cumulatively, independently revalidates accepted/rejected/error outcomes, and
  compares values without fallback.
- The Hamilton graph now performs distinct request, plan, execution,
  verification, selection, manifest, and materialization work. Coarse methods
  receive deterministic batch/spec ownership; optional staged methods expose
  `gold_generation_batches()` without changing `MethodProtocol`.
- `GoldRequest` recomputes its versioned input fingerprint at construction and
  every public boundary, including protection against unchecked `model_copy` updates.
- `execute_gold(...)` is the explicit durable execution entrypoint and returns
  exact versioned counts. Legacy platform execution remains the default and now
  populates all richer `ExperimentResult` count fields with best-effort direct-path evidence.
- Legacy `num_features_generated` counts accepted generated output columns;
  richer optional count fields are appended to `ExperimentResult`.
- PR 4 consumes accepted features, decisions/provenance, the Gold materialization
  fingerprint, and persisted Silver folds.

### Verification results

- `uv run --extra pipeline pytest tests/unit/test_gold_dataflow.py -q`: 16 passed.
- Combined Silver/Gold focused pipeline lane: 31 passed.
- `uv run pytest -m "not llm" -q`: 784 passed (15 deselected) with 88%
  aggregate coverage.
- `uv run ruff check .`: passed.
- `uv run mypy src`: passed across 98 source files.
- Lockfile, default dependency audit, repository hygiene, docs references,
  strict MkDocs, and `git diff --check`: passed. Audit reported no known
  vulnerabilities and the editable project itself as the single ignored item.

### Recorded performance

- Sixteen-test Gold lane wall time: about 31.3 seconds on the local macOS fixture lane.
- Recorded two-batch MALMAS-shaped generation plus replay: 5.30 seconds; replay
  made zero provider construction/completion calls and reused two stored code batches.
- One-batch deterministic generate/commit/replay: 2.67 seconds with candidate
  persistence and 2.66 seconds without it.
- Tiny three-row package: 48 KiB allocated with the 2,154-byte candidate matrix;
  44 KiB without candidate persistence. The accepted matrix was 2,154 bytes.
- Fixture token usage is zero/recorded rather than a live-provider measurement;
  live token-cost benchmarking remains intentionally outside required offline CI.
- Sandbox peak concurrency in these deterministic fixtures is one; batches are
  ordered so later generated features can consume earlier validation-accepted outputs.
