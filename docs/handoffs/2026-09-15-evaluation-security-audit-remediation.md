# Evaluation Integrity and Security Audit Remediation Handoff

**Date:** 2026-09-15  
**Status:** Complete — PR 1 landed and ADRs 0018–0020 accepted; superseded by
`docs/handoffs/2026-09-15-plan23-pr2-partition-aware-discovery.md` (PR 2)  
**Authoritative plan:** `docs/plan/23_evaluation_integrity_security_hardening.md`  
**Required decisions:** ADRs 0018–0020 — **accepted by the maintainer on
2026-09-15**; runtime PRs 2–7 are unlocked

## Start here

Read in this order:

1. `AGENTS.md`
2. `docs/README.md`
3. `README.md`
4. `docs/plan/00_index.md`
5. `docs/plan/23_evaluation_integrity_security_hardening.md`
6. ADRs 0001, 0009, 0013, 0014, and 0017
7. this handoff

The checkout contains extensive intentional uncommitted work. Preserve it.
Never reset, clean, broadly stash, reformat unrelated files, or commit unless
the maintainer explicitly requests it. `.env` has been removed from Git tracking
and must remain local/ignored; never print or inspect its values.

## Exact next assignment

**Superseded.** PR 1 landed 2026-09-15 (see `REPORT_LOG.md` and the PR-2
handoff) and the maintainer accepted ADRs 0018–0020. Continue with
`docs/handoffs/2026-09-15-plan23-pr2-partition-aware-discovery.md`.

The original PR-1 assignment was: add focused characterization/regression
tests for every confirmed finding; run a time-boxed OS containment spike for
the sandbox; draft ADRs 0018–0020 and add them to `docs/decisions/README.md`;
update cross-links and `REPORT_LOG.md`; stop before changing runtime behavior
and obtain maintainer acceptance of the ADRs.

## Completed in the planning thread

- Added `docs/plan/23_evaluation_integrity_security_hardening.md`.
- Added plan 23 to `docs/plan/00_index.md`, `docs/README.md`, and `mkdocs.yml`.
- Removed `.env` from Git tracking while retaining the local ignored file.
- Extended `scripts/check_repo_hygiene.py` to reject tracked `.env` and dotenvx
  key files.
- Corrected the audit record to state that the committed `.env` was encrypted.
- No runtime implementation, schema migration, dependency change, or ADR was
  created or accepted.

## Confirmed evaluation findings

### Dormant partition contract

- `DatasetComputationRequest.evaluation_holdout_fraction` defaults to 0.25.
- `silver.fold_assignments` already creates deterministic `discovery` and
  `evaluation` partitions and independent folds within them.
- `HamiltonLayerExecutor` overrides the fraction to `0.0` and calls
  `method.fit()` with all Silver features and targets.
- Platinum ignores the partition column.

Relevant files:

- `src/feature_forge/contracts/datasets.py`
- `src/feature_forge/dataflows/silver.py`
- `src/feature_forge/experiment/hamilton_executor.py`

### Fold preprocessing leakage

`dataflows/platinum.py::_prepared()` computes category codes and medians on the
complete frame before fold masks are applied. A regression fixture should put a
distinct median and an unseen category only in validation rows and assert that
the training-fold transformer does not learn them.

### Selection/reporting mismatch

- Candidate selection and reporting use the same fold evidence.
- Only `minimum_practical_gain` is consulted.
- `selection_partition`, `require_positive_lower_bound`, and
  `max_selected_features` are dormant.
- `platinum_materialization()` persists the numerical best candidate even when
  `selection_decisions()` rejects it.

Relevant files:

- `src/feature_forge/contracts/platinum.py`
- `src/feature_forge/dataflows/platinum.py`
- `src/feature_forge/dataflows/_io.py`
- `tests/unit/test_platinum_dataflow.py`

### Uncertainty defect

`uncertainty_summary()` stores the configured confidence value but always uses
1.96 and applies the interval to raw deltas. A minimizing metric therefore gets
raw-sign bounds inconsistent with `mean_directional_gain`. Characterize 80%,
95%, and 99% confidence plus an RMSE/minimization case.

## Confirmed security and lifecycle findings

### Filesystem access

The AST validator accepts this class of operation:

```python
def generate_features(df):
    import numpy as np
    df["external_size"] = len(np.fromfile("/path/outside/sandbox", dtype="uint8"))
    return df
```

Do not use a real secret path in tests. Create a sentinel in a temporary parent
directory and verify both the current reproducer and the proposed strict denial.
AST denylists are immediate defense only; ADR 0019 must select an OS enforcement
mechanism and state which platforms qualify as strict.

Current documentation overstates this boundary by saying filesystem operations
are blocked and generated execution is in-memory-only. Audit and correct
`docs/methods.md` and `docs/plan/01_architecture.md` with the implementation.

Relevant files:

- `src/feature_forge/evaluation/sandbox.py`
- `src/feature_forge/methods/malmas/pipeline/codegen.py`
- `tests/unit/test_sandbox.py`

### Timeout and thread lifecycle

The isolated integration test
`tests/integration/test_pipeline.py::TestCorePipeline::test_run_with_fake_agent`
timed out in `_exec_sandbox()` after about 30 seconds in the audit environment,
including with single-thread BLAS/OpenMP limits. `_exec_sandbox()` wraps the
synchronous worker in `asyncio.to_thread()` and `asyncio.wait_for()`; cancelling
the await does not terminate the running thread. The security review reproduced
that standalone process start succeeds, while starting the spawned child inside
`asyncio.to_thread()` hangs. `SandboxedExecutor` also has unbounded work around
process launch/join boundaries, and the warning logs `sandbox_timeout * 2` even
though the effective outer deadline is `max(sandbox_timeout * 2, 30)`.

PR 1 should add instrumentation/reproducers without raising timeouts or hiding
the failure. The implementation target starts/manages the child from the event
loop/main thread, awaits it nonblockingly, terminates and reaps it on
timeout/cancellation, and proves no process, thread, queue, or temporary-file
leak. Retain synchronous `execute()` compatibility and share lifecycle logic
between sync and async entry points.

### LLM cache collision

`LLMClient.build_cache_key()` currently hashes provider, model, messages,
temperature, token limit, and explicit kwargs. It does not include `base_url`,
DeepSeek `thinking_enabled`, or `reasoning_effort`. Distinct clients with those
settings can produce the same key. Characterize each field independently and
cover plain, JSON, structured, degraded-response-format, and repair paths.

Never put an API key or endpoint credentials/query values into the identity.
ADR 0020 should define a secret-free normalized endpoint identity and an
explicit key schema version. The proposed migration treats v1 entries as cold
misses without deleting them.

Relevant files:

- `src/feature_forge/llm/base.py`
- `src/feature_forge/llm/cache.py`
- `src/feature_forge/llm/providers/deepseek.py`
- `tests/unit/test_llm_base.py`
- `tests/unit/test_llm_cache.py`

## Proposed decisions for ADR review

### ADR 0018

- Default protocol: independent holdout, fraction 0.25.
- Explicit legacy behavior: `compatibility` profile with truthful biased-score
  provenance.
- Small/imbalanced datasets: fail closed when both partitions cannot support
  folds; never silently collapse to discovery-only under the holdout profile.
- Method fitting and selection: discovery targets only.
- Generated transformations: holdout mode accepts verified row-local candidates;
  whole-frame/statistical transforms require an explicit fitted-transform
  contract and frozen learned state.
- Final reporting: baseline versus frozen combined selection on evaluation
  folds.
- Selection: deterministic greedy forward selection, directional gain,
  confidence/practical-gain gates, stable ties, configured cap.
- Evidence: Platinum v2, with v1 readable but ineligible for v2 reuse.

### ADR 0019

- Threat model covers untrusted generated code attempting host filesystem and
  network access through allowed scientific libraries.
- Production strict mode fails closed when OS enforcement is unavailable.
- Degraded development mode is explicit and recorded in provenance.
- The chosen worker protocol bounds launch, execution, termination, and cleanup.
- A sentinel-outside-root test is the filesystem acceptance gate.

The spike must compare available local mechanisms on supported CI platforms.
Do not select working-directory changes or monkeypatch-only enforcement as the
strict solution. Record dependency, portability, import/runtime-read, and
process-startup tradeoffs.

### ADR 0020

- Cache identity v2 includes a secret-free endpoint identity and every
  output-affecting provider mode.
- Effective response-format fallback state is represented consistently.
- v1 entries remain on disk but are never returned for a v2 request.
- Cache provenance records key version without messages, credentials, or secret
  endpoint components.

## PR sequence after ADR acceptance

1. Partition-aware discovery and typed protocol settings.
2. Fold-local preprocessing and enforceable selection.
3. Platinum v2 evidence and directional Student-t uncertainty.
4. Strict sandbox containment and bounded lifecycle.
5. LLM cache identity v2.
6. Documentation, representative qualification, and claims audit.

Follow the detailed scopes and acceptance gates in plan 23. Preserve plan 22
fail-fast/cancellation behavior and worker-local Hamilton services throughout.

## Validation baseline

Passing during the audit/planning work:

- Ruff check and format check.
- `mypy src`.
- repository hygiene and documentation-reference checks.
- focused Platinum, LLM cache, execution-policy, and sandbox unit tests.
- `git diff --check`.

Not passing in the audit sandbox:

- the isolated CorePipeline fake-agent test described above. Treat it as an
  observed failure to reproduce, not yet as a cross-platform regression.

## Validation commands

Use fresh temporary artifact/cache roots. The required final gate is:

```bash
uv sync --all-groups --extra intel
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run python scripts/check_repo_hygiene.py
uv run python scripts/check_docs_references.py
uv run python scripts/generate_stage_dag_docs.py --check
uv run mkdocs build --strict
uv lock --check
git diff --check
```

Also run `mypy tests` and the full suite on Python 3.11, 3.12, and 3.13 with
single-thread BLAS/OpenMP limits. Strict sandbox qualification must run on every
operating system claimed as supported.

## Stop conditions

Stop and report to the maintainer when:

- an ADR choice changes the `MethodRegistry`, mandatory `LLMClient` DiskCache,
  sandbox entry point, or medallion stage boundaries beyond plan 23;
- strict OS enforcement requires a new system dependency or dropping a claimed
  platform;
- a method cannot honor discovery-only fitting without a new fit/transform
  contract;
- Platinum v2 cannot reconstruct selection offline from persisted evidence;
- an implementation would delete old artifact packages or LLM cache entries.

Do not work around these conditions silently.
