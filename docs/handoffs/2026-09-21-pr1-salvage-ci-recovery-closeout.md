# PR #1 — Salvage, CI Recovery, and Superseded-PR Closeout

**Date:** 2026-09-21  
**Status:** Ready for execution; no remote mutation or implementation performed by the audit session  
**Remote PR:** `https://github.com/minghao51/feature_forge/pull/1`  
**PR head:** `origin/feat/medallion-refactor` at
`690a76436669fff43bb2aaff8b6069dc684e40a0`  
**Merge base:** `74f33fdfcb1da9e7d506044cc2c80e433cee21ff`  
**Audited main:** `e2b0c6b`  
**Decision:** do not merge or cherry-pick PR #1 wholesale. Preserve its history,
salvage selected provenance and design lessons, create successor work, then
close it as superseded.

## 1. Read first / authority order

1. `AGENTS.md` and `docs/README.md`.
2. `docs/plan/21_hamilton_default_execution_handoff.md` — the current
   Hamilton-default specification and the remaining source-branch references.
3. Accepted ADRs:
   - `docs/decisions/0002-three-tier-agent-memory.md`
   - `docs/decisions/0006-experiment-execution-seam.md`
   - `docs/decisions/0010-intel-openmp-bootstrap.md`
   - `docs/decisions/0013-medallion-lite-boundaries.md`
   - `docs/decisions/0016-remove-legacy-execution-engine.md`
   - `docs/decisions/0017-failure-policy-and-cancellation-contract.md`
   - `docs/decisions/0018-independent-discovery-evaluation-protocol.md`
   - `docs/decisions/0019-sandbox-containment-bounded-lifecycle.md`
4. This handoff and `docs/handoffs/README.md`.
5. PR #1 itself and its four commits:
   - `3e09b1a` — medallion core + R12–R21
   - `c524895` — branch-local state refresh
   - `4f1610e` — CI hardening
   - `690a764` — Phase 2/R22

Where the PR conflicts with main or an accepted ADR, main/ADR wins. Never
restore the removed legacy execution engine to make the branch mergeable.

## 2. Audit result

### 2.1 Remote state

Verified on 2026-09-21:

- PR #1 is `OPEN`, `mergeable=CONFLICTING`, `mergeStateStatus=DIRTY`, with no
  comments or reviews.
- Branch diff from the merge base: 236 paths, +30,663/−1,249; 110 changed
  paths differ from current main and 116 branch-added paths are absent.
- The PR body says the CI lanes are green, but the current PR checks have two
  failures: `security` and `quality-debt`. Its three Python test lanes and
  three pipeline lanes pass.
- Current main's latest CI run is `35596522314`; it failed. The last ten
  queried main-branch push runs also report failure. Local validation at
  `e2b0c6b` remains green (1106 passed / 9 skipped / 8 xfailed), so the
  current failures are runner/dependency-sensitive rather than a simple
  deterministic local regression.

### 2.2 Already landed / equivalent — do not port

These branch blobs are byte-identical to current main:

- `src/feature_forge/contracts/artifacts.py`
- `src/feature_forge/contracts/materialization.py`
- `src/feature_forge/contracts/stages.py`
- `src/feature_forge/experiment/wandb_backend.py`
- `src/feature_forge/storage/__init__.py`
- `src/feature_forge/storage/atomic.py`
- `src/feature_forge/storage/hashing.py`
- `src/feature_forge/verification/checks.py`

The following behavior also exists on main, usually in a stronger post-plan-23
form: registry unification, DeepSeek cleanup, MALMAS codegen split, tracker
backend `none`, direction-aware evaluation, cached enhanced frames, discovery
holdout, atomic packages, Hamilton dataflows, semantic artifact verification,
and generated stage-DAG docs.

### 2.3 Direct conflicts — never port verbatim

- Branch `evaluation/sandbox.py`: predates ADR 0019 strict containment and the
  bounded lifecycle. Do not replace current sandbox code.
- Linux `fork` worker start: ADR 0010 warns that forking after Intel/OpenMP
  initialization can deadlock (`0010-intel-openmp-bootstrap.md:35-46`). Keep
  `spawn` unless a new accepted decision changes this.
- Pickle sidecar: current untrusted-code boundary deliberately blocks
  pickle/parquet reader/writer surfaces from generated code. Do not introduce
  a fallback without a fresh containment review.
- Branch lazy package initializer: removes the required
  `bootstrap_intel_runtime()` ordering. The idea may be redesigned, but the
  branch file is not portable.
- MALMAS 3-tier → 2-tier memory deletion: conflicts with accepted ADR 0002.
- Branch `docs/adr/0001–0004`: numbers collide with current ADRs; never copy
  them into `docs/decisions/` under those names.
- Branch `dataflows/_selection.py`: implements independent-candidate
  threshold/cap selection. Current main implements greedy forward selection
  plus independent persisted-evidence reconstruction (`dataflows/platinum.py`
  and `dataflows/_io.py`); importing the old helper would weaken integrity.
- Branch DuckDB catalog as a required path: conflicts with ADR 0013's
  warehouse-free authoritative package store. Keep optional/deferred.
- Branch workflow, hygiene script, lockfile, generated docs, and legacy
  case-executor files: stale against plans 21–23; do not merge them.

### 2.4 Portable value found

Highest-value ideas to reimplement against current main:

1. **One-way process IPC.** Current sandbox uses `ctx.Queue(maxsize=1)` at
   `src/feature_forge/evaluation/sandbox.py:756-758`. PR commit `4f1610e`
   uses `ctx.Pipe(duplex=False)`, avoiding queue feeder-thread and semaphore
   allocation for a single response. This is compatible in principle with
   spawn and ADR 0019, but must pass every lifecycle/leak/containment test.
2. **Scoped address-space cap.** Current `_sandbox_worker_main` applies
   `RLIMIT_AS` before worker work (`sandbox.py:1219-1222`); the cap remains
   active while the worker attempts to post its result. `4f1610e` scopes the
   cap around generated-code execution and checks current `VmSize`. Re-design
   this carefully: worker startup must succeed under 512 MB, while generated
   code must remain meaningfully bounded.
3. **BLAS/OpenMP thread caps in CI.** Current `.github/workflows/ci.yml`
   defines none of `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
   `MKL_NUM_THREADS`, or `NUMEXPR_NUM_THREADS`. The PR's value is the measured
   single-thread runner policy, not its full stale workflow.
4. **Explicit method construction.** Current
   `experiment/hamilton_executor.py:607-631` still reflects over constructor
   signatures. The PR's `from_run_context()` concept removes silent omission
   risk. Its old `CaseExecutionContext` cannot be copied because ADR 0016
   removed that engine; if adopted, make an explicit `BaseMethod` factory
   contract and record any boundary change before implementation.
5. **Resource-limit ideas/tests.** The branch has deterministic nested
   parallelism and `threadpoolctl` concepts in `experiment/resources.py` and
   broad scenarios in `test_scheduler_resume_resources.py`. Current main
   retains `ResourceConfig`/`EffectiveResourcePlan` contracts but does not
   apply them. Salvage small measured seams only; do not reintroduce the old
   scheduler or automatic resource retry wholesale.
6. **PR8 deferral criteria.** The branch contains a useful decision rule for
   an optional Astro/browser catalog. Preserve the outcome as one
   `docs/deferred-design.md` entry rather than copying its stale generated
   evidence corpus.

Lower-priority ideas, not authorized by this handoff: configurable process-pool
fallback, explicit `BaseMethod.from_run_context`, full dynamic resource plans,
or an optional rebuildable DuckDB catalog. Each needs demonstrated experiment
or operational demand.

## 3. Current CI failures — keep causes separate

### 3.1 Sandbox/startup failure family

Latest main run `35596522314`, Python 3.13:

- `test_sandbox.py::TestExecuteFullPath::test_success_path_preserves_index`
  → `SandboxTimeoutError` after 5 seconds.
- `test_hamilton_recovery.py::TestGoldReplayAfterRecovery::test_verified_gold_evidence_replays_without_provider`
  → zero accepted columns and
  `failure_counts={'SandboxTimeoutError': 1}`.
- `test_malmus.py::TestMalmusMethodIterative::test_iterative_fit_discards_feature`
  → timeout swallowed by the method, then `KeyError: 'gains'`.

The first two prove the timeout family; Malmus must preserve a stable typed
failure record so the root error cannot masquerade as `KeyError`. Local focused
runs pass, confirming runner sensitivity.

Do not assume one fix solves every failure. Add phase timing/exit diagnostics
before changing architecture: parent serialization, spawn/import, input load,
containment setup, generated-code execution, output publication, result send,
and process exit.

### 3.2 Hamilton compatibility dependency drift

The `apache-hamilton<2` job runs:

```yaml
uv pip install --upgrade "${{ matrix.hamilton-spec }}"
```

which also upgraded NumPy/Pandas in the observed runner. Its Gold mismatch was
accompanied by `std::system_error: Resource temporarily unavailable`, but it is
not yet proven to be solely the sandbox defect. Pin/constraint the compatibility
lane so it changes Hamilton only, and set matrix `fail-fast: false` so both
Hamilton cells report.

### 3.3 Security audit drift

The security job uses `uv sync --group dev` then audits that broad environment.
It found 51 advisories in 11 packages; the allowlist contains only
`CVE-2025-69872`. PR #1 does not solve this—its own `security` check fails.
Do not allowlist 51 findings blindly. Separate runtime dependencies from
notebook/docs/tooling audit scope, upgrade what is fixable, and document each
remaining temporary exception with an owner/removal trigger.

## 4. Execution plan

### Slice A — immutable preservation and handoff recovery

**Remote mutation requires explicit maintainer authorization.** Once granted:

1. Fetch and verify the branch head exactly:

   ```bash
   git fetch origin feat/medallion-refactor
   test "$(git rev-parse origin/feat/medallion-refactor)" = \
     "690a76436669fff43bb2aaff8b6069dc684e40a0"
   ```

2. Create/push an annotated immutable tag, suggested name:

   ```text
   archive/pr-1-medallion-refactor
   ```

   Tag target must be `690a76436669fff43bb2aaff8b6069dc684e40a0`.

3. Recover exactly these ten files from branch `.claude/handoffs/` into
   `docs/handoffs/` (do not check out or merge the branch):

   - `2026-07-13-medallion-refactor.md`
   - `2026-07-14-medallion-pr-index.md`
   - `2026-07-14-pr2.5-foundation-stabilization.md`
   - `2026-07-14-pr3-gold-offline-replay.md`
   - `2026-07-14-pr4-platinum-evidence.md`
   - `2026-07-14-pr5-scheduler-resume-resources.md`
   - `2026-07-14-pr6-catalog-verification-observability.md`
   - `2026-07-14-pr7-docs-operations.md`
   - `2026-07-14-pr8-decision-readiness.md`
   - `2026-07-14-pr8-optional-astro-catalog.md`

4. Byte-compare every recovered file against `git show
   690a764:.claude/handoffs/<name>` before any normalization. Audit found no
   `.claude/handoffs/` or `docs/adr/` references inside this set, so prefer
   verbatim provenance. The recovered PR8 files do reference branch-only
   `docs/decisions/pr8-*`, `docs/generated/pr8/`, and `scripts/*pr8*` paths;
   these are expected dead references in historical provenance—resolve them
   through the archive tag, never by copying the targets.
5. Add all ten to `docs/handoffs/README.md`, status:
   `Historical — plan lineage; execution superseded by plans 21–23` (PR8 pair:
   `Historical deferral record`). Amend the directory-history paragraph to
   note recovery from remote PR #1 on 2026-09-21.
6. Update `docs/plan/21_hamilton_default_execution_handoff.md`:
   - source implementation → immutable tag + SHA;
   - lines currently saying the 2026-07-13/14 handoffs were “not retained” →
     their real `docs/handoffs/` locations.
7. Update the corresponding `REPORT_LOG.md` source reference.
8. Add `D-10 Optional browser/interactive catalog` to
   `docs/deferred-design.md`:
   - defer because current static MkDocs + generated references are adequate;
   - revival requires at least two demonstrated interactive requirements,
     representative scale, a stable versioned allowlisted public-snapshot
     schema, and named security/hosting/maintenance owners.
9. Do not copy branch `docs/adr/`, `docs/generated/pr8/`, or the PR8 scripts.
   They remain accessible under the immutable tag.

Suggested commit (only when explicitly authorized):

```text
docs(handoffs): preserve PR 1 medallion planning lineage
```

### Slice B — sandbox CI recovery (highest priority code work)

Worker/reviewer workflow:

- worker: `opencode-go/deepseek-v4.1-flash`;
- reviewer: `zai/glm-5.3-flash`, read-only;
- main agent owns any changes to
  `tests/unit/test_plan23_sandbox_hardening.py`;
- never run sandbox suites concurrently—shared `/tmp` creates false leak
  failures.

Implementation order:

1. Add bounded diagnostics/phase timing sufficient to distinguish spawn/import,
   resource-cap, and response-send failures. Keep provenance secret-free.
2. Add CI single-thread environment caps first; rerun the failing GitHub lane.
3. Reimplement one-way spawn-compatible IPC in current lifecycle code. Preserve:
   - one authoritative deadline;
   - sync/async shared cleanup;
   - typed EOF/worker-death errors;
   - process-group kill/wait;
   - no worker/thread/temp-file leaks;
   - strict Landlock/network behavior.
4. Rework memory limiting only after a test pins desired semantics. At minimum:
   - 512 MB configured worker starts and returns a small frame on CI;
   - generated code that exceeds a reachable cap fails through the typed sandbox
     hierarchy;
   - cap-skipping because imported VmSize already exceeds the requested value is
     explicit in provenance/logging, never silently advertised as enforced;
   - output publication remains bounded.
5. Fix Malmus iteration records so failure paths always contain stable
   `gains`/`kept` fields or a typed error field; retain the original timeout in
   logs/results.
6. Do not port fork, pickle sidecars, or branch package initializers.

Focused sequential gates:

```bash
uv run pytest tests/unit/test_sandbox.py -q
uv run pytest tests/unit/test_sandbox_lifecycle.py -q
uv run pytest tests/unit/test_sandbox_containment.py -q
uv run pytest tests/unit/test_sandbox_io_defense.py -q
uv run pytest tests/unit/test_plan23_sandbox_hardening.py -q
uv run pytest tests/unit/test_malmus.py -q
uv run pytest tests/integration/test_hamilton_recovery.py -q
```

Suggested commits:

```text
ci: cap native worker threads on constrained runners
fix(sandbox): make spawn-worker response IPC resource-safe
fix(malmus): preserve typed iteration failures
```

Keep those concerns separable; do not hide the runtime fix in a timeout-only
increase.

### Slice C — Hamilton compatibility lane

1. Reproduce in a clean environment; do not mutate the validated developer
   environment with `uv pip install --upgrade`.
2. Constrain Hamilton without upgrading unrelated NumPy/Pandas majors.
3. Set matrix `fail-fast: false` for observability.
4. Run both `apache-hamilton==1.90.0` and `<2` cells against the five pinned
   dataflow/cache test modules from `.github/workflows/ci.yml`.
5. Preserve the integration tests—AGENTS.md explicitly says Hamilton
   reuse/recovery/integrity is never deferred past handoff.

Suggested commit:

```text
ci(hamilton): isolate compatibility dependency resolution
```

### Slice D — security audit policy

1. Export the current pip-audit result as a review artifact; classify runtime,
   test, docs/notebook, and tooling dependencies.
2. Upgrade safe/fixed dependencies and refresh the lockfile deliberately.
3. Prefer separate runtime and tooling audit lanes over a blanket allowlist.
4. Any remaining allowlist item needs vulnerability ID, affected dependency,
   reason, owner, expiry/removal trigger, and evidence that the vulnerable path
   is not shipped or reached.
5. The security job must finish green; do not mark it non-blocking.

Suggested commit:

```text
ci(security): separate runtime and tooling dependency audits
```

### Slice E — successor records and PR closeout

**Remote mutation requires explicit maintainer authorization.** After Slice A
is pushed and successor CI work is represented by issues or tracked handoffs:

1. Add a PR #1 comment containing:
   - immutable tag + SHA;
   - where the 10 handoffs moved;
   - current-main equivalents for already-landed code;
   - links to sandbox, Hamilton compatibility, and security successor work;
   - explicitly rejected scope (fork, stale sandbox replacement, 2-tier memory,
     colliding ADRs, DuckDB authority, old selection helper);
   - correction that current PR checks still fail `security` and
     `quality-debt`.
2. Close PR #1 as **superseded**. Do not merge.
3. Keep `feat/medallion-refactor` until tag fetchability and all recovered files
   are verified from a fresh fetch. Branch deletion, if desired, is a separate
   explicitly authorized operation.

## 5. Architecture candidates intentionally deferred

Do not fold these into CI recovery:

- replacing reflection with `BaseMethod.from_run_context`;
- full `ResourceConfig` / `EffectiveResourcePlan` application;
- automatic lower-concurrency retry;
- implicit process-pool → sequential fallback;
- DuckDB catalog / Astro UI;
- full verification-service port.

Record a new ADR before changing the `BaseMethod` extension contract or the
accepted sandbox/OpenMP boundaries.

## 6. Validation and acceptance

### Documentation/provenance acceptance

- [ ] Immutable tag resolves to exactly `690a76436669fff43bb2aaff8b6069dc684e40a0`.
- [ ] All 10 handoffs recovered and byte-verified; index/status updated.
- [ ] Plan 21 and REPORT_LOG no longer depend on a mutable branch reference.
- [ ] D-10 records the browser-catalog revival trigger.
- [ ] No branch ADR/generated corpus copied into canonical decision docs.
- [ ] `uv run python scripts/check_repo_hygiene.py`
- [ ] `uv run python scripts/check_docs_references.py`
- [ ] `uv run mkdocs build --strict`
- [ ] `git diff --check`

### CI-recovery acceptance

- [ ] Latest GitHub Actions run is green for Python 3.11/3.12/3.13.
- [ ] Both Hamilton compatibility cells run (not cancelled) and pass.
- [ ] Security job passes under documented non-blanket policy.
- [ ] Sandbox 512 MB and 2048 MB success paths are pinned on Linux.
- [ ] Timeout/cancellation/worker-death paths remain bounded, typed, and leak-free.
- [ ] Strict containment tests remain unchanged or stronger; no fork/pickle
      containment regression.
- [ ] Malmus exposes the root typed error, never a secondary missing-key error.
- [ ] Full local suite and every AGENTS.md gate pass sequentially:

  ```bash
  uv sync --all-groups --extra intel
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy src
  uv run python scripts/check_repo_hygiene.py
  uv run python scripts/check_docs_references.py
  uv run mkdocs build --strict
  uv lock --check
  git diff --check
  ```

### PR closeout acceptance

- [ ] Successor work and salvage map linked from PR #1.
- [ ] PR closed as superseded, never merged.
- [ ] Remote branch retained until independent tag/file verification.

## 7. Out of scope

- Plan 23 PR 6 / ADR 0020 LLM cache identity work; it remains specified in
  `docs/handoffs/2026-09-18-plan23-pr6-llm-cache-identity-v2.md`.
- Broad refactoring or formatting of branch-derived code.
- Reopening accepted ADR 0002, 0010, 0013, 0016, 0018, or 0019 without a new
  decision record.
- Any supplied-data modification.
- Merging/cherry-picking PR #1 to simplify salvage.

## 8. Audit provenance

The 2026-09-21 audit used:

- direct Git/`gh` inspection of PR #1, both refs, current CI runs and checks;
- a read-only GLM-5.3-Flash plan/governance review;
- a read-only DeepSeek V4.1 Flash architecture/CI comparison after two reviewer
  endpoint failures;
- main-agent verification of contested claims (ADR 0010 lazy/fork conflict,
  current greedy selection, sandbox resource-limit ordering, current artifact
  verification, PR8 governance, branch-only tests/docs).

No files, tags, branches, PR state, or remote refs were changed during that
audit. This handoff itself is the only implementation artifact of the follow-up
request until a future agent is explicitly authorized to proceed.
