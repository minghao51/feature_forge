# Evaluation Integrity and Sandbox Hardening

**Date:** 2026-09-16
**Status:** Active — PRs 1–5 complete; ADRs 0018–0020 accepted 2026-09-15, PRs 6–7 remaining
**Decision owner:** Maintainer
**Prerequisite:** Plan 22 implementation remains intact; plan 23 PR 1 characterization landed
**Related:** ADR 0001, ADR 0009, ADR 0013, ADR 0017, ADR 0018, ADR 0019, ADR 0020,
`REPORT_LOG.md`, `.planning/STATE.md`

## 0. Continuation checkpoint

The Hamilton-default runtime and fail-fast/cancellation contract are complete,
but the 2026-09-14 audit found correctness and containment gaps in the active
path. The repository already has Silver discovery/evaluation partition support
and typed Platinum policies, but the executor disables the holdout, Platinum
does not consume the partition labels, and several declared policy fields are
dormant. The sandbox isolates generated code in a process but does not restrict
that process from reading files available to the host user.

This plan fixes those gaps in separate, reviewable changes. Do not combine the
evaluation protocol migration, sandbox runner replacement, and LLM cache-key
migration in one change. Preserve the current dirty checkout and never reset,
clean, or overwrite unrelated work.

## 1. Confirmed findings

| Severity | Finding | Evidence |
|----------|---------|----------|
| Critical | Generated code can read host files through allowed NumPy APIs such as `np.fromfile`; AST validation accepts the call and the worker retains host filesystem permissions. | `evaluation/sandbox.py` |
| High | Methods are fitted on every Silver row and target, and the executor forces `evaluation_holdout_fraction=0.0`. | `experiment/hamilton_executor.py` |
| High | Platinum imputes and encodes the complete frame before fold splitting, so validation-fold values influence preprocessing. | `dataflows/platinum.py` |
| High | Candidate selection and the reported score use the same folds. Existing Silver partition labels are ignored. | `dataflows/silver.py`, `dataflows/platinum.py` |
| High | The reported enhanced arm uses the numerically best candidate even when the policy decision rejects it. | `dataflows/platinum.py` |
| Medium | `selection_partition`, `require_positive_lower_bound`, and `max_selected_features` are declared but not enforced. | `contracts/platinum.py` |
| Medium | The paired interval always uses `1.96`, ignores configured confidence, and applies raw deltas even for minimizing metrics. | `dataflows/platinum.py` |
| Medium | LLM cache identity omits the normalized provider endpoint and behavior settings such as DeepSeek thinking mode and reasoning effort. | `llm/base.py`, `llm/providers/deepseek.py` |
| Medium | A sandbox integration test times out at the outer 30-second guard; cancelling `asyncio.to_thread()` cannot stop its running thread. | `methods/malmas/pipeline/core.py`, `tests/integration/test_pipeline.py` |

## 2. Goals

1. Make the default reported score independent of feature discovery and
   selection.
2. Fit every preprocessing operation only on the training portion of its fold.
3. Prevent generated transformations from learning statistics from evaluation
   or validation rows through whole-frame execution.
4. Apply every public selection-policy field or reject unsupported combinations
   before expensive work.
5. Persist enough evidence to reconstruct both selection and final evaluation.
6. Prevent generated code from reading arbitrary host files in the production
   sandbox.
7. Guarantee bounded sandbox launch, execution, termination, and cleanup.
8. Make LLM cache identity cover every request setting that can change output.
9. Keep artifact, cache, and result migrations explicit and reproducible.

## 3. Required decisions

Runtime implementation starts only after these records are accepted:

- **ADR 0018 — Independent discovery and evaluation protocol.** Define the
  default holdout fraction, small-dataset failure behavior, fold-local
  preprocessing, greedy selection semantics, evidence schema version, and the
  explicit compatibility profile.
- **ADR 0019 — Sandbox containment and bounded worker lifecycle.** Define the
  threat model, supported operating systems, strict/degraded modes, filesystem
  and network guarantees, worker protocol, and failure behavior when strict
  isolation is unavailable.
- **ADR 0020 — LLM cache request identity v2.** Define the secret-free provider
  identity, behavior fields, cache-key version, old-entry compatibility policy,
  and provenance requirements.

These decisions affect ADR 0001's sandbox and mandatory-cache boundaries and
ADR 0013's durable Platinum evidence. Amend the ADR index and cross-links when
the records are accepted.

## 4. Target evaluation contract

### 4.1 Protocol profiles

Add an explicit evaluation protocol to typed settings:

- `holdout` is the default. It uses deterministic discovery and evaluation
  partitions, with a configurable fraction defaulting to `0.25`.
- `compatibility` reproduces the current all-row behavior only when selected
  explicitly. Results and manifests identify the estimate as selection-biased;
  user documentation must not call it held-out performance.

Under `holdout`, partition creation must fail with an actionable `DatasetError`
when both partitions cannot independently support the requested folds. It must
never silently fall back to all-row discovery. The fraction and protocol enter
the Silver fingerprint and resolved configuration.

### 4.2 Discovery boundary

- `HamiltonLayerExecutor` fits the method only on rows whose Silver partition
  is `discovery`.
- Generated feature code may execute on the complete feature frame after the
  method has been fitted; it must never receive evaluation targets.
- Candidate trials and all selection decisions use discovery folds only.
- The chosen feature set is frozen before any evaluation-fold score is read.

Generated transformations gain an explicit scope contract:

- `row_local` candidates must produce a row's value independently of companion
  rows; enforce this with deterministic row-subset/permutation metamorphic
  checks before accepting the candidate.
- `fitted` candidates require separate fit-on-training and transform operations
  with serializable learned state. Until that contract is implemented, holdout
  mode rejects them rather than executing a whole-frame statistic.

The compatibility profile may replay legacy whole-frame scripts, but its
provenance must identify the transductive/selection-bias risk. A script that
computes statistics over all evaluation or validation rows cannot be described
as leakage-safe merely because it never receives the target.

### 4.3 Fold-local preprocessing

Replace `_prepared(frame)` with a fold-local transformer:

1. construct the train/validation row masks;
2. fit numeric imputation and categorical encoding on training rows only;
3. transform validation rows with stable unknown-category handling;
4. fit the estimator on the transformed training rows;
5. retain the fitted preprocessing specification in evaluation provenance.

Use a scikit-learn `Pipeline`/`ColumnTransformer` or an equivalently typed small
helper. Do not add a second evaluation framework.

### 4.4 Selection semantics

On discovery folds, apply deterministic greedy forward selection:

1. evaluate each remaining candidate when appended to the already selected
   set;
2. rank by mean directional gain, with a stable feature-name tie-break;
3. require `minimum_practical_gain`;
4. when configured, require a positive directional lower confidence bound;
5. stop at `max_selected_features` or when no remaining candidate qualifies.

`selection_partition="discovery"` is required by the holdout profile.
Unsupported partition/profile combinations fail during request validation.
Compatibility behavior remains explicit and is never reported as independent
evaluation.

After selection freezes, evaluate exactly two arms on evaluation folds:
baseline and the combined selected feature set. If no candidate qualifies, the
enhanced arm mirrors baseline and gain is zero. The public result must come from
this accepted enhanced arm, never from a rejected best trial.

### 4.5 Uncertainty and durable evidence

- Compute paired **directional** fold deltas.
- Use a Student-t critical value derived from the configured confidence level
  and `pair_count - 1` degrees of freedom. If SciPy is imported directly, add it
  as a direct dependency.
- Persist discovery candidate/step fold metrics, final evaluation fold metrics,
  selected-set membership, policy decisions, preprocessing identity, and the
  directional interval.
- Bump the Platinum evidence schema. Old verified packages remain readable but
  cannot satisfy v2 reuse fingerprints.
- Extend semantic verification so reconstructed decisions and aggregates must
  match persisted fold evidence.

## 5. Target sandbox and cache contract

### 5.1 Immediate AST defense

Extend the canonical static policy to reject file-capable APIs exposed by
allowed libraries, including NumPy load/save/fromfile/memmap/data-source APIs
and equivalent Pandas readers/writers. Match normalized attribute paths and
terminal attribute names so simple aliasing does not bypass the rule. Clear
unneeded environment variables in the worker, monkeypatch known library I/O
surfaces as defense in depth, and never expose secrets.

This is an immediate defense, not the final containment boundary.

### 5.2 OS-enforced containment

The production sandbox must prove that generated code cannot read a sentinel
outside its allowed input/output roots. ADR 0019 selects the mechanism after a
time-boxed spike. The preferred design is:

- load trusted runtime modules and the input frame before applying restrictions;
- permit only the specific temporary input/output paths and required runtime
  library reads;
- deny other filesystem reads/writes and socket creation at the operating-system
  boundary;
- fail closed in production when strict enforcement is unavailable;
- expose a clearly named degraded development profile with explicit provenance.

Do not describe working-directory changes, monkeypatches, or AST checks alone as
filesystem isolation.

### 5.3 Bounded worker lifecycle

Add an async sandbox path that starts and owns the child process from the event
loop/main thread rather than calling `multiprocessing.Process.start()` inside
`asyncio.to_thread()`. Factor the sync and async entry points around one process
handle/cleanup helper. All phases must be bounded: worker launch, input load,
generated-code execution, output publication, process join, and cleanup. Use
nonblocking queue polling only after the child starts, or an explicit subprocess
control/result protocol if it proves more reliable. Kill the child/process group
on timeout, wait for termination, close queues, and remove every temporary file.

The async caller must await the async sandbox entry point directly and must not
rely on cancellation of `asyncio.to_thread()` to stop a running sandbox. Use one
authoritative deadline and log that effective value. Tests assert elapsed-time
bounds and the absence of live worker processes or executor threads after
timeout or task cancellation.

### 5.4 LLM cache identity v2

Add a secret-free `cache_identity()` to `LLMClient` and include it with a key
schema version in every plain, JSON, and structured-output cache key. Identity
includes:

- provider and model;
- normalized endpoint origin/path, with credentials and query values removed;
- thinking/reasoning settings;
- effective response-format mode and other provider behavior flags;
- temperature, token limit, rendered messages, and provider request kwargs.

Never include API keys. Treat v1 entries as misses after the migration; retain
them until normal cache maintenance removes them. Record the key schema version
and secret-free identity in cache provenance.

## 6. Implementation sequence

### PR 1 — Characterization and ADRs

- Add focused regression tests that demonstrate each confirmed finding.
- Time-box the OS-isolation spike and document supported behavior.
- Draft ADRs 0018–0020 and update the ADR index.
- Make no runtime behavior changes before maintainer acceptance.

Acceptance: every finding has a deterministic test or a documented
platform-specific reproducer; ADRs state compatibility and rollback choices.

### PR 2 — Partition-aware discovery

- Add typed evaluation protocol/fraction settings.
- Stop hard-coding a zero holdout.
- Fail closed for undersized holdout datasets.
- Fit methods on discovery rows only and thread partition provenance through
  Gold/Platinum requests and fingerprints.
- Add the generated-transformation scope contract; holdout mode accepts only
  verified row-local candidates until fitted transforms are supported.

Acceptance: evaluation targets are never observed by method fitting or
selection; compatibility mode remains explicit and reproducible.

### PR 3 — Fold-local preprocessing and selection

- Introduce the fold-local preprocessing helper.
- Evaluate candidate steps on discovery folds.
- Enforce all selection policy fields with stable ordering.
- Evaluate the frozen combined selection on evaluation folds.

Acceptance: validation-only values/categories do not affect fitted transforms;
rejected candidates never affect the public enhanced score.

### PR 4 — Platinum v2 evidence and uncertainty

- Correct directional Student-t intervals.
- Persist discovery trials and final evaluation evidence.
- Add v1 read compatibility, v2 verification, and v1-to-v2 reuse rejection.
- Update result/report fields and operational verification commands.

Acceptance: an offline loader reconstructs selections, confidence intervals,
and final aggregates from persisted evidence alone.

### PR 5 — Sandbox containment and lifecycle

- Land the expanded AST I/O policy first.
- Implement the ADR 0019 strict OS enforcement.
- Make launch/execution/termination/cleanup bounded.
- Add adversarial filesystem/network and timeout/leak tests across supported
  Python versions.

Acceptance: a generated program cannot read an external sentinel, cannot create
a network connection, and leaves no worker or temporary artifact after timeout.

### PR 6 — LLM cache identity v2

- Implement `cache_identity()` and key versioning.
- Cover plain, JSON, structured, degraded-response-format, and repair calls.
- Document intentional cold misses and retention of v1 entries.

Acceptance: behaviorally identical requests reuse cache entries; endpoint,
thinking, response-mode, or material-kwargs changes produce distinct keys; no
secret material appears in keys, logs, or persisted payloads.

### PR 7 — Documentation, qualification, and claims audit

- Update README, API reference, operations, migration guide, plan/state files,
  generated DAGs, and `REPORT_LOG.md`.
- Remove or qualify `production-ready` claims until strict sandbox and held-out
  evaluation gates pass.
- Run a provider-free fixture qualification and one representative experiment;
  store resolved secret-free configuration and verification reports below
  `experiments/`.

Acceptance: user-facing claims match the selected protocol and sandbox mode;
all required repository checks pass.

## 7. Required tests

1. Method-fit spy proves evaluation rows and targets are never passed to Gold.
2. Candidate-selection spy proves evaluation fold scores are unavailable until
   selection freezes.
3. Numeric imputation and categorical encoding use training-fold statistics;
   unseen validation categories follow the documented policy.
4. Row-local candidates are invariant to row subsets/permutations; a candidate
   using whole-frame statistics is rejected in holdout mode.
5. Small data fails closed under holdout and succeeds only in explicit
   compatibility mode.
6. Minimum gain, positive lower bound, stable ties, and maximum feature count
   are each enforced.
7. A rejected numerical winner yields a baseline-mirroring enhanced arm.
8. Minimization and maximization intervals use directional deltas and respond to
   80%, 95%, and 99% confidence settings.
9. Platinum v2 offline reconstruction and tamper detection pass; v1 packages do
   not enter v2 reuse chains.
10. AST validation blocks direct and aliased NumPy/Pandas I/O attempts.
11. Strict sandbox execution cannot read an external sentinel or connect a
    socket, including through allowed libraries.
12. Launch and execution timeouts terminate within a bounded allowance and leak
    no process, thread, queue, or temporary file.
13. Cache-key tests vary endpoint, thinking, reasoning, response mode, and kwargs
    independently and scan persisted metadata for secret values.
14. Sequential/process parity and fail-fast/cancellation semantics remain
    unchanged after evaluation and sandbox migrations.

## 8. Validation gates

Run the repository-required checks from `AGENTS.md`, plus:

```bash
uv run mypy tests
uv run python scripts/generate_stage_dag_docs.py --check
uv run mkdocs build --strict
uv lock --check
git diff --check
```

Run the full suite on Python 3.11, 3.12, and 3.13 with the documented
single-thread BLAS/OpenMP environment. Sandbox qualification must run on every
operating system claimed as strict; degraded platforms must have an explicit
fail-closed production test.

## 9. Rollback and compatibility

- Each PR is independently revertible.
- Evaluation protocol and Platinum schema changes invalidate reuse through
  fingerprints rather than deleting old packages.
- Compatibility evaluation is explicit and truthfully labeled.
- LLM cache v1 files remain untouched; v2 keys produce cold misses.
- Sandbox fallback never silently weakens production containment.
- No change may bypass `MethodRegistry`, `LLMClient` DiskCache, verified stage
  packages, or the sandbox execution entry point.

## 10. Definition of complete

- ADRs 0018–0020 are accepted and implemented literally.
- Default scores use independent evaluation rows and fold-local preprocessing.
- Every declared selection field changes behavior and has a focused test.
- Platinum v2 evidence reconstructs selection and reporting offline.
- Strict sandbox tests prove filesystem and network denial plus bounded cleanup.
- LLM cache keys distinguish every output-affecting request setting without
  persisting secrets.
- Required multi-version checks and representative qualification pass.
- Documentation and status files no longer overstate evaluation validity or
  sandbox guarantees.
