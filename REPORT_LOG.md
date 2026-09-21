# Report Log

Chronological record of material changes, findings, and disclosures for
Feature Forge. Referenced by `AGENTS.md` (engineering conventions).

## What counts as an entry

- Changes to architectural boundaries or project guidance (link the ADR).
- Experiment results or findings worth remembering across sessions.
- Incidents: unexpected LLM cost, data issues, broken reproducibility.
- Material documentation restructuring.

Small code changes covered by tests and the changelog do not need an entry.

## Format

One entry per event, newest first:

```
### YYYY-MM-DD — Short title
- What happened / what was found.
- Links: ADR, experiment dir, PR.
```

## Entries

### 2026-09-21 — Remote PR #1 salvage and CI-recovery audit

- Audited open PR #1 (`feat/medallion-refactor@690a764`) against current main
  (`e2b0c6b`): the PR is conflicting/dirty and predates ADRs 0016–0020, so it
  must not be merged or wholesale cherry-picked. Eight medallion contract/
  storage/check blobs are already byte-identical on main; registry, tracker,
  directional evaluation, holdout, Hamilton dataflow, and verification scope
  has otherwise landed or been superseded by plans 21–23.
- Identified durable salvage: recover the ten July medallion handoffs as
  historical provenance; preserve the branch tip under an immutable tag before
  close; add the deferred Astro/browser-catalog decision and measurable revival
  trigger to `docs/deferred-design.md`. Do not import the colliding branch ADRs
  or stale generated PR8 corpus.
- Verified current remote CI is red (run `35596522314`; the last ten queried
  main push runs also report failure). Python 3.13 showed a 5-second sandbox
  startup timeout, Hamilton recovery persisted
  `failure_counts={'SandboxTimeoutError': 1}`, and Malmus masked its timeout
  with `KeyError: 'gains'`; Hamilton `<2` additionally changed unrelated
  NumPy/Pandas versions and emitted `Resource temporarily unavailable`;
  security found 51 advisories in the broad dev environment. These are split
  into sandbox, Hamilton-compatibility, and security-policy work rather than
  treated as one root cause.
- Portable PR lessons are one-way spawn-compatible Pipe IPC, scoped/observable
  address-space limiting, and CI BLAS/OpenMP thread caps. Rejected verbatim
  ports: Linux fork (ADR 0010 OpenMP conflict), pickle sidecars/current sandbox
  replacement (ADR 0019), branch lazy init (drops Intel bootstrap ordering),
  2-tier memory (ADR 0002 conflict), old independent-candidate selection
  (superseded by plan-23 greedy evidence reconstruction), and authoritative
  DuckDB catalog (ADR 0013).
- Execution handoff:
  `docs/handoffs/2026-09-21-pr1-salvage-ci-recovery-closeout.md`.
- AI assistance: read-only plan/governance review by GLM-5.3-Flash,
  architecture/CI audit by DeepSeek V4.1 Flash after two reviewer endpoint
  failures, and direct source/ADR/remote verification plus handoff synthesis by
  the pi session agent. No source, remote ref, tag, or PR state was changed.

### 2026-09-18 — Post-audit fix batch: leakage fail-closed, typed worker-death errors, portability and provenance pins

- Verified three-way audit findings (src/tests/docs-meta reviewers, then an
  independent verification pass over every must-fix claim with quoted
  evidence): 12 confirmed/partial, 1 refuted. The refuted carry-forward is
  closed: `LocalArtifactStore.commit` does NOT replace committed namespaces —
  `atomic_publish_directory` fails closed with `FileExistsError` under a
  per-namespace mkdir lock (`storage/atomic.py:60-63`); only `.tmp-` staging
  dirs and stale locks are ever removed. Overwrite risk does not exist.
- `dataflows/platinum.py::_partition_scope` now fails closed: under the
  default holdout protocol a Silver package whose `fold_assignments` lacks a
  `partition` column raises `DatasetError` (mirroring the executor's
  fit-time guard) instead of silently evaluating discovery AND reported
  evidence on all rows — the exact leakage ADR 0018 bans, reachable via
  reused pre-ADR-0018 Silver packages. All-rows fallback remains only for
  `compatibility`. Negative test added; fixtures updated mechanically
  (including a maintainer redesign of the plan-23 marker fixtures so every
  scope hosts two non-degenerate folds — the preprocessing pins got
  STRONGER: the train-median pin now discriminates whole-frame leakage
  (3.5), discovery-inclusive leakage (2.5), and correct train-only stats
  (2.0)).
- `evaluation/sandbox.py`: worker death without posting (e.g. RLIMIT_AS
  killing the queue feeder thread) surfaced as raw `EOFError`, bypassing the
  `CodeExecutionError` hierarchy the pipeline handles; both queue-read
  paths (`_wait_for_response`, `_poll_response`) now translate `EOFError` →
  `CodeExecutionError`. Pinned by a deterministic closed-write-end test
  (both paths) plus an end-to-end typed-error/leak-freedom test. Also
  blocked `socket.socketpair` in the worker runtime shim (was left usable
  next to the `_BlockedSocket` socket shim).
- Test-infrastructure integrity: the async leak pin in
  `test_sandbox_lifecycle.py` no longer hardcodes `/tmp` (was vacuous under
  `TMPDIR != /tmp`); the two strict-profile lifecycle pins gained the
  `_require_landlock()` guard (previously FAIL, not skip, on non-Landlock
  hosts); new pins for the production default being strict (Settings +
  CorePipeline wiring) and the worker-side `containment_unavailable` →
  `SandboxContainmentError` path (in-process `_sandbox_worker_main`, no
  production seam added).
- Two new §7-item-13 strict xfails in `test_plan23_llm_cache_identity.py`
  (PR-6 backlog pointers): v2 `cache_identity()` must vary with provider
  request kwargs independently, and persisted cache entries must carry a
  versioned, secret-free `cache_provenance` record scannable off disk
  (verified to go through the real cache write path). xfail count 6 → 8.
- Honesty fixes: bare `assert`s under `python -O` replaced by explicit
  raises (`platinum.py` artifact-set check, `api.py` manifest check);
  `EnvironmentSnapshot`'s dead `None`-settings fallback now reports
  `"unknown"` instead of fabricating `"strict"`; ADR 0019 network-denial
  wording corrected (ABI ≥ 4 Landlock denies TCP bind/connect only;
  AF_UNIX/NETLINK remain kernel-permitted, Python-level construction blocked
  by the runtime shim incl. socketpair; below ABI 4 seccomp denies
  `socket(2)` outright). PR-6 ADR pointer corrected to ADR 0020 (was
  misattributed to ADR 0018 decision 6) in `.planning/STATE.md` and
  `docs/plan/00_index.md`; plan-23 status bumped to PRs 1–5 complete.
- Full suite: 1106 passed / 9 expected skips / 8 xfailed, zero XPASS; all
  AGENTS.md gates green (ruff, format, mypy src, hygiene, docs refs,
  mkdocs --strict, uv lock, git diff --check).
- Handoff: `docs/handoffs/2026-09-18-post-audit-fix-batch.md`.
- AI assistance: implementation by a worker subagent (DeepSeek V4.1 Flash
  via opencode-go), review by a reviewer subagent (GLM-5.3-Flash),
  marker-module edits, verification passes, coordination, and commits by the
  session agent (pi) at the maintainer's request; no dependency or
  supplied-data changes.

### 2026-09-18 — Two-tier test workflow: coverage opt-in, timing-flake fix, xdist spike

- Made suite coverage opt-in: removed `--cov=feature_forge` and both report
  flags from pytest `addopts` (CI already passes `--cov --cov-report=xml`
  explicitly, so it is unaffected; no `fail_under` gate exists). Every dev
  invocation previously paid instrumentation plus a 7.2 MB htmlcov rewrite;
  opt-in is `--cov=feature_forge --cov-report=term-missing`.
- Formalized the tier split procedurally instead of via markers: the declared
  marker taxonomy (`slow`/`integration`/`llm`/…) carries zero tests, and the
  directory boundary already is the tier. Measured on this host: unit tier
  `uv run pytest tests/unit` = 1020 tests / 0 failures / ~2¾ min (quiet
  machine: ~2 min); the 87 integration tests are ~55% of suite time and pin
  Hamilton reuse/recovery/integrity, so the full `uv run pytest` stays the
  pre-handoff gate (AGENTS.md and README now document both tiers).
- Fixed a flaky-by-construction pre-existing test surfaced while measuring:
  `TestSandboxedExecutor::test_valid_code` allowed 2.0 s for a fresh spawn
  worker whose `import feature_forge` alone costs 1.4–2.1 s on this host —
  it failed 4/5 standalone. Budget raised to 10 s with a comment; timeout
  mechanics remain pinned by hanging-code tests (`test_sandbox_lifecycle.py`,
  plan-23 §7.11 module), so no acceptance coverage is lost.
- xdist spike (via `uv run --with pytest-xdist`, no dependency added):
  `-n 4` gives 1.9× (2m52s vs 5m30s sequential) but 3 leak-pin failures —
  cross-worker `/tmp` transients from other xdist processes outlive the 3 s
  settle window once concurrent sandbox workers may hold files for up to 10 s.
  Conclusion: not drop-in; adoption later would require serialized sandbox
  tests or per-test temp dirs. Not adopted now.
- Full suite after the changes: 1101 passed / 9 expected skips / 6 xfailed,
  zero XPASS; all AGENTS.md gates green (ruff, format, mypy src+tests,
  hygiene, docs refs, uv lock).
- AI assistance: investigation, timing analysis, and changes by the session
  agent (pi) at the maintainer's request; no dependency or supplied-data
  changes.

### 2026-09-18 — Plan 23 PR 5: sandbox containment and bounded worker lifecycle (ADR 0019)

- Landed plan 23 PR 5, implementing ADR 0019 literally across three
  sequenced slices. **AST I/O defense first** (decision 4): the canonical
  static policy now rejects file-capable NumPy/Pandas APIs —
  `np.load/save/savez/fromfile/memmap/loadtxt/genfromtxt/fromregex/
  DataSource/open_memmap` and the full pandas `read_*`/`to_*` families —
  matching normalized attribute paths **and** terminal names so
  `import numpy as n2; n2.fromfile(...)` and `np.lib.format.open_memmap`
  cannot alias past the rule; `from numpy import load` imports are rejected
  too. Ordinary numeric/stringframe operations (`str.find`, `to_numpy`)
  stay allowed. Worker runtime guards monkeypatch the blocked surfaces as
  defense in depth (never described as isolation).
- **Strict OS enforcement** (decisions 2–3): raw-ctypes Landlock (no new
  dependency — the pre-sanctioned python-landlock comparison was moot since
  the proven ctypes path from the spike worked unmodified). The worker loads
  trusted runtime modules and the input frame **before** restriction, then
  applies an ABI-masked ruleset (ABI 7 on this host) granting read on exactly
  the predeclared input inode + narrowly required runtime roots
  (`sys.prefix`, stdlib, numpy/pandas trees — not broad `/usr`/`/etc`),
  write/truncate on the predeclared output inode, an **empty-allow TCP
  bind/connect policy** (ABI ≥ 4) so the "strict" label is never granted
  without socket denial, `no_new_privs` before `landlock_restrict_self`,
  and a raw classic-BPF seccomp socket-denial filter for ABI < 4.
  `test_execute_cannot_read_external_sentinel` passes via AST rejection,
  while `tests/unit/test_sandbox_containment.py` proves the kernel boundary
  directly (forked child under the real ruleset: sentinel read EACCES,
  127.0.0.1:9 connect EACCES, granted output still writable).
- **Profiles fail closed**: `SandboxProfile` (`strict` default |
  `degraded_development`) is a typed evaluation setting threaded through all
  production constructors (`kit.py`, `core.py`, `hamilton_executor.py`);
  strict re-probes at runtime in parent and worker and raises
  `SandboxContainmentError` when Landlock/seccomp is unavailable (explicit
  fail-closed production test); degraded execution records itself in
  `EnvironmentSnapshot.sandbox_profile/sandbox_degraded`, tracker config,
  and every sandbox log event (`sandbox_execute_start/complete` carry
  `sandbox_profile/degraded/mechanism/landlock_abi`). Worker env scrubbing
  uses an explicit allowlist (locale + thread-cap vars only) so provider
  secrets never reach generated code.
- **Bounded lifecycle** (decision 5): `SandboxedExecutor.execute_async` starts
  the child with `Process.start()` directly on the event-loop thread (pinned
  by a thread-identity spy test) and polls the response queue nonblockingly;
  sync `execute` and async share one `_WorkerHandle` cleanup helper; one
  monotonic deadline per execution, logged with its effective timeout;
  timeout/cancellation kills the worker's own process group (worker calls
  `setpgid(0,0)`; parent verifies group identity before `killpg` SIGKILL),
  joins boundedly, escalates, closes queues, and unlinks both temp files.
  `methods/malmas/pipeline/core._exec_sandbox` awaits `execute_async`
  directly — no `asyncio.to_thread`, no competing outer `wait_for`.
- **Worker-feed regression found and fixed during multi-version
  qualification**: under single-thread BLAS/OpenMP env the worker initially
  deadlocked — two independent causes, both diagnosed from `/proc` VSZ
  sampling: (1) the worker's structlog INFO event lazily imported
  OpenTelemetry inside the RLIMIT_AS-capped child, exhausting address space
  so the response queue's feeder thread died with `RuntimeError: can't start
  new thread`; the worker now never emits structlog events — containment is
  reported in the response tuple and logged by the parent. (2) the env scrub
  cleared thread-cap vars, making BLAS/OpenMP default to one pool per core
  (+0.5–1.3 GB VSZ) and pinning the worker at the 2048 MB RLIMIT_AS ceiling;
  the allowlist scrub above preserves the caps. Both fixes verified with and
  without the single-thread env on Python 3.11/3.12/3.13.
- Tests: removed exactly the six PR-5 `xfail(strict=True)` markers (which
  parametrize to 17 test instances — hence the suite moves from 23 to **6**
  xfailed, not the 17 the handoff projected; the remaining six all belong to
  the PR-6 LLM-cache module, zero XPASS). Two same-change fixture
  adjustments, both preserving pinned intent (PR-4 precedent):
  `test_to_thread_cancellation_cannot_stop_sandbox_thread` became
  `test_exec_sandbox_cancellation_stops_sandbox_work` with an
  `execute_async`-blocking double (post-fix `_exec_sandbox` never uses a
  thread; the test still proves outer-`wait_for` cancellation stops sandbox
  work promptly), and the async-entry test scopes `mp.active_children()` to
  children started during the call (process-global assertion collided with
  joblib/Loky workers from earlier tests in full-suite runs). New suites:
  `tests/unit/test_sandbox_io_defense.py` (11),
  `tests/unit/test_sandbox_containment.py` (4),
  `tests/unit/test_sandbox_lifecycle.py` (5). Suite: **1101 passed /
  9 expected skips / 6 xfailed, zero XPASS**; all AGENTS.md + plan §8 gates
  green (ruff, format, mypy src+tests, hygiene, docs refs, stage DAGs
  unchanged, mkdocs strict, uv lock, git diff --check). Multi-version
  evidence: sandbox modules + the full integration directory
  (155 tests, 0 failures, 1 optional-xgboost skip) on Python 3.11, 3.12,
  and 3.13 with single-thread BLAS/OpenMP (OMP/OPENBLAS/MKL/NUMEXPR = 1),
  junit XML kept at `/tmp/junit-3.{11,12,13}.xml` during the session.
- Reviewer pass (independent read-only subagent over the full PR-5 diff)
  returned REQUEST-CHANGES with design confirmed ADR-aligned; findings fixed
  in this change: worker env allowlist narrowed to locale + thread/allocation
  caps exactly (HOME/PATH/TMPDIR/PYTHONDONTWRITEBYTECODE dropped; exact-key
  test asserts the retained set); every `_start_worker` phase boundary
  (including post-spawn) is now deadline-checkpointed with the unbounded
  fork/exec limitation documented in place; cleanup provably terminates the
  worker (post-SIGKILL bounded reap loop, `sandbox_worker_terminate_failed`
  error if still alive) and retries/warns on temp-file unlink failures instead
  of swallowing them; an `_ACTIVE_HANDLES` registry makes queue/process cleanup
  observably complete (leak tests assert it empties); error payloads are
  truncated to 300 chars and worker log fields to 100 so exception text and
  caller-controlled `source`/`agent_name` cannot flood logs; strict provenance
  reports mechanism `unavailable` (never a mechanism it cannot install) while
  degraded honestly reports `process_only_ast`; and the Landlock path-beneath
  attr now packs the exact 12-byte `__attribute__((packed))` UAPI layout
  (`u64 allowed_access; s32 parent_fd`) instead of relying on two u64s. The
  sync `core._exec_sandbox` fallback and the classic-BPF program were
  confirmed correct as-is.
- **Shared-/tmp flake found during multi-version validation**: running two
  pytest processes concurrently (full suite + version matrix) made the leak
  pins intermittently fail because the scan of the shared temp directory
  caught the *other* run's transient `ff_sandbox_*`/`feature_forge_input_*`
  files inside the before→after window. The leak assertions in
  `test_plan23_sandbox_hardening.py` and `test_sandbox_lifecycle.py` now
  settle within a bounded 3 s window (genuine leaks persist and still fail;
  concurrent-run transients vanish); verified with three deliberately
  concurrent pytest rounds plus sequential 3.11/3.12/3.13 runs.
- Carry-forward for the maintainer (reported, not changed — decision point 3
  of the PR-5 handoff): Platinum materialization still lacks an
  `existing_package` guard and `LocalArtifactStore.commit` atomically
  replaces a namespace directory, so recomputation at the same run_id can
  overwrite a previously committed package.
- AI assistance: three sequenced implementation slices delegated to
  glm-5.3-flash/gpt-5.6-luna worker subagents (AST I/O policy; Landlock
  strict enforcement + profiles; bounded async lifecycle), integrated by
  the session agent (which owns marker removals, the two fixture
  adjustments, the worker-feed regression diagnosis/fix, and the gates);
  independent reviewer pass recorded above (REQUEST-CHANGES findings fixed,
  re-reviewed by the session agent). Pending maintainer review.
  Links:
  `docs/plan/23_evaluation_integrity_security_hardening.md` §5–§6 PR 5,
  `docs/decisions/0019-sandbox-containment-bounded-lifecycle.md`,
  `docs/handoffs/2026-09-18-plan23-pr5-sandbox-containment.md`.

### 2026-09-18 — Plan 23 PR 4: Platinum v2 evidence and uncertainty (ADR 0018 decisions 7–8)

- Landed plan 23 PR 4. Directional Student-t intervals
  (`dataflows/platinum.py::_t_critical`, scipy now a **direct dependency**
  per ADR 0018 decision 7 and maintainer acceptance 2026-09-16):
  `uncertainty_summary` sign-adjusts raw fold deltas by `metric_direction`,
  and every statistic (`mean_directional_gain`, std, se, bounds) derives
  from the directional deltas with the two-sided Student-t critical value
  for `uncertainty_policy.confidence_level` and `pair_count - 1` df
  (pinned `t* = 4.302652729911275` for 95%/df=2). The greedy gate's
  `_directional_lower_bound` uses the same `_t_critical` margin (fail-closed
  `-inf` below two paired folds), so the margin is owned in one place.
- Evidence schema v2 (ADR 0018 decision 8): Platinum packages persist, next
  to the eight v1 artifacts, `evidence.json` (typed `PlatinumEvidenceIndex`
  with `evidence_schema_version: "2"` — per-artifact schema versions stay
  `"1"`), `discovery_fold_metrics.parquet` (per-candidate + greedy-step +
  discovery-baseline arms, arm-labeled), `selection_steps.json` (the frozen
  greedy transcript incl. per-step rankings and lower bounds), and
  `preprocessing.json` (per-arm per-fold `FoldPreprocessor.specification()`);
  `PLATINUM_REQUIRED_ARTIFACTS` is the live 12-artifact contract. The v2
  greedy transcript also documents that
  `PlatinumSelectionDecision.lower_bound` carries the evaluation-aggregate
  interval, while the discovery bounds that gated selection persist in the
  steps (ADR 0018 decision-5 reconstruction is now offline-reproducible).
- Offline reconstruction + tamper detection (plan §7.9): `load_platinum_package`
  reconstructs from persisted evidence alone — discovery partition scoping,
  selected-set membership (steps ⟷ evidence.json ⟷ decisions), the full
  directional interval, aggregate directional/legacy gains, every greedy
  step's ranking gains and chosen fold scores against the persisted discovery
  arms, decision interval bounds, and preprocessing fold coverage — failing
  closed with `DatasetError` on any mismatch (byte tampering already fails
  per-artifact SHA-256 verification). v1 packages (no `evidence.json`) load
  with unchanged v1 semantics.
- v1 reuse rejection (maintainer decision 2026-09-16): `PLATINUM_IDENTITY_SCHEMA_VERSION
  = "2"` enters `platinum_input_fingerprint` as a keyword input — every
  Platinum fingerprint changes exactly once, so pre-ADR v1 packages stay
  integrity-verified and loadable but can never satisfy a v2 reuse
  fingerprint; nothing is recomputed or rewritten in place (ADR 0014), and
  Bronze/Silver/Gold identities are untouched (Platinum-only charter). The
  manifest options now record `platinum_identity_schema_version`.
- Result/report fields: `ExperimentResult` gains additive
  `directional_gain`/`gain_lower_bound`/`gain_upper_bound`/
  `evaluation_protocol` (executor populates them from the loaded package;
  `gain` stays as legacy raw gain, no longer the headline); tracker runs gain
  `evaluation_protocol` + `selection_biased` config and the three directional
  metrics — compatibility-protocol results stay explicitly labeled
  selection-biased. `docs/operations.md` documents the v2 evidence set, the
  reconstruction guarantees, and the v1 read-only stance.
- Tests: removed exactly the three PR-4 `test_interval_*` xfail markers
  (26 → 23 xfailed, zero XPASS). One fixture fix was required:
  `test_interval_is_directional_for_minimizing_metrics` pinned
  `[-3.0, -1.0, -2.0]` deltas whose 95%/df=2 Student-t lower bound is
  negative (−0.484) — arithmetically unsatisfiable under the correct
  interval — so the fixture moved to `[-3.0, -2.5, -2.0]` (lower ≈ 1.258)
  preserving the test's directional intent. New suites:
  `tests/unit/test_platinum_uncertainty.py` (6),
  `tests/unit/test_platinum_evidence_v2.py` (9, incl. byte + hash-consistent
  semantic tamper — uncertainty bounds, selected set, decision bounds — and
  v1 read-compat),
  `tests/unit/test_platinum_reuse_and_results.py` (7, incl. store-level v1
  rejection + v2 acceptance, executor field population, tracker rendering).
  Result-row key-set pins in `test_execution_policy.py` and
  `test_fail_fast_sequential.py` extended with the four additive keys. Suite:
  **1064 passed / 9 expected skips / 23 xfailed, zero XPASS**; all AGENTS.md
  + plan §8 gates green (ruff, format, mypy src+tests, hygiene, docs refs,
  stage DAGs unchanged (`--check` ok), mkdocs strict, uv lock, git diff
  --check).
- AI assistance: three sequenced implementation slices delegated to
  glm-5.3-flash worker subagents (Student-t intervals + scipy dep; evidence
  schema v2 persistence + offline reconstruction; identity version +
  result/report fields), integrated by the session agent (which owns the
  marker removals, the fixture arithmetic fix, the reviewer follow-ups, and
  the gates), then an independent reviewer-subagent pass over the full diff
  (verdict APPROVE; its two actionable findings — decision interval bounds
  unverified in reconstruction, a misleading dead fixture override — were
  fixed; its pre-existing write-once observation about Platinum lacking an
  `existing_package` guard is carried into the PR-5 handoff). Pending
  maintainer review. Links:
  `docs/plan/23_evaluation_integrity_security_hardening.md` §6 PR 4,
  `docs/decisions/0018-independent-discovery-evaluation-protocol.md`,
  `docs/handoffs/2026-09-16-plan23-pr4-platinum-v2-evidence-uncertainty.md`.

### 2026-09-16 — Plan 23 PR 3: fold-local preprocessing and selection (ADR 0018 decisions 4–6)

- Landed plan 23 PR 3. New
  `evaluation/preprocessing.py::FoldPreprocessor` replaces Platinum's
  whole-frame `_prepared`: per fold, numeric median imputation and
  ordinal first-appearance categorical encoding are fitted on that fold's
  training rows only; validation rows transform with train-only statistics
  and unseen/missing categories map to the documented stable sentinel
  `-1`; fitted specifications are retained per fold and flow into the
  Platinum manifest's `evaluation_provenance` (durable persistence is PR
  4's evidence schema v2).
- Partition-aware Platinum evidence (ADR 0018 decisions 3/6):
  `candidate_fold_evidence` scores candidates and runs greedy selection on
  discovery-partition folds only (`CandidateFoldEvidence` carries the
  frozen `GreedySelectionOutcome` plus the post-freeze selected arm, so
  selection consumers can never read evaluation-fold scores before the
  freeze — pinned by the plan §7.2 spy test); `baseline_fold_evidence` and
  the combined selected arm are evaluated on evaluation-partition folds
  only; when no candidate qualifies the enhanced arm mirrors baseline with
  zero gain — a rejected numerical winner is never reported.
- Greedy forward selection (ADR 0018 decision 5): each remaining candidate
  is evaluated appended to the selected set, ranked by mean directional
  gain with a stable feature-name tie-break; `minimum_practical_gain`,
  `require_positive_lower_bound` (against the directional lower confidence
  bound — the margin stays the existing 1.96 normal approximation; PR 4
  swaps in Student-t, which only tightens it), and `max_selected_features`
  are all enforced; `PlatinumRequest` validation rejects
  `selection_partition="evaluation"` always and anything but `"discovery"`
  under holdout. Fold metrics gain an additive `partition` column and stay
  loadable by `load_platinum_package` (no schema v2 work). No fingerprint
  inputs changed; the change is independently revertible.
- Tests: removed exactly the eight PR-3 xfail markers in
  `tests/unit/test_plan23_evaluation_integrity.py` (the three PR-4 interval
  tests remain xfail-strict), extended the `max_selected_features` pin into
  real greedy-cap coverage (cap=1/2/uncapped + determinism), and added the
  stable-tie-break test and the §7.2 selection-freeze spy; new
  `tests/unit/test_fold_preprocessing.py` (8). Suite: 1039 passed / 9
  expected skips / 26 xfailed, zero XPASS; all AGENTS.md + plan §8 gates
  green (ruff, format, mypy src+tests, hygiene, docs refs, stage DAGs
  regenerated, mkdocs strict, uv lock, git diff --check).
- AI assistance: three sequenced implementation slices delegated to
  glm-5.3-flash worker subagents (fold-local preprocessing module;
  partition-aware platinum evidence; greedy selection + policy validation),
  integrated by the session agent (which owns the marker removals, the
  spy/tie-break tests, and the gates), then an independent reviewer-
  subagent pass over the full diff (its findings — datetime categories
  breaking `specification()` JSON safety, imprecise ADR citation in
  validator messages, and two undocumented latent fallback paths — were
  all fixed or documented). Pending maintainer review. Links:
  `docs/plan/23_evaluation_integrity_security_hardening.md` §6 PR 3,
  `docs/decisions/0018-independent-discovery-evaluation-protocol.md`,
  `docs/handoffs/2026-09-16-plan23-pr4-platinum-v2-evidence-uncertainty.md`.

### 2026-09-16 — Plan 23 PR 2: partition-aware discovery (ADR 0018 decisions 1–3)

- Landed plan 23 PR 2. `Settings.evaluation` gains `protocol`
  (`holdout`|`compatibility`, default `holdout`) and
  `evaluation_holdout_fraction` (default `0.25`, `0 < f < 0.5`);
  `DatasetComputationRequest` validates coherent protocol/fraction
  combinations (compatibility ⟺ fraction 0.0), so the legacy all-row profile
  is selectable only explicitly.
- Fail-closed partitioning: `resolve_partition` raises an actionable
  `DatasetError` when either partition cannot independently host `cv_folds`
  rows; the executor no longer hard-codes `evaluation_holdout_fraction=0.0`
  (`_dataset_request` threads the configured protocol/fraction). Silver
  fingerprints/manifests, Gold requests/fingerprints, and Platinum evaluation
  policies all carry the protocol, so reuse chains distinguish protocols and
  old packages are invalidated by fingerprint, never deleted. Note: pre-ADR
  Gold/Platinum packages now fail fingerprint validation on load (offline
  replay of old packages requires recomputation) — intended reuse
  invalidation per plan §9.
- Discovery boundary: methods are fit only on discovery-partition rows
  (`_fit_method_on_discovery`); evaluation rows/targets never reach method
  fitting. New row-local scope contract (`evaluation/scope.py`): under
  holdout, candidates must pass deterministic row-subset/permutation
  metamorphic probes before acceptance; detected whole-frame/position-
  dependent candidates are rejected (`not_row_local`) and never enter the
  working frame; Gold replay mirrors generation (accepted-only columns).
  Compatibility replays legacy whole-frame scripts unchanged.
- Tests: removed exactly the three PR-2 markers (the executor source pin
  became a behavioral test), added fail-closed/behavioral/spy/fingerprint
  tests plus `tests/unit/test_scope_contract.py` (9) and two executor
  call-site wiring drills; e2e fixtures grew to 24 rows so the default
  holdout protocol can host stratified folds in both partitions (small data
  fails closed by design). Suite: 1021 passed / 9 expected skips / 34
  xfailed, zero XPASS; all AGENTS.md gates green (ruff, mypy src+tests,
  hygiene, docs refs, stage DAGs, mkdocs strict, uv lock, git diff --check).
- AI assistance: three disjoint implementation slices delegated to
  glm-5.3-flash worker subagents (settings/fail-closed partitioning;
  executor/contract fingerprint threading; scope contract), integrated by
  the session agent, then an independent reviewer-subagent pass (its
  findings — a broken legacy request construction in
  `scripts/qualify_hamilton_cache.py`, inaccurate old-package-loadability
  comments, PR-3-bound docstring overclaims, and missing executor wiring
  tests — were all fixed). Pending maintainer review. Links:
  `docs/plan/23_evaluation_integrity_security_hardening.md` §6 PR 2,
  `docs/decisions/0018-independent-discovery-evaluation-protocol.md`,
  `docs/handoffs/2026-09-16-plan23-pr3-fold-local-preprocessing.md`.

### 2026-09-15 — ADRs 0018–0020 accepted; plan 23 PR 2 unlocked

- The maintainer accepted ADR 0018 (independent discovery/evaluation
  protocol), ADR 0019 (sandbox containment and bounded lifecycle), and ADR
  0020 (LLM cache identity v2); statuses and the ADR index were flipped to
  Accepted and plan 23 is now Active.
- PR-1 handoff closed and superseded by the PR-2 handoff
  (`docs/handoffs/2026-09-15-plan23-pr2-partition-aware-discovery.md`),
  which assigns plan 23 PR 2 only (partition-aware discovery) and maps the
  PR-1 xfail markers each later PR must remove.
- AI assistance: bookkeeping by the session agent; no runtime changes.

### 2026-09-15 — Plan 23 PR 1: characterization tests, isolation spike, proposed ADRs 0018–0020

- Landed plan 23 PR 1 with **no runtime behavior changes**: 40
  characterization tests across
  `tests/unit/test_plan23_sandbox_hardening.py` (18),
  `tests/unit/test_plan23_evaluation_integrity.py` (15), and
  `tests/unit/test_plan23_llm_cache_identity.py` (7) encode all nine
  confirmed audit findings — 37 `xfail(strict=True)` desired-behavior
  tests plus 3 regression pins for currently-correct behavior (timeout
  cleanup, key determinism, single-best capping); the markers are removed —
  and the finding-9 seam tests rewritten against the new async entry point —
  by the PR that fixes each finding. Notable nuance: in one observed run the
  worker surfaced `np.fromfile(path)` as a `CodeExecutionError` (mechanism
  unconfirmed — possibly numpy/python interaction under restricted
  builtins); either way the AST-policy gap and the required
  `SandboxValidationError` contract remain demonstrated. Re-confirm the
  observed failure mode during PR 5.
- Time-boxed OS-isolation spike (Linux 6.18/WSL2, this host): **Landlock
  ABI 7**, unprivileged user namespaces, seccomp, and `bwrap` available; a
  live Landlock ruleset denied an out-of-root sentinel read (`EACCES`) while
  imports and temp-root writes kept working. Host/LSM variability remains an
  open question for PR 5. Evidence:
  `experiments/sandbox_isolation_spike/2026-09-15/report.json`; record:
  `docs/spikes/2026-09-15-sandbox-os-isolation.md`.
- Drafted **ADRs 0018–0020** (independent discovery/evaluation protocol,
  sandbox containment and bounded lifecycle, LLM cache identity v2) as
  Proposed and indexed them. PRs 2–7 are gated on maintainer acceptance.
- AI assistance: implementation delegated to subagents (glm-5.3-flash
  workers) for the three test modules and the spike; ADRs and integration
  by the session agent; independent review pending. Links:
  `docs/plan/23_evaluation_integrity_security_hardening.md`,
  `docs/decisions/0018-…`, `0019-…`, `0020-…`.

### 2026-09-15 — Plan 23 evaluation-integrity and security-remediation handoff

- Converted the independent audit findings into an agent-ready seven-change
  plan covering discovery/evaluation isolation, fold-local preprocessing,
  enforceable selection, Platinum v2 evidence, directional uncertainty,
  sandbox containment and bounded lifecycle, and LLM cache identity v2.
- Runtime work is gated on proposed ADRs 0018–0020. Added a continuation handoff
  whose exact next assignment is characterization tests, an OS-isolation spike,
  and ADR drafting; no runtime behavior or schema was changed.
- AI assistance: plan and handoff drafted by Codex with independent evaluation
  and security review agents. Links:
  `docs/plan/23_evaluation_integrity_security_hardening.md`,
  `docs/handoffs/2026-09-15-evaluation-security-audit-remediation.md`.

### 2026-09-14 — Independent project audit

- Found that generated features are fitted on all Silver rows and targets before
  Platinum cross-validation, while the platform sets the evaluation holdout
  fraction to zero. Platinum also imputes/encodes the full frame before fold
  splitting. Reported scores therefore are not independent held-out estimates.
- Platinum selects its best candidate on the same folds used for reporting;
  configured selection partition, positive-lower-bound requirement, and maximum
  feature count are not enforced. The reported enhanced arm uses the best
  candidate even when the selection decision rejects it.
- The sandbox AST policy accepts direct NumPy filesystem reads; its child
  process runs with the host user's filesystem permissions. LLM cache identity
  omits provider endpoint and DeepSeek thinking settings, permitting stale
  responses across behaviorally different requests. The paired uncertainty
  interval uses a fixed 1.96 multiplier and raw rather than directional deltas.
- `.env` was tracked despite the README and `.gitignore` contract. The committed
  form was dotenvx-encrypted; it was removed from tracking while preserving the
  local file, and `check_repo_hygiene.py` now rejects tracked `.env` and dotenvx
  key files.
- Audit validation: ruff check/format, mypy src, hygiene, docs references,
  `git diff --check`, and focused Platinum/cache/execution/sandbox tests passed.
  The full suite did not pass in this sandbox: the isolated
  `TestCorePipeline::test_run_with_fake_agent` timed out in sandbox execution
  and failed after about 30 seconds, including with single-thread BLAS limits.
- AI assistance: findings and this audit record prepared by Codex; no runtime
  changes made. Links: `src/feature_forge/experiment/hamilton_executor.py`,
  `src/feature_forge/dataflows/platinum.py`,
  `src/feature_forge/evaluation/sandbox.py`,
  `src/feature_forge/llm/base.py`.

### 2026-09-14 — Plan 22 final validation sweep and audit corrections

- The post-implementation audit (reviewer subagent) verified plan 22 §2–§9
  conformance; no blocking defects. Audit follow-ups landed: mypy-strict
  typing fixed in the new test modules (9 errors), the Python 3.12+
  multi-threaded-fork `DeprecationWarning` filtered with rationale on the
  mandated fork-context tests, the missing §6 test 4 process-path mid-run
  token test added (deterministic manager-event choreography, both mp
  contexts), `max_workers >= 1` validated at the adapter boundary, the
  progress bar now closes at its total including cancelled rows, and
  `docs/operations.md` documents that `error`-based aggregations classify
  cancelled rows as failed runs (prefer the `state` field).
- Corrected the PR 3 worker's misreported suite count (1007 → true 998 at
  the time; the PR worker had double-counted the 9 skips) in
  `.planning/STATE.md` and the continuation handoff.
- Final validation: full suite **1000 passed / 9 expected optional-XGBoost
  skips on Python 3.11, 3.12, and 3.13** under single-thread BLAS/OpenMP
  limits; `mypy src` + `mypy tests`, ruff check/format, repo hygiene, docs
  references, stage-DAG freshness, mkdocs strict, `uv lock --check`, and
  `git diff --check` all green. Plan 22 §9 is complete; the retry gap
  (`ResourceConfig.max_attempts`) remains explicitly open.
- AI assistance: audit follow-ups and corrections implemented by AI
  assistance (pi coding agent; reviewer subagent audit) within the
  maintainer-accepted ADR 0017 scope; final review pending.
- Links: `docs/plan/22_fail_fast_cancellation_contract.md`; ADR 0017;
  `docs/handoffs/2026-09-14-fail-fast-cancellation-complete.md`.

### 2026-09-14 — Plan 22 PR 4: operator documentation and completion audit (ADR 0017)

- PR 4 of `docs/plan/22_fail_fast_cancellation_contract.md` closed the final
slice: operator documentation and the plan 21/22 completion audit.
`docs/operations.md` gained an "Execution failure policy and cancellation"
section — settings/env/per-run override forms (`failure_policy=...` wins for
the invocation, recorded in tracker provenance, never mutates cached
Settings), cooperative `CancellationToken` example (parent-process-only,
thread-safe, first-reason-wins), the explicit fail-fast vs. hard-termination
distinction (bounded window ≤ `max_workers` may complete after the stop
point; running cases finish, never killed), KeyboardInterrupt re-raise
semantics, the additive result `state` field with exact one-ordered-row-per-
case cardinality and score/stage-free cancelled rows, resume-from-verified-
prefix note, and the explicit dormant-retry gap. `FF_EXECUTION__FAILURE_POLICY`
was added to the operations config table.
- `docs/api_reference.md` documents the `run()` keyword params (precedence +
no-mutation guarantee), `ExperimentResult.state`/`resolved_state`, the
`CancellationToken` API (`feature_forge.experiment.execution`), the redacted
cancelled `FailureRecord` shape (`failure_class=cancelled`, stable error type
`CaseCancelled`), and the `ProcessPoolExecutionAdapter` bounded submission
window. `docs/migration_guide.md`'s failure-policy subsection now states
runtime enforcement has landed (per-run override, token, additive `state`; no
migration action required). `README.md`, `docs/index.md`, and the
`config/settings.yaml` `execution:` comment carry brief pointers.
- Completion audit (plan 21 claims audited literally; full evidence in the
table below). Forward-looking retry/cancellation overstates were corrected
with minimal `[Audited 2026-09-14: ...]` notes; no gate evidence was deleted.
The plan index (Current Phase / Next Steps) and `.planning/STATE.md` now mark
plan 22 PRs 1–4 implemented pending only final validation and keep the
dormant `ResourceConfig.max_attempts` retry gap visible.

| Plan 21 claim | Verdict | Action taken |
|---|---|---|
| §4 invariant: platform owns scheduling, process boundaries, tracking, cancellation, failure handling | VERIFIED (cooperative case-boundary cancellation implemented, plan 22 PRs 2–3) | none |
| §7.1/§7.4/§7.5 case/attempt identity, `plan_case`/`execute_case`, incremental algorithm | VERIFIED (`src/feature_forge/experiment/hamilton_executor.py:191` plan_case is registry-metadata-only; platform allocates identity before scheduling) | none |
| §9.3: `plan_case()`/CLI dry-run construct no driver, open no cache backend | VERIFIED (`tests/unit/test_cli.py::test_run_plan_json_is_one_document_and_side_effect_free`; plan_case reads metadata only) | none |
| §11: journal records case/stage/node/cache/**retry**/cancellation/terminal events | OVERSTATED ("retry": no retry events exist — case-level retry unimplemented) | removed "retry," + `[Audited 2026-09-14]` note; cancellation events verified as redacted `case_cancelled` |
| §11: KeyboardInterrupt/SystemExit propagate; input order preserved | VERIFIED (plan 22 §2.6 hygiene in `src/feature_forge/experiment/execution.py`; index-keyed reassembly) | none |
| §12 PR 4 deliverable 5: "wire ... retries, cancellation ..." | OVERSTATED as shipped-feature implication (retry never wired; cancellation landed later via plan 22) | historical text kept; `[Audited 2026-09-14]` scope note appended |
| §14.2: integration coverage "fail-fast, cancellation, retry, partial resume" | OVERSTATED (retry coverage does not exist) | `[Audited 2026-09-14]` note appended citing fail-fast/cancellation/resume test files |
| §16 legacy-removal gates 1–6 (incl. 954-passed three-version suites, parity, replay drills) | HISTORICAL (past gate evidence; accurate as history) | none (evidence preserved) |
| §6.1/§12 PR 1 transitional `legacy` engine default/config examples | HISTORICAL (intermediate-state descriptions superseded by ADR 0016; doc header + §0 already record removal) | none |
| §18: "dry-run, resume, replay, retry, cancellation, sequential, process have end-to-end coverage" | OVERSTATED for retry | retry removed from the unconditional claim + `[Audited 2026-09-14]` note marking it explicitly open |

- Continuation handoff: `docs/handoffs/2026-09-14-fail-fast-cancellation-complete.md`
(summary of PRs 1–4, key files, validation evidence, confirmed decisions,
next steps including the Python 3.11/3.12/3.13 + both-mp-contexts
re-validation note).
- AI assistance: implemented by an AI coding agent (worker subagent via pi)
under maintainer direction; ADR 0017 + plan 22 PR 4 scope.
- Links: `docs/operations.md`, `docs/api_reference.md`,
`docs/migration_guide.md`, `docs/plan/21_hamilton_default_execution_handoff.md`,
`docs/plan/00_index.md`, `.planning/STATE.md`,
`docs/decisions/0017-failure-policy-and-cancellation-contract.md`,
`docs/plan/22_fail_fast_cancellation_contract.md`.

### 2026-09-14 — Plan 22 PR 3: bounded process scheduling and interruption hygiene (ADR 0017)

- PR 3 of `docs/plan/22_fail_fast_cancellation_contract.md` closed the PR 2
documented gap: `ProcessPoolExecutionAdapter` no longer eagerly submits the
full matrix. Submission is a bounded window of at most `max_workers` futures,
refilled only while the stop conditions permit — a parent-side
`stop_on_result` verdict per completed result (the platform wires fail-fast
as `resolved_state == FAILED`) plus an optional `CancellationToken` checked
before each submission and between completions. After a stop condition
nothing new is submitted; queued futures are cancelled with `Future.cancel()`
and, with the never-submitted tail, become typed CANCELLED rows built by a
platform-supplied `cancelled_result` factory that reuses the payload's
parent-allocated identity (no extra `plan_case` calls) and journals the same
redacted `case_cancelled` event as PR 2. Already-running workers finish and
keep their real results; original order/cardinality are preserved via
index-keyed reassembly; default calls (no hooks) keep today's semantics.
The executor is shut down and joined on every exit path (normal, fail-fast,
token-cancelled, exception, interrupt) — no worker-process leak.
- Parallel wiring: `run(parallel=True)` now passes the effective policy +
token into the adapter, so fail-fast/token semantics match the sequential
path (run() docstring caveat removed); cancelled process rows skip tracker
effects exactly like sequential ones.
- KeyboardInterrupt hygiene (plan 22 §2.6): neither scheduler swallows an
interrupt or converts it into a failed experiment. The process path stops
submission, cancels pending futures, joins the pool, journals redacted
cancellation for cases proven not to have started, and re-raises; the
sequential path journals the never-started remainder after an in-flight
`execute_case` interrupt and re-raises (in-flight cases stay governed by
their existing timeouts — no new kill mechanisms).
- Tests: new `tests/unit/test_fail_fast_process.py` (plan 22 §6 items 1, 5,
6, 7, 8/9 process slice, 11 + worker-leak sweeps, Linux-default and explicit
`spawn` contexts, top-level pickleable workers, `multiprocessing.Manager`
event synchronization — no timing sleeps) and real-chain sequential/process
fail-fast + continue parity tests in
`tests/integration/test_hamilton_execution_parity.py`. Pending-future
cancellation (item 6) is scripted at the future level: with a bounded window
the executor's feeder marks futures running before workers can lag, so a
parent-observable queued-not-started future is only deterministically
constructible there; the interrupted-collection test tolerates the startup
race where a queued case is cancelled-before-start under load (both outcomes
are truthful and asserted).
- AI assistance: implemented by an AI coding agent (Claude via pi) under
maintainer direction; ADR 0017 + plan 22 PR 3 scope.

### 2026-09-14 — Plan 22 PR 2: sequential fail-fast and cancellation semantics (ADR 0017 Accepted)

- ADR 0017 was accepted by the maintainer on 2026-09-14; PR 2 of
  `docs/plan/22_fail_fast_cancellation_contract.md` landed the SEQUENTIAL
  runtime slice. `ExperimentalPlatform.run()` gained keyword
  `failure_policy` / `cancellation_token` overrides resolved run-scoped
  (run override > `settings.execution.failure_policy`) into a deep-copied
  Settings snapshot — cached/global Settings are never mutated — and the
  effective policy is recorded in tracker provenance config.
- Sequential scheduling (plan 22 §2.4): the token is checked before each case
  starts; under `fail_fast` the first terminal failed result (derived
  `resolved_state == FAILED`) stops scheduling. Never-started cases are
  drained in matrix order into typed cancelled results: identity allocated
  by the parent via the read-only `plan_case` seam (with the worker's
  unresolved-case content-fingerprint fallback), no stage packages, no
  fabricated scores, explicit `state=cancelled`, `error` set to the redacted
  record message. A case in flight when cancellation arrives keeps its real
  result. Default continue behavior and result cardinality/order are
  unchanged.
- Parent lifecycle: one redacted `case_cancelled` event per unstarted case
  via the same `LocalRunRepository` path (`event_type=case_cancelled`,
  `state=cancelled`, typed `CaseCancelled` FailureRecord, case fingerprint,
  empty details — no exception text, prompts, inputs, or secrets). Tracker
  effects stay exactly-once for cases that actually started and never init a
  run for unstarted cases. Cancellation touches no cache/staging/package
  bytes (verified by digest snapshot in a real-chain integration test).
- Process path unchanged: eager submission remains until plan 22 PR 3
  (bounded submission window); retries and `ResourceConfig.max_attempts`
  remain untouched non-goals.
- Tests: new `tests/unit/test_fail_fast_sequential.py` (plan 22 §6 matrix
  items 1, 2, 3, 4, 8, 9, 10, 12 + policy resolution/provenance), a real-chain
  fail-fast resume/redaction integration test in
  `tests/integration/test_platform_e2e.py`, and a run()-plumbing test in
  `tests/unit/test_platform.py`.
- AI assistance: implemented by a worker subagent under the pi coding agent;
  pending maintainer review.
- Links: `docs/decisions/0017-failure-policy-and-cancellation-contract.md`;
  `docs/plan/22_fail_fast_cancellation_contract.md`.

### 2026-09-14 — Plan 22 PR 1: failure policy and cancellation contracts (ADR 0017 Proposed)

- PR 1 of `docs/plan/22_fail_fast_cancellation_contract.md` recorded ADR 0017
  (Status: Proposed — pending maintainer acceptance) and landed the typed
  contract slice only: `FailurePolicy` / `ExecutionPolicyConfig` with
  `Settings.execution` (env `FF_EXECUTION__FAILURE_POLICY`), the
  `config/settings.yaml` `execution:` section, `ExperimentResult.state` with
  `resolved_state` derivation serialized additively via `_result_to_dict`,
  the parent-process thread-safe `CancellationToken` (first-reason-wins), and
  the redacted cancelled `FailureRecord` factory (`CaseCancelled`).
  `RunState` reuses the existing `feature_forge.contracts.stages.RunState`
  vocabulary instead of introducing a duplicate enum.
- No runtime scheduling changes: sequential fail-fast, bounded process
  submission, `run()` policy/token wiring, and tracker/lifecycle behavior are
  plan 22 PRs 2-3 and proceed only after the maintainer accepts ADR 0017.
- Focused contract tests added in `tests/unit/test_execution_policy.py`;
  `CaseComputationInput` is guarded to stay free of token/policy fields.
- AI assistance: implemented by a worker subagent under the pi coding agent;
  pending maintainer acceptance of ADR 0017.
- Links: `docs/decisions/0017-failure-policy-and-cancellation-contract.md`;
  `docs/plan/22_fail_fast_cancellation_contract.md`.

### 2026-09-14 — ADR 0016 legacy-engine removal landed

- The dedicated removal change authorized by ADR 0016 is implemented. The
  imperative legacy execution path (`ExecutionEngine.LEGACY`, `_run_legacy`,
  and the `ExperimentCaseExecutor` wiring) and the `legacy` artifact-policy
  branch were removed; Hamilton is the sole supported case execution engine.
- Stale `engine=legacy` configuration (env `FF_DATAFLOW__ENGINE` or YAML
  `dataflow.engine`) and `artifact_policy: legacy` now fail fast with an
  actionable migration error pointing at `docs/migration_guide.md`; legacy
  configuration is never silently reinterpreted as Hamilton and engines are
  never mixed within one attempt.
- `scripts/qualify_legacy_removal.py` was deleted along with the engine it
  qualified. The historical Hamilton/legacy parity record is preserved at
  `experiments/legacy_removal/2026-09-14/`.
- Tests that required live dual-engine execution were migrated to Hamilton
  invariants. Operational rollback after removal is a package/version downgrade
  to the last compatibility release, not a runtime engine switch (ADR 0016
  item 6).
- Documentation updated for the landed removal: `README.md`, `docs/index.md`,
  `docs/README.md`, `docs/operations.md`, `docs/api_reference.md`,
  `docs/migration_guide.md`, `config/settings.yaml` comments,
  `docs/plan/00_index.md`, and `.planning/STATE.md`.
- AI assistance: the removal was implemented by AI assistance under the pi
  coding agent — parallel worker subagents owned the code/tests and docs/status
  slices, a reviewer subagent audited ADR 0016 conformance, and the primary
  agent verified the tree and ran the full validation suite (932 passed,
  9 expected optional-XGBoost skips). Pending maintainer review and acceptance.
- Links: ADR 0016; `experiments/legacy_removal/2026-09-14/`;
  `docs/migration_guide.md`; `docs/plan/22_fail_fast_cancellation_contract.md`.

### 2026-09-14 — Legacy removal accepted; fail-fast contract planned

- The maintainer accepted plan 21 gate 6. ADR 0016 supersedes ADR 0012's
  compatibility period and authorizes a dedicated removal of
  `ExecutionEngine.LEGACY`, `_run_legacy`, and the legacy artifact-policy branch.
  Removal is not implemented in this documentation-only decision slice; the
  current checkout still supports the rollback flag until that separate change.
- Rollback after removal will be package/version rollback, not silent fallback
  or a runtime engine switch. Stale legacy configuration must fail with
  actionable migration guidance.
- Added plan 22 for backward-compatible continue/fail-fast policy, cooperative
  case-boundary cancellation, explicit result state, ordered/cardinality-stable
  cancelled results, bounded process submission, redacted lifecycle events,
  tracker rules, and `KeyboardInterrupt` cleanup/re-raise semantics.
- Plan 22 deliberately excludes hard-killing running work and the dormant
  case-retry fields. Its first implementation PR must accept ADR 0017; legacy
  removal and scheduler changes must remain independently reviewable.
- AI assistance: the primary agent audited the execution/config/lifecycle
  contracts and produced ADR 0016 plus the agent-ready scheduler plan. A final
  subagent review was attempted but unavailable because its weekly quota was
  exhausted.
- Links: ADR 0016; `docs/plan/22_fail_fast_cancellation_contract.md`;
  `docs/plan/21_hamilton_default_execution_handoff.md`.

### 2026-09-14 — Legacy-removal technical qualification

- Completed plan 21 technical gates 1–5 while retaining `engine=legacy`.
  Full suites passed on Python 3.11, 3.12, and 3.13: 954 passed and 9 expected
  optional-XGBoost skips on each interpreter under single-thread BLAS/OpenMP
  limits.
- Added real recomputation parity under the Linux-default and explicit spawn
  process contexts. Hamilton/legacy metrics use a documented absolute tolerance
  of `1e-9`; non-degenerate sandbox decisions prevent baseline-only false passes.
- Added end-to-end cache deletion/corrupt-metadata recovery, hash-tampered Silver
  suffix recomputation, same-attempt resume, and provider-free Gold replay drills.
  Unreadable Hamilton SQLite metadata is quarantined and rebuilt; durable
  medallion packages remain authoritative and untouched.
- Ran the offline two-seed bundled breast-cancer matrix through both engines.
  All four Hamilton packages verified for each seed; baseline/cv/gain parity
  deltas were at most `3.71e-17`; 51 serialized cache results were present.
- A 100,000-row direct Bronze/Silver cache follow-up reached 10/10 eligible hits
  and measured 1,400.1 ms cold versus 327.0 ms warm (4.28x), with two source
  loads and zero provider/sandbox calls. This supersedes the tiny-fixture timing
  concern but is not a universal workload speedup claim.
- AI assistance: three scout agents audited the technical gates, matrix
  feasibility, and recovery gaps; three worker agents implemented isolated
  recovery, parity, and qualification slices. The primary agent reviewed,
  hardened corrupt-cache recovery, executed persistent evidence and the
  three-version full matrix, and integrated documentation.
- Remaining gate: a maintainer decision and ADR are required before removing
  `engine=legacy`. The definition-of-complete cancellation/fail-fast requirement
  also needs an explicit audit before closing the overall Hamilton program.
- Links: `experiments/legacy_removal/2026-09-14/qualification_report.json`;
  `experiments/legacy_removal/2026-09-14/cache_benchmark/report.json`;
  `docs/plan/21_hamilton_default_execution_handoff.md`.

### 2026-09-12 — Legacy correctness and dependency cleanup, PR 6

- Replaced MALMAS's object-identity baseline cache key with canonical full-data
  and evaluation identity (X/y values and schema, folds, metric, task, seed,
  and model). Added direction-aware selection so lower RMSE/MAE/NRMSE deltas
  are correctly treated as improvements.
- Deduplicated generated code before sandbox execution and retained only batches
  that succeed train schema checks. The same successful set executes once on
  test data; index/name/dtype failures are reported per feature and failed test
  features cannot be selected or replayed. MALMAS `fit_transform` returns its
  cached enhanced training frame instead of executing generated code twice.
- Renamed the per-round cap to `max_selected_features` with a migration-only
  `min_effective` input alias. Unknown MALMAS modes and agents now fail before
  provider construction. Normal fits reset memory/router learning, while
  `warm_start=True` explicitly enables persisted-memory reuse.
- Changed the standard model default to scikit-learn random forest. OpenFE,
  CAAFE fidelity, XGBoost, LightGBM, and CatBoost now live behind named extras
  and emit precise `feature-forge[...]` install instructions. Added a clean
  wheel package-contract CI lane and verified a local core-only wheel install.
- Validation: `uv sync --all-groups --extra intel` passed; full suite 922 passed
  and 9 optional-XGBoost tests skipped; Ruff, formatting, strict source/test mypy, lock,
  hygiene, docs references, and diff checks passed. The first full run exposed
  a SQLite WAL initialization race; bounded retry fixed it and five repeated
  spawned-worker cache tests passed.
- AI assistance: a subagent implemented the isolated dependency/package-doc
  slice and ran its focused/full lanes; another audited the reference branch.
  The primary agent integrated, extended MALMAS correctness, fixed findings,
  and performed final validation. A final reviewer subagent was unavailable due
  to its usage quota.
- Links: `docs/plan/21_hamilton_default_execution_handoff.md`;
  `docs/migration_guide.md`.

### 2026-09-11 — Hamilton operations, telemetry, and documentation, PR 5

- Added bounded, redacted Hamilton node/cache telemetry with layer tags,
  hit/miss/error outcomes, duration, and serialized-byte metadata in each
  attempt lifecycle journal. Node arguments, results, exception text, prompts,
  feature values, and secrets never enter telemetry.
- Added WAL-backed concurrent Hamilton metadata, atomic result publication,
  cache status/inspection, age/size retention, guarded clear, and artifact
  list/semantic verification CLI commands. GC is scoped to the Hamilton cache
  root and tests prove medallion packages and the independent mandatory LLM
  DiskCache remain untouched. Attempt IDs are now validated before journal path
  construction.
- Added deterministic generated Bronze/Silver/Gold/Platinum Mermaid DAG docs,
  freshness checks, operator recovery guidance, shipped-API example cleanup,
  tracker-default corrections, strict MkDocs support, and minimum/latest
  Hamilton CI lanes.
- Direct provider-free Bronze/Silver qualification reached 10/10 eligible warm
  node hits (100%), two source loads across two attempts, 20 serialized results,
  and 27,894 serialized bytes. The tiny four-row fixture was slower warm
  (73.3 ms vs 21.6 ms cold); representative benchmarking and cache-overhead
  reduction are tracked in `.planning/STATE.md` before any speedup claim.
- Validation: PR 5 focused lane 140 passed; stage/executor lane 52 passed;
  strict MkDocs, Ruff, formatting, strict mypy, repository hygiene, docs
  references, generated-doc freshness, and `uv sync --all-groups --extra intel`
  passed. Two full-suite attempts reached 896/895 passes; one/two unrelated
  sandbox tests failed because the shared host could not create worker threads
  (`std::system_error: Resource temporarily unavailable`). The affected tests
  passed in focused reruns (the remaining sandbox case required explicit
  BLAS/OpenMP single-thread environment limits).
- Subagents audited telemetry/cache/CLI/docs gaps, implemented the user-doc and
  generated-DAG slices, expanded focused tests, and reviewed the integrated
  result; the primary agent corrected findings, integrated, and validated.
- Links: ADRs 0012–0015; `docs/operations.md`;
  `docs/plan/21_hamilton_default_execution_handoff.md`.

### 2026-09-11 — Hamilton default case execution, PR 4

- Added worker-local `HamiltonLayerExecutor` orchestration across separate
  Bronze, Silver, Gold, and Platinum drivers, with cryptographic/semantic
  boundary verification, exact ordered lineage, source-once handoff, durable
  cross-attempt reuse, lifecycle events, and typed stage/failure results.
- Switched `ExperimentalPlatform` and YAML defaults atomically to Hamilton while
  preserving explicit `engine=legacy`; parent-owned tracking and process-worker
  payloads remain outside Hamilton's live-service graph.
- Added source-independent `run plan --format json`, stable case identities,
  unique attempt namespaces, prompt/source/LLM-sensitive Gold identity, and
  attempt-free pure Silver cache requests.
- Public full-case coverage verifies all four packages, sequential/process
  parity, second-attempt reuse, and Gold/Platinum invalidation after an LLM
  identity change. Focused PR 4 lane: 88 passed. Full suite: 832 passed and two
  sandbox tests failed from transient host process/thread exhaustion; both
  passed immediately when rerun alone. Ruff, formatting, strict mypy, hygiene,
  and docs-reference checks passed.
- Subagents performed the initial architecture/platform analysis, isolated
  stage-DAG fixes, platform wiring, and review; the primary agent integrated,
  corrected, tested, and documented the result.
- Links: ADRs 0012–0015; `docs/plan/21_hamilton_default_execution_handoff.md`.

### 2026-09-09 — Hamilton durable foundation, PR 2A

- Ported the strict versioned artifact/run contracts, deterministic secret-free
  hashing, atomic staging/publication, write-once local artifact storage, and
  integrity verification primitives from the reference implementation.
- Kept package exports incremental so later identity, catalog, and resume
  slices are not imported before their prerequisites exist.
- Validation: `tests/unit/test_contracts_and_storage.py` — 4 passed; targeted
  Ruff, formatting, and `git diff --check` passed.
- Links: ADRs 0013–0014; `docs/plan/21_hamilton_default_execution_handoff.md`.

### 2026-09-09 — Dual layer identities, PR 2B foundation

- Added versioned Bronze, Silver, Gold-input, Gold-materialization, and
  Platinum-input fingerprints with secret-free canonical serialization.
- Added nullable `reuse_fingerprint` to V1 manifests for backward-compatible
  durable reuse metadata; existing manifests remain readable.
- Validation: durable contract/identity tests — 6 passed; strict mypy passed
  for contracts, storage, and verification modules.

### 2026-09-09 — Incremental resume planning, PR 2C foundation

- Added typed resume policy, stage dispositions, failure/lifecycle contracts,
  contiguous-prefix planning, verified suffix execution, and partial-failure
  reporting.
- Resume semantic validation currently enforces manifest integrity and exact
  upstream lineage; layer-specific dataflow loaders remain owned by PR 3.
- Validation: durable/resume tests — 8 passed; strict mypy and Ruff passed for
  the new contracts, storage, verification, and resume modules.

### 2026-09-10 — Hamilton Bronze/Silver dataflow foundation, PR 3A

- Added the first Hamilton driver/profile surface and ported the deterministic
  Bronze/Silver preparation DAG with typed dataset/materialization contracts.
- The offline CI profile now exercises Hamilton caching while retaining network
  and artifact persistence restrictions; the source registry is loaded once
  per DAG execution.
- Validation: focused durable/dataflow suite — 9 passed; driver construction
  exposes 22 Hamilton variables. Full Gold/Platinum driver composition remains
  pending.

### 2026-09-09 — Hamilton execution foundation (PR 1)

- Added ADRs 0012–0015 defining Hamilton case orchestration, medallion-lite
  boundaries, atomic evidence publication, and the independent default-on
  Hamilton cache policy.
- Added typed `DataflowConfig`, `ExecutionEngine`, and Hamilton cache settings.
  The foundation release defaults to the explicit `legacy` engine while keeping
  Hamilton caching enabled; contradictory engine/policy combinations fail
  validation.
- Added `apache-hamilton>=1.90,<2` to core dependencies. PyPI now publishes
  `sf-hamilton` as a redirect to the Apache-maintained package.
- Validation: `tests/unit/test_config.py` — 30 passed; targeted Ruff checks
  passed after formatting fixes.
- Links: ADRs 0012–0015; `docs/plan/21_hamilton_default_execution_handoff.md`.

### 2026-09-06 — P4 follow-ups: mypy-fleet backlog items 2–5 done, 1 deferred

- **`LLMClient._retry` generic typing** (item 2): now
  `Callable[..., Awaitable[R]] -> R` — kills the two `no-any-return`
  ignores and gives fake providers real types; tenacity's `.wraps()`
  preserves the callable type (no cast needed).
- **`pydantic.mypy` plugin enabled** (item 3): first-class strict checking of
  pydantic model construction. Net effect: 8 suppressions removed across
  src/tests (the `Settings(_env_file=...)` call-arg ignores the fleet had to
  add are gone). Kept: one justified ignore for a deliberately-invalid
  `FeatureSpec()` construction.
- **`AgentRegistry.get_agent` narrowed to `type[BaseFeatureAgent]`** (item 4):
  the true contract (built-ins construct with `(config, llm_client)`); removes
  the fleet's `cast` helper in `tests/unit/test_agents.py` and surfaced a real
  mixing bug in `get_all_agents` (builtin `type[BaseFeatureAgent]` dict updated
  with discovered `type[Agent]` — now explicitly widened at the boundary).
- **`_fingerprint_cache` id-reuse flake fixed** (item 5): entries now store a
  `weakref` to the frame and validate identity before reuse — a recycled
  `id()` after gc can no longer serve another DataFrame's fingerprint.
- **Deferred** (item 1): declaring `BaseMethod.sandbox`/`evaluator` (or an
  `attach_kit()`) instead of dynamic `hasattr`/`getattr` attachment — needs a
  small ADR (touches the BaseMethod extension contract); left in backlog.

### 2026-09-06 — P4: parallel boosters investigated; opt-in knob, pin stays default (ADR 0010 amended)

- The planned "`n_jobs=-1` when Intel inactive" change was **empirically
  falsified**: on the 14-core dev host (carrying a realistic co-located
  workload), serial `n_jobs=-1` regressed 185–234x (OMP spin-wait collapse:
  xgb 0.64s→117.7s, lgbm 1.49s→348.3s) and nested lgbm `n_jobs=4` regressed
  16–40x while nested xgb `n_jobs=4` won 5x — library- and shape-dependent,
  unstable on shared hosts.
- Shipped instead: opt-in `Settings.evaluation.booster_n_jobs` (default `1`,
  `-1` or `>=1` allowed) plumbed through `ModelFactory` → the three OpenMP-backed
  built-ins only (xgboost/lightgbm/random_forest); custom entry-point models
  untouched. Hard invariant kept: when Intel acceleration is active,
  `_booster_n_jobs()` forces `1` regardless of config (logs
  `booster_n_jobs_forced_single_thread` once) — the ADR 0010 deadlock guard
  cannot be configured away.
- Full benchmark table and revival criteria recorded in ADR 0010's 2026-09-06
  amendment. 6 new tests in `TestBoosterNJobs`.
### 2026-09-06 — P3: LLM cache TTL + size-cap GC (D-9 resolved)

- Audit finding "old entries are never deleted" addressed with two
  optional `LLMConfig` knobs (default `None` = unchanged behavior, cache
  stays a reproducibility artifact): `cache_ttl_days` (per-entry expiry,
  honored on read, physically removed via GC) and `cache_size_limit_mb`
  (total-volume cap).
- diskcache semantics confirmed empirically and documented: the size cap
  is enforced **eagerly on every write** (least-recently-stored eviction,
  no background thread); writes whose expiry is already past are dropped;
  small values live inside the SQLite page file so byte accounting from
  `expire()`/`cull()` is best-effort (counts are exact).
- New: `DiskCache.stats()`/`maintain()`, `scripts/cache_gc.py` (CLI with
  `--dry-run`, config fallback to `Settings.llm`), factory wiring, 8 new
  tests (`TestCacheGC`, `TestCacheGCConfig`). Policy recorded as D-9 in
  `docs/deferred-design.md`.
- Validation: 29/29 in `tests/unit/test_llm_cache.py`; mypy/ruff clean;
  end-to-end script smoke (dry-run + real GC on a temp cache).
### 2026-09-05 — P2: mypy tests debt burned down (1015 → 0), CI gate enforced

- The full `tests/` tree now passes `uv run mypy` under the same strict
  settings as `src/`: **1015 errors → 0 across 55 files** (~1,760 insertions,
  annotation-dominant). Done by a 6-way subagent partition over disjoint
  file sets (2 workers hit a provider usage limit after finishing their
  edits — verified by the zero-error run and the full suite).
- Quality bar held: real annotations (fixtures `tmp_path: Path`, fakes typed
  to `LLMClient` hook signatures), genuine fixes over casts; ~10 justified
  `# type: ignore[code]` and a handful of `cast()` across the whole tree —
  zero blanket `Any`. No test logic/assertions changed; the hermetic
  `_hermetic_settings` fixture is intact.
- Notable genuine fixes surfaced: `Settings(evaluation={...})` dicts → typed
  `EvaluationConfig(...)`, fake providers' override signatures aligned with
  the `LLMClient` contract (several were silently absorbing `json_mode`/
  `prompt_meta` via `**kwargs`), `.values` → `.to_numpy()`.
- **CI**: `quality-debt` job (tolerated failures via `continue-on-error`)
  renamed to `mypy-tests` and now **enforces** the zero-error bar.
- Footgun fixed: `uv sync --all-groups` (the documented validation command)
  silently prunes the `intel` extra — sklearnex uninstalls, the bootstrap
  fails closed, and the Intel smoke test starts skipping. The canonical sync
  is now `uv sync --all-groups --extra intel` (AGENTS.md).
- Validation: `uv run mypy tests` → 0 errors; `uv run mypy src` → 0 errors;
  full suite 783 passed; ruff check/format clean; hygiene + docs checks pass.
- Follow-up candidates surfaced by the work (not applied, P4 backlog):
  1. `BaseMethod` attaches `sandbox`/`evaluator` dynamically (`hasattr`/
     `getattr`) — declared attrs or an `attach_kit()` would type natively.
  2. `LLMClient._retry` typed `(Callable[..., Any]) -> Any` — a generic
     `Callable[..., Awaitable[T]] -> T` would drop two `no-any-return`
     ignores in `base.py` and give fake providers real types.
  3. Enable the `pydantic.mypy` plugin (removes `Settings(_env_file=...)`
     call-arg suppressions).
  4. `AgentRegistry.get_agent` declared `type[Agent]` but returns
     `BaseFeatureAgent` constructors — return type fix kills test casts.
  5. Pre-existing flake: `BaseFeatureAgent._fingerprint_cache` keyed by
     `id(X)` can collide across tests under random ordering — clear it in
     cache tests or key without `id()`.
### 2026-09-05 — ADR 0011: strict structured outputs (hy3 json_mode fix, pydantic-validated)

- **Root cause found by live probing** (`scripts/probe_structured_output.py`,
  kept as a diagnostic): the OpenCode Go gateway **rejects `json_object`
  response_format for hy3** (400 — the incident), but **accepts strict
  `json_schema`**, and follows schema instructions in prompt-only mode.
  Also: hy3 is a reasoning model — tiny `max_tokens` budgets yield empty or
  truncated content (explains earlier "empty content" confusion).
- **New `LLMClient.complete_structured(messages, response_model)`**
  (`llm/structured.py`, ADR 0011): pydantic model → OpenAI strict
  `json_schema` → tolerant parsing (`<think>` blocks, fences, prose-wrapped
  JSON) → pydantic validation → exactly one repair round. Providers may
  override `_structured_kwargs()` (Anthropic: prompt-mode).
- **Gateway negotiation, memoized per client**: any 400 on a
  `response_format` call degrades that client to prompt-only enforcement
  (logged `llm_json_mode_degraded` / `llm_structured_mode_degraded`). The
  legacy `complete_json(schema_description: str)` path gains the same
  negotiation + tolerant parser — no config knob, self-healing per backend.
- **Cache safety**: response_format kwargs are part of the cache key, so a
  mode switch can never serve stale responses; validated content cached
  under the producing request shape.
- **Router migrated** as the exemplar call site (`RouterSelection` pydantic
  model replaces hand-parsed dicts); remaining call sites migrate
  incrementally.
- **Live end-to-end verified** against hy3: strict json_schema accepted,
  reply validated into the typed model, no degradation. One truncated-reply
  incident during verification confirmed the repair round fires correctly.
- Hygiene: `trace_generation` now passes through when LANGFUSE keys are
  absent (killed the "Authentication error" stderr spam on every call).
- Validation: 90 targeted + full suite green, ruff/format/mypy-src clean.
### 2026-09-05 — Audit follow-ups: secrets CI guard, Intel provenance, import hardening, doc fixes

- **gitleaks CI job** (`secrets` in `ci.yml`, full-history scan): closes the
  "loaded gun" gap — a plaintext key committed by accident now fails CI.
  `.gitleaks.toml` allowlists the inert dotenvx `encrypted:` ciphertext in
  history so the guard doesn't false-positive on it.
- **Intel CI job** (`intel` in `ci.yml`, ADR 0010 follow-up): installs the
  `intel` extra and runs `check_openmp_single_runtime.py` + `intel_smoke_test.py`
  on synthetic data (no API key). Both verified green locally.
- **Runtime provenance**: `runtime_info()` in `runtime/intel_bootstrap.py`
  reports *actual* activation + `sklearnex_version`; wired into
  `CaseExecutor.execute` tracker config so run metadata records which numeric
  backend produced the scores (config flag alone was ambiguous).
- **Import hardening**: `bootstrap_intel_runtime` now fails *closed* (warning +
  no patching) if settings resolution raises, instead of crashing the whole
  package import on a malformed YAML/.env. Also silenced the sklearnex banner
  (`patch_sklearn(verbose=False)`) to keep stdout clean.
- **Fixes**: `config.py` Intel comment corrected (threading/in-process, not the
  superseded "process isolation" design); `core.py` now logs
  `feature_eval_backend_forced_threading` instead of silently overriding a
  configured loky backend; `AGENTS.md` validation block uses working
  `uv run mypy src` and documents reliable pytest-summary capture
  (`grep -E "[0-9]+ (passed|failed)"` / `--junitxml`).
- Validation: full suite 763 passed; ruff/format/mypy-src/hygiene/docs checks
  green.
### 2026-09-05 — Secrets handling simplified: dotenvx dropped, `.env` untracked (local plaintext)

- The machine migration lost `.env.keys` (the dotenvx private key), making
  every committed `encrypted:` value permanently undecryptable — dotenvx was
  dead weight. Dropped it entirely rather than re-keying.
- `.env` is now **untracked and gitignored** (local plaintext secrets only,
  never committed). The dotenvx-encrypted ciphertext that remains in git
  history is inert: lost, not leaked — no rotation needed.
- Wired `env_file=".env"` into `Settings.model_config`
  (`src/feature_forge/config.py`) so pydantic-settings actually loads the
  local file. Previously `.env` was only honored via `dotenvx run` env
  injection, which no longer exists — the documented "`.env` is a config
  source" claim was broken without it. Priority verified empirically:
  real env vars > `.env` > `config/settings.yaml`.
- Updated `.env`, `.env.example`, `.gitignore`, `README.md`,
  `docs/API_KEY_SETUP.md`, `docs/migration_guide.md`, `docs/notebooks/index.md`,
  and `.planning/` (OVERVIEW/STATE/STYLE) to the new policy.
- Follow-up (not done): add a secret-scanning CI step (gitleaks) as a
  guardrail against future plaintext commits.
- Incident + hardening: wiring `env_file` exposed that the test suite was
  **not hermetic** — `tests/unit/test_api.py::test_fit_without_llm_raises`
  auto-created a *real* LLM client from the local key and fired live API
  calls inside a unit test (CI never caught this because it has no
  `FF_LLM__API_KEY`). Added an autouse `_hermetic_settings` fixture in
  `tests/conftest.py` that nulls `env_file` and strips credential env vars
  per test; `.env` loading is covered explicitly via `Settings(_env_file=)`
  tests in `tests/unit/test_config.py`. Operational note: the failed calls
  also showed the hy3/OpenCode-Go gateway returning 400 on `json_mode`
  completions — investigate before the next real experiment run.

### 2025-08-31 — Intel Extension + OpenMP deadlock hardening (ADR 0010)

- Investigated the reported XGBoost/LightGBM hang: root cause is the classic
  `libiomp5` (sklearnex/DAAL) vs `libgomp` (XGBoost/LightGBM pip wheels)
  OpenMP runtime deadlock, which only materializes once Intel acceleration is
  enabled. In the current venv there is **no** `sklearnex` and all boosting/sklearn
  libs link `libgomp`, so there is no conflict today.
- Added a single guarded bootstrap (`feature_forge/runtime/intel_bootstrap.py`)
  invoked as the first import-time action in `feature_forge/__init__.py`: it
  patches sklearn **first**, sets `KMP_DUPLICATE_LIB_OK=TRUE` before libiomp5
  loads, and engages only when `scikit-learn-intelex` is installed + the
  `intel_acceleration` setting is on.
- Deadlock mitigation: boosting estimators pinned to `n_jobs=1`
  (`evaluation/model_factory.py`); evaluation uses the **same-process
  threading backend** (no fork-after-OpenMP) whenever Intel acceleration is
  actually active (`methods/malmas/pipeline/core.py`). The override keys off
  real activation (`is_intel_active()`), so default behavior is unchanged when
  the `intel` extra is absent. (Correction 2026-09-01: originally specified
  `backend="loky"; verified empirically that forking after libiomp5 is
  initialized can deadlock in the child, so threading is the safe choice.)
- Added `intel` optional dependency group (`scikit-learn-intelex`, `threadpoolctl`)
  and config flag `Settings.intel_acceleration` (default True). Enable with
  `uv add --optional intel` + `FF_INTEL_ACCELERATION=true`.
- Full OpenMP unification (single runtime) remains out of scope for the pip/uv
  workflow; documented in ADR 0010 as follow-up (Intel-optimized XGBoost/LightGBM
  via the `intel` conda channel).

### 2026-09-01 — Intel/OpenMP fix verified end-to-end (ADR 0010 follow-up)

- Installed the `intel` extra (`scikit-learn-intelex==2026.1.0`, `daal`, `tbb`)
  and verified the deadlock mitigation actually works: a smoke test trains the
  platform's boosting models (pinned `n_jobs=1`) under a parallel (`n_jobs=-1`)
  CV with sklearnex active and completes in ~2.5s, no 100% CPU hang. Repro
  confirmed the *real* deadlock trigger is `n_jobs>1` on the model itself
  (libgomp threads alongside libiomp5) and that `fork` after OpenMP init also
  deadlocks — hence the `threading` (no-fork) evaluation backend.
- Added CI/diagnostic tooling: `scripts/intel_smoke_test.py` (subprocess-based,
  hard timeout), `scripts/check_openmp_single_runtime.py` (`threadpoolctl`
  guard asserting `KMP_DUPLICATE_LIB_OK` + `n_jobs=1` invariants), and
  `scripts/intel_ab_benchmark.py` (A/B on vs off). Verified A/B: identical CV
  AUC (0.977 both) and comparable wall time (~5.1s vs 4.7s) — DAAL is numerically
  stable here.
- Added `tests/integration/test_intel_smoke.py` (skipif not active) and kept
  `tests/test_intel_bootstrap.py`. Full suite green (exit 0); also fixed the
  pre-existing `test_config.py::test_default_values` (now yaml-independent) and
  4 unused `# type: ignore` comments in `wandb_backend.py`.

### 2026-08-30 — Versioned prompt files + cache provenance (ADR 0009)

- Prompt YAMLs renamed to `<name>.v<N>.yaml` (all 15 now at
  `.v1.yaml`); `PromptRegistry` resolves a bare name to the highest
  version and supports `version=N` pins for exact reruns of old
  experiments.
- `PromptProvenance` (`prompt_name`, `prompt_version`, `prompt_sha256`
  over system+user content) is returned by the registry and threaded
  through every LLM call site as `prompt_meta=`: stored inside cache
  payloads, emitted on `llm_request`/`llm_cache_hit`/`llm_cache_miss`/
  `llm_response` log events, and persisted in method run artifacts
  (`artifacts["prompt_meta"]`). Verified end-to-end against a real
  `DiskCache`.
- `prompt_meta` is deliberately NOT part of the cache key — the key
  already hashes the rendered messages, so provenance is metadata, not
  identity (ADR 0009).
- Docs referencing prompt paths updated (`docs/methods.md`,
  `docs/migration_guide.md`, ADR 0008).
- Audit follow-up (same day): iterative `iteration_record` dicts in
  malmus/llmfe/caafe now persist `prompt_meta` (ADR 0009 clause 4
  conformance); `llm_cache_hit` logs additionally emit the stored
  `cached_prompt_meta` so hit lines are self-contained; the
  `docs/methods.md` Prompt section now documents versioned-file
  semantics, pinning, and `provenance()`; the versioned-filename test
  was tightened to a full regex match.
- Validation: full suite 750 passed / 1 failed (the known pre-existing
  hy3-yaml config test). ruff/format/hygiene/docs checks pass; mypy
  clean on all touched src files.
- Links: ADR 0009 (`docs/decisions/0009-versioned-prompts-cache-provenance.md`),
  `src/feature_forge/methods/_prompting.py`, `src/feature_forge/llm/base.py`.

### 2026-08-30 — Cache-key audit correction: prompt edits DO auto-invalidate keys (docs were wrong)

- Empirically verified (and re-read `compute_cache_key`): the SHA-256
  cache key is computed over the **full rendered `messages`** plus
  provider/model/temperature/max_tokens/kwargs. Editing prompt YAML
  changes the rendered text → different key → cold call; identical text
  reproduces the identical key. The cache is content-addressed; stale
  responses are impossible by construction.
- The repeated claim "prompt text changes do not auto-invalidate cache
  keys" (ADR 0001 consequence, ADR 0005 consequence, carried into ADR
  0008 and yesterday's log entries) is wrong — it conflated
  *invalidation* (automatic, content-addressed) with *orphan clean-up*
  (old entries are never deleted; no TTL configured).
- Corrections applied as dated annotations to ADRs 0001/0005/0008
  (statements preserved, correction noted inline, per the no-silent-
  rewrite convention).
- Remaining deliberate gaps (future work, ADR-gated if pursued):
  prompt-template provenance in run artifacts / cache payloads (answer
  "which template version produced this response"), and orphan-entry
  GC/TTL policy for `memory_files/llm_cache`.
- Links: `src/feature_forge/llm/cache.py` (`compute_cache_key`),
  `src/feature_forge/llm/base.py` (`build_cache_key` call sites), ADRs
  0001/0005/0008.

### 2026-08-30 — Prompts: Jinja2 templating engine + full prompt externalization (ADR 0008)

- Adopted ADR 0008: `PromptRegistry` now renders through a per-method
  Jinja2 environment (`StrictUndefined`, `autoescape=False`,
  `keep_trailing_newline=True`), replacing `str.format()`. Missing
  template variables still fail before any LLM call (now as
  `UndefinedError` instead of `KeyError`).
- Params models are now pure Pydantic domain schemas
  (`PromptParams` base in `methods/_prompting.py`); their `render()`
  methods were removed — rendering goes through the registry
  (`registry.render(name, params)` /
  `registry.render_messages(name, params)`).
- All LLM prompt text is now externalized: the five templated YAMLs
  migrated to `{{ var }}` syntax, and the MALMAS memory summarization
  prompts moved from `memory/prompts.py` f-strings to new
  `malmas/prompts/summarize_agent.yaml` + `summarize_global.yaml`
  (Prompt model gained an optional `user:` template). MALMAS agents'
  code-coupled user prompts stay in Python, now explicitly documented in
  the ADR.
- `jinja2>=3.1.6` added as a direct dependency. ADR 0005 annotated as
  partially superseded; decision index updated.
- Validation: full suite 744 passed / 1 failed (the pre-existing
  `test_config.py::test_default_values` failure tied to the working
  tree's uncommitted hy3 backend switch — unrelated). ruff/format/
  hygiene/docs checks pass; mypy clean on touched src files.
- Links: ADR 0008 (`docs/decisions/0008-jinja2-prompt-templating-full-externalization.md`),
  ADR 0005, `src/feature_forge/methods/_prompting.py`.

### 2026-08-30 — LLM cache audit: verified + made cache dir configurable; fixed sandbox RLIMIT_AS starvation

- Audited the LLM call path against the ADR 0001 mandate ("enforced
  DiskCache, SHA-256 keys"): it was already fully wired
  (`llm/cache.py` diskcache/SQLite at `memory_files/llm_cache`, keyed by
  `compute_cache_key`; `factory.py` auto-attaches when
  `llm.cache_responses`; all four providers pass it through; live cache
  held 3 entries). No architectural change, so no ADR needed.
- Added `LLMConfig.cache_dir` (default `memory_files/llm_cache`) so the
  cache location is configurable via YAML/`FF_LLM__CACHE_DIR` instead of
  a CWD-relative hardcoded default (`config.py`, `llm/factory.py`,
  `config/settings.yaml`).
- Fixed sandbox worker memory-starvation: default RLIMIT_AS 512 MB is
  below the pandas/pyarrow/OpenBLAS runtime VSZ (~1 GB), so parquet reads
  failed to map shared objects ("failed to map segment") and queue writes
  hit "can't start new thread". Defaults raised to 2048 MB
  (`SandboxLimits`, `SandboxedExecutor`, `EvaluationConfig.sandbox_max_memory_mb`);
  empirical repro: 512 MB fails, >=1024 MB passes. The 3 long-failing
  `test_leakage_and_caching.py` tests now pass.
- Fixed a pre-existing unit-test landmine:
  `test_sandbox.py::test_import_fallback_no_error` called the real
  `_apply_resource_limits(128)` **in the pytest process**, permanently
  shrinking its address space to 128 MB and cascading MemoryError/OSError
  ENOMEM through every later test (limit is inherited by spawned workers
  too). Test now runs the real call in a subprocess.
- Validation: full suite 739 passed / 1 failed — the failure is
  `test_config.py::test_default_values`, pre-existing relative to the
  working tree's uncommitted `settings.yaml` backend switch (hy3),
  unaffected by these edits. ruff/format/hygiene/docs checks pass; mypy
  clean on the touched src files (repo-wide strict mypy drift and test
  annotations are pre-existing).
- Links: ADR 0001 (`docs/decisions/0001-architectural-boundaries.md`),
  `src/feature_forge/llm/cache.py`, `src/feature_forge/evaluation/sandbox.py`.

### 2026-08-29 — Documentation reorganized: founding roadmap reclassified, plan/18-20 published, two stowaways archived

- Reclassified `docs/MALMAS_Technical_Roadmap.md` (Apr 2026 founding vision) as
  provenance in place: added an in-file status header, relabeled the mkdocs nav
  entry to "MALMAS Founding Roadmap" and nested it under *Implementation Plan*.
  Its references to removed paths (`main_demo/`, `global_config.py`, `malmas/`,
  `baselines/`) are now explicitly historical; realized architecture lives in
  `docs/plan/` + `docs/decisions/`.
- Archived two completed/one-off plan docs (number-collision fix):
  `docs/plan/01_consolidate_notebooks.md` -> `docs/archive/01_consolidate_notebooks.md`,
  `docs/plan/02_llm_code_parsing_research.md` -> `docs/archive/02_llm_code_parsing_research.md`
  (archival headers added per `docs/archive/README.md`).
- Added `plan/18` (SAGE memory), `plan/19` (UX), and new `plan/20_long_term_roadmap.md`
  to the mkdocs nav and to the `docs/README.md` docs map. `20_long_term_roadmap.md`
  is the combined long-term plan: founding vision -> realized state (`STATE.md` +
  ADRs) -> `experiments/` validation frontier -> 5 evidence-gated themes (each tied
  to a `docs/deferred-design.md` trigger) -> explicit "deliberately not doing" list.
- Updated `docs/plan/00_index.md` (Current Phase = library complete / validating the
  science; Next Steps point at the long-term roadmap + D-2).
- Validation: `check_docs_references.py` ok, `check_repo_hygiene.py` ok,
  `mkdocs build` ok (the pre-existing `README.md`/`index.md` strict-mode warning
  is unrelated to these edits). No code changed.
- Links: `docs/plan/20_long_term_roadmap.md`, `docs/README.md`, `mkdocs.yml`.

### 2026-08-29 — Default LLM backend switched to OpenCode Go (hy3)

- `config/settings.yaml` now points at the OpenCode Go subscription gateway
  (`provider: openai`, `model: hy3`, `base_url: https://opencode.ai/zen/go/v1`);
  previously `deepseek-chat` via `https://api.deepseek.com`.
- `FF_LLM__API_KEY` in `.env` set to the OpenCode Go key (copied from
  `~/.local/share/opencode/auth.json`; written plaintext because `.env.keys`
  is absent on this machine — re-encrypt with `dotenvx encrypt` when available).
- Smoke-tested end-to-end through `create_llm_client`: `complete()` and
  `complete_json()` (JSON mode) both pass. `hy3` is a reasoning model —
  `reasoning_content` counts toward `max_tokens`, so keep per-call budgets
  generous.
- Caveat recorded: Go subscription is value-capped (~$12/5h, $30/wk, $60/mo);
  use pay-as-you-go for full experiment matrices. Enforced DiskCache (ADR 0001)
  keeps repeat prompts free.
- Links: `docs/API_KEY_SETUP.md`, `config/settings.yaml`.

### 2026-08-29 — ADR population, deferred-research enrichment, docs map expansion

- Mined the implementation plan docs into six new ADRs (0002–0007: 3-tier
  memory, dynamic router, methods namespace + protocol, prompt colocation,
  experiment execution seam, observability stack), each verified against
  `src/` before acceptance.
- Extended `docs/deferred-design.md` with D-5–D-8 (dataset registry growth,
  Langfuse depth, OTel export, tracker-native sweeps) and refined D-1–D-4
  evidence.
- Expanded `docs/README.md` into a full documentation map with per-plan-doc
  descriptions; flagged `docs/plan/01_consolidate_notebooks.md` and
  `docs/plan/02_llm_code_parsing_research.md` as archive candidates.
- Fixed the stale `AgentMemory` unbounded-growth claim in
  `.planning/STATE.md` (trimming is implemented; see ADR 0002).
- Links: `docs/decisions/README.md` index, `docs/deferred-design.md`.

### 2026-08-29 — Guidance and documentation scaffolding

- Rewrote `AGENTS.md` into the decision-order format: purpose, decision
  order, architectural boundaries, engineering conventions, validation.
- Added scaffolding: `docs/README.md` (documentation map),
  `docs/decisions/` with ADR conventions + template, ADR 0001 recording
  the architectural boundaries, `docs/deferred-design.md`,
  `docs/archive/README.md`, and this log.
- Links: [ADR 0001](docs/decisions/0001-architectural-boundaries.md).

### 2026-07-11 — R11/R17: tracker backends unreachable, then unsafe by default (backfilled)

- R11: `TrackerConfig.backend` was never consulted — `ExperimentalPlatform.run()`
  hardcoded `NoOpTracker`, so WandB/MLflow were unreachable from the platform
  despite being advertised. Fixed with `create_tracker_from_config`
  (`src/feature_forge/experiment/factory.py`).
- R17: with R11 live, the default backend `"wandb"` would call `wandb.init`
  at run time and fail for unconfigured users. Default changed to `"none"`
  in `TrackerConfig` and `settings.yaml` — safe by default, tracking is
  opt-in.
- Backfilled 2026-08-29 from the remediation tables in `.planning/STATE.md`
  so ADR 0006's citation resolves; this log did not exist when the findings
  were made.
- Links: ADR 0006 (`docs/decisions/0006-experiment-execution-seam.md`).

### 2026-09-09 — Hamilton-default runtime review and implementation handoff

- Reviewed the default branch and the medallion feature branch (preserved as
  immutable tag `archive/pr-1-medallion-refactor` at `690a764`).
  The feature branch contains strong medallion contracts and verification but
  does not connect Hamilton to the public durable execution path; its combined
  graph contains disconnected stages and its cache policy produced only 4 hits
  across 10 eligible Silver nodes on an identical second execution.
- Maintainer decisions: `HamiltonLayerExecutor` coordinates stage DAGs;
  Hamilton becomes the default engine; Hamilton caching is enabled by default
  in every profile; verified medallion packages remain authoritative evidence;
  the legacy engine remains an explicit rollback path during parity.
- Added `docs/plan/21_hamilton_default_execution_handoff.md` with the technical
  contracts, dual reuse/content fingerprint model, stage DAGs, cache semantics,
  six dependency-ordered PRs, agent ownership, tests, migration, and rollback
  gates. Updated the authoritative plan index, documentation map, MkDocs nav,
  and long-term-roadmap status note.
- The plan was drafted with three parallel AI subagents reviewing governance,
  Hamilton runtime/identity design, and PR validation/merge sequencing. Their
  critical corrections—incremental planning, separate reuse and evidence
  fingerprints, worker-local executor construction, and an atomic default
  switch—were incorporated.

## AI assistance disclosure

- 2026-09-16: Plan 23 PR 2 and PR 3 (partition-aware discovery; fold-local
  preprocessing and selection) implemented by AI assistance (pi coding
  agent with sequenced glm-5.3-flash worker subagents and an independent
  reviewer pass); pending review and acceptance by the maintainer.

- 2026-09-14: ADR 0016 legacy-engine removal (documentation/status slice:
  user docs, migration guidance, operations, plan/state status, and report
  log) implemented by AI assistance (pi coding agent with parallel worker
  subagents); pending review and acceptance by the maintainer.

- 2026-09-09: Repository/Hamilton review and the default-runtime implementation
  handoff were drafted with Codex and three parallel subagents. The maintainer
  selected `HamiltonLayerExecutor`, Hamilton as the default engine, and
  default-on caching; agents supplied governance, identity/resume, cache, and
  stacked-PR analysis.

- 2026-08-30: Prompt versioning scheme (ADR 0009), provenance threading,
  and cache-payload instrumentation implemented by AI assistance (pi
  coding agent); version-file layout per maintainer direction; accepted
  by the maintainer.

- 2026-08-30: Jinja2 prompt migration (ADR 0008), memory-prompt
  externalization, and param-schema refactor implemented by AI
  assistance (pi coding agent); accepted by the maintainer.

- 2026-08-30: LLM cache audit, `cache_dir` config knob, sandbox
  RLIMIT_AS default bump, and the `test_import_fallback_no_error`
  subprocess fix drafted and implemented by AI assistance (pi coding
  agent); root causes empirically reproduced before fixing; accepted by
  the maintainer.

- 2026-08-29: `AGENTS.md` rewrite, ADR 0001, and documentation
  scaffolding drafted with AI assistance (pi coding agent); reviewed and
  accepted by the maintainer.
- 2026-08-29: ADRs 0002–0007, deferred-design entries D-5–D-8, and the
  expanded documentation map drafted by parallel AI worker agents and
  audited by an AI reviewer agent (pi coding agent); reviewer-flagged fixes
  applied and accepted by the maintainer.
