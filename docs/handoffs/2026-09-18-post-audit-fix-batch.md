# Plan 23 — Post-Audit Fix Batch, Slice Commits, and Push

**Date:** 2026-09-18
**Status:** Ready for implementation
**Precedes:** PR 6 (LLM cache identity v2, ADR 0020)
**Scope:** Apply the verified post-audit fixes to the uncommitted PR 1–5 tree,
then commit in three slices and push `main` (which also publishes `5fd0a20`).

## Context

Three parallel audits (src / tests / docs-meta) plus a verification pass over
every must-fix finding produced the findings below. Verification re-checked
each claim against the working tree with quoted evidence; line numbers below
are the **verified** ones (several audit citations were stale and are noted).
One session carry-forward was **refuted** and is closed.

Current tree state: 63 uncommitted paths on top of `5fd0a20` (plan-23 PRs 1–5
plus the 2026-09-18 test-workflow change). Full suite green as of the last
run: 1101 passed / 9 skipped / 6 xfailed, zero XPASS. Host: Linux, Landlock
ABI 7.

## Agent workflow for this handoff

- **Worker:** `worker` agent pinned to `model: opencode-go/deepseek-v4.1-flash`
  (`~/.pi/agent/agents/worker.md`). Implements fixes slice by slice; runs
  targeted tests **sequentially** (never in parallel tool calls — concurrent
  pytest processes share `/tmp` and false-positive the leak pins).
- **Reviewer:** `reviewer` agent pinned to `model: zai/glm-5.3-flash`
  (`~/.pi/agent/agents/reviewer.md`). Read-only review after each slice;
  REQUEST-CHANGES loop until clean.
- **Main agent owns edits** to the three plan-23 marker modules
  (`tests/unit/test_plan23_sandbox_hardening.py`,
  `test_plan23_llm_cache_identity.py`,
  `test_plan23_evaluation_integrity.py`) — worker must not touch them
  (they encode the maintainer's pinned acceptance record; prior worker feed
  regressions came from edits there).
- Never commit/reset/clean/stash; never print `.env` values. The main agent
  performs the commits at the end per the strategy below.

## Fix batch (verified findings, priority order)

### F1 — MAJOR — worker death bypasses typed error hierarchy
- **Where:** `src/feature_forge/evaluation/sandbox.py:835-846`
  (`_wait_for_response`) and `:848-858` (`_poll_response`).
- **Verified:** both catch only `queue.Empty`. A worker killed by RLIMIT_AS
  (feeder-thread failure documented at `sandbox.py:49-59`, mirrored
  `:1138-1152`) closes the queue write-end with no item → raw `EOFError`
  escapes `execute()`. No intentional queue-close path exists (worker only
  `put`s; `_close_response_queue` runs pre-read or post-cleanup only).
  Today a raw `EOFError` bypasses both `(SandboxTimeoutError, TimeoutError)`
  and `CodeExecutionError` handling at `methods/malmas/pipeline/core.py:70-77`
  and kills the pipeline.
- **Fix:** wrap both queue reads:
  `except EOFError as exc: raise CodeExecutionError("sandbox worker died before reporting a result") from exc`
  (scope to the queue reads only; do not widen other exception handling).
- **Test:** new regression test in `tests/unit/test_sandbox_lifecycle.py` —
  worker that exits without posting (e.g. code that `os._exit(1)`s? No — that
  is blocked; instead start a handle manually or monkeypatch the worker target
  to a no-op process) asserting `CodeExecutionError` (not `EOFError`) and no
  temp/worker leak.
- **Note:** `_classify_failure` (`experiment/hamilton_executor.py:163-172`)
  classifies `CodeExecutionError` as DETERMINISTIC — same as today's raw
  fallthrough; no recovery-semantics change. OOM→RESOURCE reclassification is
  out of scope.

### F2 — MAJOR — `_partition_scope` fails open under holdout (leakage)
- **Where:** `src/feature_forge/dataflows/platinum.py:86-104` (silent
  fallback at `:102-103`).
- **Verified:** under `protocol == "holdout"`, a SilverPackage whose
  `fold_assignments` lacks a `partition` column silently evaluates discovery
  and reported evidence on **all rows** — the leakage ADR 0018 bans. Mirror
  path raises: `experiment/hamilton_executor.py:217-222` (`DatasetError`).
  Reachability: reused pre-ADR-0018 Silver packages pass artifact-level
  validation (`dataflows/_io.py:160`) and flow into Platinum directly. The
  docstring admits the fallback exists "to keep legacy fixtures evaluable".
- **Fix:** in `_partition_scope`, when `protocol == "holdout"` and the
  `partition` column is missing → `raise DatasetError(...)` matching the
  executor's message style. Keep the all-rows fallback **only** for
  `protocol == "compatibility"`.
- **Test fallout (mechanical, add the `partition` column to fixtures):**
  - `tests/unit/test_platinum_dataflow.py:91`
  - `tests/unit/test_platinum_evidence_v2.py:128`
  - `tests/unit/test_platinum_reuse_and_results.py:116`
  - `tests/unit/test_scope_contract.py:184` (verify no case feeds platinum
    nodes under holdout without the column)
  - `tests/unit/test_plan23_evaluation_integrity.py` unaffected (sets
    `partition` or uses the real fold builder).
- **Also add** a small negative test: holdout + missing partition column →
  `pytest.raises(DatasetError)`.

### F3 — MAJOR — strict-profile tests fail (not skip) on non-Landlock hosts
- **Where:** `tests/unit/test_plan23_sandbox_hardening.py:214-228`
  (`test_timeout_cleans_up_worker_and_temp_files`) and `:259-277`
  (`test_async_sandbox_entry_point_is_cancellable_and_bounded`) —
  **main agent edits** (marker module).
- **Verified:** both build `SandboxedExecutor(timeout_seconds=1.0)` with
  default `SandboxProfile.STRICT` (`sandbox.py:565-571`);
  `_ensure_strict_available` (`sandbox.py:818-827`, audit cited ~552) raises
  `SandboxContainmentError` pre-launch on macOS/Windows → `pytest.raises(
  SandboxTimeoutError)` fails. They pass here only because the host is ABI 7.
- **Fix:** import the `_require_landlock()` guard pattern from
  `tests/unit/test_sandbox_containment.py:54-58` and apply to both tests
  (skip reason citing plan 23 §8 sandbox qualification). Prefer the guard over
  switching to degraded: these are §7.12 strict-path pins.
- **Systemic note (backlog, not this batch):** ~8 more files execute
  default-strict executors (`test_sandbox.py:206+`, `test_evaluation.py:135-232`,
  `test_utils.py:53+`, `test_metamorphic.py`, `test_gold_dataflow.py:147`);
  a shared conftest guard is the eventual right fix. CI (Ubuntu 24.04) has
  Landlock, so this does not block CI.

### F4 — MAJOR — async leak pin vacuous under `TMPDIR != /tmp`
- **Where:** `tests/unit/test_sandbox_lifecycle.py:29-30`.
- **Verified:** `glob("/tmp/...")` while sandbox temp files honor `TMPDIR`
  via `tempfile.NamedTemporaryFile` (`sandbox.py:771-781`). Twin helper in
  `test_plan23_sandbox_hardening.py:50` correctly uses
  `Path(tempfile.gettempdir())`.
- **Fix:** `set(glob.glob(str(Path(tempfile.gettempdir()) / "feature_forge_input_*")) + ...)`.
  Keep the settle-tolerant assertion semantics unchanged.

### F5 — MAJOR — no pin that the production default is strict
- **Verified:** default `config.py:372` (`sandbox_profile: SandboxProfile =
  SandboxProfile.STRICT`); executor built in `MalmasPipeline.__init__`
  (`core.py:131-135`, **not** `_exec_sandbox` — that only consumes an injected
  sandbox). `SandboxProfile.STRICT` has zero occurrences in `tests/`; nothing
  asserts `Settings().evaluation.sandbox_profile` or the pipeline default.
- **Fix (test-only, no host dependency — inspect `.profile`, no launch):**
  in `tests/unit/test_sandbox_containment.py` add
  `test_production_default_profile_is_strict`: assert
  `Settings().evaluation.sandbox_profile is SandboxProfile.STRICT` and, via a
  duck-typed config double or the real `MalmasPipeline.__init__` with a stub
  LLM client, that the pipeline's executor `.profile is STRICT` when none is
  injected. Guards against a future "helpful" auto-downgrade (plan §5.2).

### F6 — MAJOR — worker-side containment failure path untested
- **Verified:** worker reports `("containment_unavailable", str(exc), {})` at
  `sandbox.py:1281` (inside the containment try at `:1270-1282`); parent maps
  it to `SandboxContainmentError` at `:896-897`. Zero hits for
  `containment_unavailable` in `tests/`. Spawn children re-import the module,
  so parent monkeypatches do **not** propagate; an env-var seam would need an
  `_WORKER_ENV_ALLOWLIST` addition (production change — reject).
- **Fix (no production change):** in `tests/unit/test_sandbox_containment.py`
  call `_sandbox_worker_main` **directly in-process**: monkeypatch
  `_apply_strict_containment` to raise, no-op monkeypatch
  `_establish_worker_process_group` and `_scrub_worker_environment` (they
  would mutate the pytest process's pgid/environ), supply a real tiny parquet
  input + `mp.Queue`; assert `queue.get()[0] == "containment_unavailable"`,
  then feed the tuple through the parent's `_consume_response` and assert
  `SandboxContainmentError`. Worker returns immediately after the put, so the
  in-process call is inert.

### F7 — MINOR — wrong ADR pointer for PR 6 (verified by main agent)
- `.planning/STATE.md:92` and `docs/plan/00_index.md:149` say "PR 6 (LLM
  cache identity v2, **ADR 0018 decision 6**)" — ADR 0018 decision 6 is the
  two-arm evaluation (already landed). Cache identity v2 is **ADR 0020**
  (`docs/decisions/0020-llm-cache-request-identity-v2.md`).
- **Fix:** replace with "ADR 0020" in both files.

### F8 — MINOR — stale status lines (verified by main agent)
- `docs/plan/23_evaluation_integrity_security_hardening.md:4`: "PRs 1–4
  complete … PRs 5–7 unlocked" → "**PRs 1–5 complete; PRs 6–7 remaining**".
- `.planning/STATE.md:3`: "Last updated: 2026-09-16" → `2026-09-18`.
- `.planning/STATE.md:4` status line: same "PRs 1–4 complete" staleness →
  bump to PRs 1–5.

### F9 — QoL (cheap, fold into F2's commit) — bare `assert` under `-O`
- `dataflows/platinum.py:811-813`: replace
  `assert {...} == PLATINUM_REQUIRED_ARTIFACTS` with
  `if ... != ...: raise DatasetError(...)`. No test fallout.
- `api.py:154` has the same pattern — fix for consistency or record as
  accepted (decide in review; one-line either way).

### F10 — QoL — REPORT_LOG PR-4 entry links the PR-5 handoff
- `REPORT_LOG.md:268` links `2026-09-18-plan23-pr5-sandbox-containment.md`;
  the PR-4 entry should link `2026-09-16-plan23-pr4-…md`. (Line 186 is the
  PR-5 entry's correct link.) `.claude/` is gitignored, so links dangle on
  fresh clones anyway — pre-existing convention, just fix the reference.

## Verified-but-deferred (accepted risks / backlog — record, don't fix now)

| Item | Verified state | Disposition |
|---|---|---|
| Landlock ABI≥4 denial is TCP-only; AF_UNIX/NETLINK + UDP still creatable at kernel level; `socketpair` not shimmed (`sandbox.py:109-117`, wiring `:302-321`, shims `:1290-1291`) | CONFIRMED; kernel test probes TCP only | Fixing at kernel level (seccomp everywhere) **requires an ADR 0019 amendment** per governance. This batch: extend `_BlockedSocket` shim to `socketpair` + correct ADR 0019 wording ("TCP bind/connect at ABI≥4, socket(2) denial via seccomp below"). Amendment for unconditional seccomp → PR 6/7 window |
| `EnvironmentSnapshot`/`_environment(None)` fabricates `sandbox_profile="strict"` (`hamilton_executor.py:88-98`; contract defaults `runs.py:40-41`) | PARTIAL — fabrication exists but branch is dead in production (only call site passes real settings) | Change `_environment` fallback to `"unknown"` (validation- and fingerprint-safe); **leave** contract defaults (v1 fixture compatibility) |
| Platinum materialization lacks `existing_package` guard (`platinum.py:768-815`; siblings `bronze.py:309`, `silver.py:448`, `gold.py:399-402`) | PARTIAL — guard missing; **overwrite risk refuted**: `atomic_publish_directory` (`storage/atomic.py:60-63`) raises `FileExistsError`, never replaces | Add `existing_package` call (idempotency + parity + friendly `DatasetError`); schedule with PR 6/7, not this batch |
| ~~`LocalArtifactStore.commit` atomically replaces namespace dir~~ | **REFUTED** — commit is fail-closed under a mkdir lock; only `.tmp-` staging dirs and stale locks are ever removed | Closed. Strike from carry-forward lists |
| Tiny frames skip row-local probes (`scope.py:27-28,91,101`; pin at `test_scope_contract.py:116-121`) | PARTIAL — vacuous-pass unit real, but the claimed 2-row Gold bypass is **unreachable** end-to-end: gold gates the full Silver frame (`gold.py:148,166-169`) and holdout floors each partition at `cv_folds` (`holdout.py:112-117`) | Leave behavior; add the reachability note to ADR 0018 or plan 23 §4.2 discussion; revisit only if a direct-caller API emerges |
| §7 item 13 sub-clauses with no stubs: kwargs-variation + persisted-metadata secret scan (`docs/plan/23_…md:328-329`, §5.4 `:220-224`) | CONFIRMED — exactly 6 xfails exist; the other two sub-clauses absent | Add two `xfail(strict=True)` stubs to `test_plan23_llm_cache_identity.py` (PR-6 backlog pointers) — **main agent edits** (marker module); fold into this batch so PR 6 cannot forget them |
| Test-helper duplication (temp-leak/child-scan, code wrappers, np/pd I/O parametrize lists; 4× Platinum `_packages` fixtures) | CONFIRMED (copies already drifting) | Backlog refactor `tests/unit/` helper module — do **not** mix into this batch |
| `settings.yaml` missing `protocol`/`evaluation_holdout_fraction` surface | CONFIRMED partial-by-design | Add both with comments when touching config for PR 6/7 |
| Sandbox QoL: stale module docstring, dead `try/except: raise` (`sandbox.py:1334`), duplicate `import builtins` (`:1293`), `max_memory_mb` 512-vs-2048 default mismatch (`:398` vs `:570`) | CONFIRMED | Backlog; safe to fold the trivial ones into F1's slice if reviewer approves |

## Commit strategy (after fixes + gates green)

Three slices, in order — each self-contained and gate-green:

1. **`feat(evaluation): plan 23 PRs 1–4 — holdout protocol, fold-local preprocessing, platinum evidence v2`**
   - `src/feature_forge/contracts/` (all), `dataflows/` (_io, silver, gold,
     platinum), `evaluation/holdout.py`, `evaluation/preprocessing.py`,
     `evaluation/scope.py`, `experiment/execution.py`,
     `experiment/hamilton_executor.py` (non-F1 parts), `platform.py`
     (non-sandbox parts), `contracts/runs.py`
   - tests: `test_fold_preprocessing.py`, `test_scope_contract.py`,
     `test_platinum_evidence_v2.py`, `test_platinum_reuse_and_results.py`,
     `test_platinum_uncertainty.py`, `test_plan23_evaluation_integrity.py`,
     modified `test_hamilton_recovery.py`, `test_platform_e2e.py`,
     `test_execution_policy.py`, `test_fail_fast_sequential.py`,
     `test_dataflow_inventory.py`, `test_hamilton_cache.py`,
     `test_silver_dataflow.py`
   - docs: ADR 0018 + 0020 + 0020 index row, `docs/README.md`,
     `docs/decisions/README.md`, `docs/operations.md`,
     `docs/generated/stage_dags.md`, `mkdocs.yml` (non-spike rows),
     `uv.lock` (scipy), `pyproject.toml` (scipy dep),
     `scripts/qualify_hamilton_cache.py`, REPORT_LOG PR 1–4 entries (fix
     F10 link), `config/settings.yaml`
2. **`feat(sandbox): plan 23 PR 5 — strict OS containment and bounded worker lifecycle`**
   - `src/feature_forge/evaluation/sandbox.py`, `exceptions.py`,
     `evaluation/__init__.py`, `evaluation/kit.py`, `config.py`
     (SandboxProfile), `methods/malmas/pipeline/core.py`, `platform.py`
     (sandbox plumbing), `hamilton_executor.py` (async + F1)
   - tests: `test_sandbox_io_defense.py`, `test_sandbox_containment.py`,
     `test_sandbox_lifecycle.py`, `test_plan23_sandbox_hardening.py`,
     `test_evaluation.py` (timeout budget fix)
   - docs: ADR 0019, `docs/spikes/`, `scripts/spike_sandbox_isolation.py`,
     `experiments/sandbox_isolation_spike/`, plan-23 doc + `00_index.md` +
     `.planning/STATE.md` status updates (F7/F8 land here), REPORT_LOG PR 5
3. **`chore(test): two-tier workflow — coverage opt-in, timeout budget fix`**
   - `pyproject.toml` (addopts), `AGENTS.md`, `README.md` (Development),
     REPORT_LOG workflow entry
   - Note: `test_evaluation.py` straddles 2 and 3 — put the file in slice 2
     and mention the budget fix in slice 3's message body, or move it to 3;
     reviewer's call. Do not split a file across commits.

Then: `git push origin main` (publishes `5fd0a20` + the three new commits).
Report back the resulting `git log --oneline -5` and remote status.

**Slicing mechanics:** files not listed (e.g. `docs/README.md`) default to the
slice whose diff hunks they belong to; where a file carries hunks for two
slices (platform.py, hamilton_executor.py, config.py, pyproject.toml,
REPORT_LOG), prefer `git add -p` hunk selection over file-level assignment;
if hunk-splitting gets error-prone, collapse slices 1+2 into one plan-23
commit — acceptable fallback, note it in the report.

## Validation gates (run sequentially, never in parallel)

```bash
uv sync --all-groups --extra intel
uv run pytest                       # full suite; expect 1101+ passed, 6 xfailed, 0 XPASS
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run python scripts/check_repo_hygiene.py
uv run python scripts/check_docs_references.py
```

Plus, after doc edits: `uv run mkdocs build --strict` and `uv lock --check`
and `git diff --check`. Capture pytest summary via junitxml
(`--junitxml=/tmp/junit-fixbatch.xml`) since `-q` output can interleave.

## Acceptance criteria

- [ ] F1–F10 applied with tests updated/added as specified; new negative
      tests for F2, F5, F6 pass; F3 tests skip cleanly when Landlock absent
      (simulate via monkeypatched `_probe_landlock_abi` returning None, or
      verify by inspection)
- [ ] Two new §7-item-13 strict xfails present (kwargs variation, persisted
      metadata secret scan) with PR-6/ADR-0020 references
- [ ] Refuted carry-forward recorded in REPORT_LOG (commit does not replace
      namespaces; overwrite risk closed)
- [ ] All gates green; suite count unchanged or higher (new tests added),
      xfail count 6 → 8
- [ ] Three commits created per strategy (or documented fallback), pushed;
      working tree clean (except gitignored)
- [ ] REPORT_LOG entry for the fix batch with AI-assistance disclosure
      (worker: DeepSeek V4.1 Flash; reviewer: GLM-5.3 Flash; coordination by
      the session agent)

## Out of scope

- PR 6 implementation (ADR 0020) — next handoff after this batch lands
- xdist adoption, test-helper dedup, seccomp-everywhere ADR amendment,
  `existing_package` platinum guard, `settings.yaml` protocol surfacing
- Any behavioral change to `row_local_violations` (see deferred table)
