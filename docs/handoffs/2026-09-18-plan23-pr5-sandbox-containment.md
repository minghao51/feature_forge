# Plan 23 PR 5 — Sandbox Containment and Bounded Worker Lifecycle Handoff

**Date:** 2026-09-18  
**Status:** Ready — scope is plan 23 PR 5 only; one implementation-time
  decision is pre-sanctioned by ADR 0019 (see Decision points)  
**Authoritative plan:** `docs/plan/23_evaluation_integrity_security_hardening.md` (§5.1–§5.3, §6 PR 5, §7 tests 10–12, §8)  
**Governing decision:** ADR 0019 (Accepted 2026-09-15) — implement literally  
**Predecessor:** `docs/handoffs/2026-09-16-plan23-pr4-platinum-v2-evidence-uncertainty.md` (PR 4, complete 2026-09-18)

## Start here

1. `AGENTS.md` (decision order, validation gates)
2. `docs/plan/00_index.md` → item 12
3. `docs/plan/23_evaluation_integrity_security_hardening.md` (§0, §1, §5, §6 PR 5, §7, §9)
4. ADR 0019 (all six decisions), ADR 0001 (sandbox/cache boundaries), ADR 0014 (write-once packages)
5. `docs/spikes/2026-09-15-sandbox-os-isolation.md` +
   `experiments/sandbox_isolation_spike/2026-09-15/report.json` (mechanism evidence)
6. `REPORT_LOG.md` entries dated 2026-09-15/16/18
7. this handoff

The checkout contains intentional uncommitted work (PRs 1–4 files are
untracked/modified). Preserve it; never reset, clean, broadly stash, reformat
unrelated files, or commit unless the maintainer explicitly requests it.
`.env` is local/ignored; never print or inspect its values.

## PR 4 state you build on

- Platinum reports paired **directional** Student-t intervals
  (`dataflows/platinum.py::_t_critical` owns the margin everywhere; scipy is
  a direct dependency, `pyproject.toml`). Evidence schema v2 is durable:
  12-artifact packages with `evidence.json` (`PlatinumEvidenceIndex`,
  `evidence_schema_version: "2"`), `discovery_fold_metrics.parquet`,
  `selection_steps.json`, `preprocessing.json`; `load_platinum_package`
  reconstructs aggregates, the interval, selected-set membership, greedy
  gating, decision bounds, and preprocessing coverage offline and fails
  closed on mismatch/tamper. v1 packages load with v1 semantics but never
  satisfy v2 reuse fingerprints (`PLATINUM_IDENTITY_SCHEMA_VERSION = "2"` in
  `contracts/identity.py` — Platinum only).
- `ExperimentResult` carries `directional_gain`, `gain_lower_bound`,
  `gain_upper_bound`, and `evaluation_protocol`; tracker config/metrics
  render them; compatibility-protocol results are labeled selection-biased.
- Suite baseline after PR 4: **1064 passed / 9 expected skips / 23 xfailed,
  zero XPASS**.

Marker ownership map — the only remaining plan-23 markers:

| Test module | Owner PR |
|---|---|
| `tests/unit/test_plan23_sandbox_hardening.py` (6 xfail markers: findings 1 and 9) | **PR 5** |
| `tests/unit/test_plan23_llm_cache_identity.py` (xfail markers, findings 8) | PR 6 |

Do not touch `tests/unit/test_plan23_llm_cache_identity.py`. Remove each
PR-5 marker in the same change that makes its test pass; remove any
`# type: ignore` comments that become unused (`warn_unused_ignores` is on).

## Exact next assignment: PR 5 — sandbox containment and bounded worker lifecycle

Implement ADR 0019 as plan §6 PR 5 scopes it, in this order:

1. **AST I/O defense first** (ADR 0019 decision 4; plan §5.1). Extend the
   canonical static policy (`evaluation/sandbox.py` — `BANNED_IMPORTS` is the
   module-level single source; `methods/malmas/pipeline/codegen.py` imports
   it) to reject file-capable APIs exposed by allowed libraries: NumPy
   load/save/fromfile/memmap/data-source APIs and equivalent Pandas
   readers/writers, matching normalized attribute paths **and** terminal
   attribute names so simple aliasing (`np_load = np.fromfile`) cannot
   bypass. Clear unneeded environment variables in the worker; keep the
   library I/O monkeypatches as defense in depth; never expose secrets.
   This unblocks `TestFinding1NumpyAndPandasFileApis` (plan §7.10).
   Interim state after this slice: those tests XPASS-strict (markers still
   present — the main agent removes them); the sentinel/lifecycle tests
   remain xfail.
2. **Strict OS enforcement** (ADR 0019 decisions 2–3; plan §5.2). Load
   trusted runtime modules and the input frame **before** applying
   restrictions; apply a Landlock ruleset permitting only the specific
   temporary input/output paths plus required runtime-library reads; deny
   other filesystem reads/writes; deny sockets via Landlock network rules
   (host has ABI 7) — note the spike's live demo handled filesystem rights
   only and left loopback `connect()` permitted, so network scoping must be
   added explicitly; `no_new_privs` before `landlock_restrict_self`; mask
   handled-access bits to the detected ABI. Strict = filesystem **and**
   network denial; the "strict" label is never granted without socket
   denial. macOS/Windows get a clearly named degraded development profile
   only; the production profile fails closed when strict enforcement is
   unavailable, and degraded execution records that fact in provenance.
   This unblocks `test_execute_cannot_read_external_sentinel` (plan §7.11).
3. **Bounded worker lifecycle** (ADR 0019 decision 5; plan §5.3). An async
   sandbox entry point starts and owns the child process from the event
   loop/main thread (never `multiprocessing.Process.start()` inside
   `asyncio.to_thread`); factor sync and async entry points around one
   process handle/cleanup helper; bound worker launch, input load,
   generated-code execution, output publication, process join, and cleanup;
   kill the child/process group on timeout, wait for termination, close
   queues, remove every temporary file; one authoritative deadline, logged
   with its effective value. Async callers await the async entry point
   directly (`methods/malmas/pipeline/core.py::_exec_sandbox` today wraps
   `asyncio.to_thread`). This unblocks `TestFinding9BoundedLifecycle`
   (plan §7.12: elapsed-time bounds; no live worker process, thread, queue,
   or temp file after timeout or task cancellation).
4. **Adversarial + leak tests** across supported Python versions; degraded
   platforms carry an explicit fail-closed production test. Sandbox
   qualification must run on every OS claimed strict; this host is Linux
   WSL2 kernel 6.18, Landlock ABI 7 (spike report above).

Out of scope for PR 5 (do not start): LLM cache identity v2 (PR 6),
docs/claims audit (PR 7), and any new evaluation-protocol behavior. Do not
regress PRs 2–4 semantics.

## Decision points (maintainer stance already recorded)

1. **Landlock binding** — ADR 0019 consequences pre-sanction a small direct
   dependency (e.g. `python-landlock`) over raw ctypes **if it proves more
   reliable**; "decided at implementation time". Confirm with the maintainer
   before `uv add` if the ctypes path (mirroring
   `scripts/spike_sandbox_isolation.py`, already proven on this host) is
   not clearly worse. Do not add the dependency without that comparison.
2. **Gold v1 read compatibility** (carried from the PR-4 decision record,
   2026-09-16): PR 4 applied versioned-identity acceptance to **Platinum
   only** (the plan charters Platinum compat). **Gold keeps
   load-only-when-regenerated** — pre-ADR Gold packages still fail
   request-fingerprint validation on load, and regeneration is re-running
   the case. If PR 5 touches Gold package loading at all, preserve this
   stance unless the maintainer changes it.
3. **Carry-forward from the PR-4 review (2026-09-18, not a blocker):**
   Platinum materialization has no `existing_package` guard (unlike
   Bronze/Silver in `bronze.py`/`silver.py`), and `LocalArtifactStore.commit`
   atomically replaces the directory at a namespace — so a recompute at the
   same run_id namespace overwrites a previously committed package. Reuse
   rejection (fingerprints) is unaffected, but the overwrite path is worth
   assessing while touching storage/lifecycle code; raise it with the
   maintainer rather than silently changing write-once semantics in PR 5.

## Validation gates

Full set from `AGENTS.md` (`uv sync --all-groups --extra intel`; pytest;
ruff check/format; mypy src; hygiene; docs references) plus plan §8:
`uv run mypy tests`, `uv run python scripts/generate_stage_dag_docs.py
--check` (run without `--check` first if node signatures change — the
sandbox work should not change dataflow nodes), `uv run mkdocs build
--strict`, `uv lock --check`, `git diff --check`. The suite baseline is
1064 passed / 9 expected skips / 23 xfailed; after PR 5 it must be
1064+new passed / 9 skips / 17 xfailed with zero XPASS failures (23 − 6
sandbox markers; the 17 remaining xfails belong to PR 6). Reliably
capturing the pytest summary from piped output: `uv run pytest 2>&1 |
grep -E "[0-9]+ (passed|failed)"`; use `--junitxml=<path>` when
machine-readable results are needed. Plan §8 also asks for the full suite
on Python 3.11/3.12/3.13 with single-thread BLAS/OpenMP when runtime
semantics change — the sandbox lifecycle is such a change; run at least the
sandbox + integration modules on all three versions if the full matrix is
impractical, and record what ran.

## Working style for this thread

- Leverage subagents for implementation and review: glm-5.3-flash `worker`
  agents for well-scoped slices (suggested split: one for the AST I/O
  policy; one for Landlock strict enforcement + degraded/fail-closed
  profiles; one for the bounded async lifecycle + cleanup helper), a
  `reviewer` agent pass before hand-back; the main agent integrates and
  runs the full gates.
- Sequenced single dispatches with main-agent integration between slices —
  all three slices touch `evaluation/sandbox.py` and/or
  `methods/malmas/pipeline/core.py`; never run two workers concurrently.
- Each worker brief must carry: the repo ground rules (below), the exact
  slice design, the pinned per-slice validation commands with their
  expected interim results, and an explicit "do not run the full suite /
  the marker module while a sibling slice is in flight". PR-4 pattern that
  worked: briefs also state which already-landed work sits in the touched
  files so workers do not rework it, and ask for a structured report
  (files changed, validation results, deviations).
- Update `REPORT_LOG.md` (disclose AI assistance) and flip this handoff's
  successor when PR 5 lands; write the PR 6 handoff the same way.

## Main agent owns

- Integration between slices; removal of exactly the six PR-5 xfail markers
  in the same change that keeps their tests passing. Before removing a
  marker, verify the pinned test's arithmetic against the implementation
  (PR 4 found one PR-1 fixture arithmetically unsatisfiable under the
  correct Student-t interval — the fixture was adjusted in the same change,
  preserving the test's intent, and the adjustment documented in the test).
- Any updates to existing non-forbidden tests that pin surface contracts
  the PR changes (PR 4 hit the result-row key-set pins in
  `test_execution_policy.py` and `test_fail_fast_sequential.py`; PR 5 may
  hit timeout/lifecycle pins the same way) — the full-suite gate, not
  worker-scope gates, is where these surface.
- `uv run mypy tests` at integration time (PR 4's worker gate of
  `mypy src` alone missed one `arg-type` error in a worker-written test);
  `# type: ignore` cleanup; DAG-docs regeneration if node signatures
  changed; full gate suite; multi-version evidence recording.
- Applying reviewer follow-ups, then REPORT_LOG/STATE updates and the
  PR-6 handoff (same style, including any carry-forward notes).

## Execution bounds (worker → reviewer dispatch)

Each brief must include the repo ground rules: read `AGENTS.md` first;
surgical edits only; never commit/reset/clean/stash (the checkout
intentionally carries uncommitted PR 1–4 work); never edit
`tests/unit/test_plan23_sandbox_hardening.py`,
`tests/unit/test_plan23_llm_cache_identity.py`, or
`tests/unit/test_plan23_evaluation_integrity.py` (the main agent owns
marker removals); run only the pinned per-slice validation commands, never
the full suite while a sibling slice is in flight.

## Reviewer brief (after integration)

Independent `reviewer` pass over the full PR-5 diff against:

- ADR 0019 decisions 1–6 (literal): threat-model coverage, strict mode =
  filesystem **and** network denial (the "strict" label never granted
  without socket denial), restrictions applied only after trusted modules
  and the input frame load, `no_new_privs` ordering, ABI masking,
  fail-closed production on unsupported platforms, degraded provenance,
  one authoritative deadline per execution with its effective value logged.
- Plan §5.1–§5.3/§6 PR 5/§7.10–§7.12/§9 acceptance: AST policy blocks
  direct and aliased NumPy/Pandas I/O; a generated program cannot read an
  external sentinel or connect a socket (including through allowed
  libraries); launch/execution timeouts stay within bounded allowance and
  leak no process, thread, queue, or temporary file after timeout or task
  cancellation; degraded platforms carry an explicit fail-closed
  production test; each PR independently revertible without silent
  weakening.
- Scope guard: no LLM-cache/PR-6 or docs-claims/PR-7 work; the 17
  remaining xfail markers (PR-6 module) untouched; no new
  evaluation-protocol behavior; PRs 2–4 semantics unregressed.
- Write-once boundaries (ADR 0014): no in-place package rewrites; decision
  point 3 above is reported, not silently changed.
- Honesty: AST checks/monkeypatches/env-scrubbing described as defense in
  depth, never as filesystem isolation; no secret material in worker env,
  logs, or persisted payloads.
- Test quality: adversarial tests genuinely attempt the reads/sockets they
  claim to block; leak tests assert absence of live workers/threads/
  queues/temp files (not just non-raises); marker removals are exactly the
  six sandbox markers; multi-version evidence recorded for what ran.
- Ask for verdict APPROVE/REQUEST-CHANGES with file:line citations and
  one-line confirmations per checklist item (this format worked in PR 4);
  the main agent fixes actionable findings before hand-back and carries
  systemic observations into the PR-6 handoff.
