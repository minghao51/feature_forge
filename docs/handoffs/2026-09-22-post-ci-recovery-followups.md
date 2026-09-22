# Post-CI-Recovery Follow-Ups: Method Record Contracts, Sandbox Guard Hardening, and Security Policy Runbook

**Date:** 2026-09-22  
**Status:** Ready for execution  
**Predecessor:** `docs/handoffs/2026-09-21-pr1-salvage-ci-recovery-closeout.md` (Complete 2026-09-22)  
**Base:** `main` at `9d42f6c` (CI run `35731276754` fully green, 13/13 jobs)  
**Decision owner:** Maintainer  

This handoff executes the follow-up items documented during the PR #1
salvage/CI-recovery session. Every item below was surfaced by a reviewer or
worker during Slices B–E and deliberately deferred to keep those slices
separable. None is a regression; all are small, independently landable
hardening/completeness work.

## 1. Read first / authority order

1. `AGENTS.md` and `docs/README.md`.
2. The predecessor handoff (context for why these items exist).
3. Accepted ADRs:
   - `docs/decisions/0002-three-tier-agent-memory.md`
   - `docs/decisions/0017-failure-policy-and-cancellation-contract.md`
   - `docs/decisions/0019-sandbox-containment-bounded-lifecycle.md`
4. This handoff and `docs/handoffs/README.md`.

Where an item conflicts with an accepted ADR, the ADR wins. Nothing in this
handoff reopens an ADR boundary.

## 2. Verified findings (anchors current at `9d42f6c`)

### 2.1 Legacy iteration-failure shapes in caafe and llmfe

`src/feature_forge/methods/caafe/method.py:181-184` and
`src/feature_forge/methods/llmfe/method.py:145-148` still write
`iteration_record["error"] = str(exc)` with no `gains` guarantee — the exact
defect fixed for Malmus in `5d2cd42` (the CI `KeyError: 'gains'` mask).
`IterationRecord` in `src/feature_forge/methods/base.py:41-46` documents
this as "pending migration".

### 2.2 malmas feature-failure records retain no root cause

`src/feature_forge/methods/malmas/pipeline/core.py:420-439` builds
`feature_failures` entries with `feature`/`phase`/`reason_code` only; the
sandbox-timeout branch (`core.py:70-76`) logs `timeout=` and returns `None`,
which downstream maps to the generic `reason_code: "execution_failed"` — the
original exception type/message is discarded from artifacts.

### 2.3 Sandbox escape-surface stubs leak module globals via stringification

Reviewer finding (Slice B step 4 re-review): runtime guards replace
`np.ctypeslib`/`np.ctypes` with a plain blocked *function*
(`src/feature_forge/evaluation/sandbox.py:1704-1710` `patch()` → `blocked`).
`"{0.ctypeslib.__globals__}".format(np)` therefore stringifies worker module
globals into generated-code error text — information disclosure into logs,
not an escape (no object exfiltration via `format`), but cheap to close with
a `__getattr__`-raising sentinel object.

### 2.4 numpy.ctypeslib preload assumption is undocumented

`src/feature_forge/evaluation/sandbox.py:1746-1760`: the submodule re-import
closure assumes `numpy.ctypeslib` is already in `sys.modules` at guard time
(true on current numpy; pinned by
`test_runtime_guard_survives_ctypeslib_reimport`). A future lazy-loading
numpy would let a runtime `import numpy.ctypeslib` load a fresh unpatched
submodule; the AST layer still covers that path, but the assumption deserves
a comment so nobody "simplifies" the guard away.

### 2.5 BLOCKED_ESCAPE_PATHS is a dead policy mirror

`sandbox.py:~804-816`: `BLOCKED_ESCAPE_PATHS` (np/numpy/pd/pandas dotted
forms) is fully subsumed by terminal-name matching in
`_blocked_escape_reason`. Dead but harmless; mirrors the auditable-policy
pattern of `BLOCKED_IO_PATHS`.

### 2.6 Sandbox memory-limit defaults disagree

`SandboxLimits.max_memory_mb=512` (`sandbox.py:448-450`) vs
`SandboxedExecutor.__init__ max_memory_mb=2048` (`:802-803`). The dataclass
default is dead — the only construction site passes executor values
(`:825`). Confusing to readers; the operational default since the thread-cap
runner policy (`152d2d7`) is 2048.

### 2.7 Security policy has no operational runbook; main has no branch protection

When `security`/`security-tooling` goes red on a new advisory (pip-audit DB
drift is expected — 51→53 advisories moved during a single session), the
response pattern (targeted `uv lock --upgrade-package`, else a 6-field
allowlist entry) lives only in REPORT_LOG history. `main` has no branch
protection at all (verified via API, 2026-09-22), so no check is required.

## 3. Worker/reviewer workflow

- Worker: `opencode-go/deepseek-v4.1-flash` (full tools, no commits).
- Reviewer: `zai/glm-5.3-flash`, read-only (`git diff` inspection).
- Main agent owns any change to `tests/unit/test_plan23_sandbox_hardening.py`
  (none anticipated; if a slice needs one, STOP and report).
- Never run sandbox suites concurrently — shared `/tmp` causes false leak
  failures.
- Commit messages use the suggested forms below; one concern per commit.

## 4. Execution plan

### Slice F1 — typed failure records across caafe, llmfe, and malmas

Extends the `5d2cd42` contract to the remaining methods. Additive only; no
ADR 0017 semantics change (methods keep their current continue/fail-fast
behavior).

1. **caafe/llmfe**: adopt `BaseMethod._record_iteration_failure`
   (`methods/base.py:183-194`): initialize `gains`/`kept` at record
   construction, record partial measurable gains before any fallible
   post-measurement step, replace `error = str(exc)` with the typed payload,
   add `error_type` to the warning logs (`caafe_iteration_failed`,
   `llmfe_iteration_failed`).
2. **malmas**: extend `feature_failures` entries with a typed
   `error: {"type": ..., "message": ...}` payload (keep `reason_code` for
   compatibility); thread the original exception through the
   `_exec_sandbox` timeout/`CodeExecutionError` branches
   (`core.py:70-84`) so the artifact retains the root cause the log already
   has.
3. Update the `IterationRecord` docstring (`base.py:41-46`) — remove the
   "pending migration" caveat once all four iterative methods conform.
4. **Tests**: new `tests/unit/test_iteration_record_contract.py` covering
   caafe + llmfe (fake-executor/fake-LLM doubles; factor the `test_malmus.py`
   `_FakeExecutor`/`FakeJsonLLM` pattern into a shared helper only if that
   stays clean — do not regress `test_malmus.py`) and malmas
   `feature_failures` payload (timeout branch carries
   `type: "SandboxTimeoutError"` + message). Pin `record["gains"]` access on
   every recorded path per method.

Gates (sequential): new contract file, `test_malmus.py`,
`test_malmas_method.py`, `test_iterative_pipeline.py`, full `uv run pytest`.

Suggested commit: `fix(methods): typed iteration failures across caafe, llmfe, malmas`

### Slice F2 — sandbox guard hardening (B4 review nits)

Security-adjacent, smallest slice — do it first if trimming.

1. Replace the plain blocked-function stub with a `__getattr__`-raising
   sentinel class: attribute access (including `__globals__`) raises
   `AttributeError`; `__repr__`/`__str__` return a static benign string
   (e.g. `"blocked-by-sandbox-policy"`); callable sentinel (for names invoked
   directly) raises the same typed path. All existing escape/IO-defense tests
   must pass unchanged; the pinned failure messages that reference the stub
   may need wording updates in allowed test files only.
2. Add the preload-assumption comment at the `sys.modules` submodule patch
   (`sandbox.py:1746-1760`).
3. `BLOCKED_ESCAPE_PATHS`: keep as the auditable policy mirror with a
   one-line comment stating it is deliberately redundant with terminal
   matching (mirrors `BLOCKED_IO_PATHS`); do not remove.
4. Pin the disclosure closure: a test asserting that
   `"{0.ctypeslib.__globals__}".format(np)` (and a `__dict__` variant) yields
   only the sentinel's benign text, never module globals, in generated-code
   error output.

Gates (sequential): `test_sandbox_io_defense.py`, `test_sandbox.py`,
`test_sandbox_memory.py`, `test_sandbox_lifecycle.py`,
`test_sandbox_containment.py`.

Suggested commit: `fix(sandbox): non-stringifying escape-surface sentinels`

### Slice F3 — sandbox limits default alignment

1. Align `SandboxLimits.max_memory_mb` to `2048` (the operational default
   since `152d2d7`/`9b783fa`) with a docstring line pointing at the
   thread-cap runner policy origin; OR remove the dead default in favor of
   explicit construction — implementer's judgment, state which in the report.
2. Verify no other `SandboxLimits(...)` construction sites exist
   (`:825` is the only one in `src/`; check `tests/` and update any direct
   constructions).
3. Behavior-neutral by construction; full unit suite gate suffices plus the
   sandbox suites if any default-asserting test exists.

Suggested commit: `chore(sandbox): align limits dataclass default with executor default`

### Slice F4 — security policy runbook and maintainer checklist

1. Add `security/README.md`: the red-lane response runbook — (a) classify the
   advisory surface (runtime `--no-default-groups` closure / dev env /
   all-extras closure), (b) prefer targeted
   `uv lock --upgrade-package <name>` fixes, (c) an allowlist entry is the
   last resort and must carry all six fields per
   `security/pip_audit_allowlist.txt`'s header, (d) the diskcache
   CVE-2025-69872 expiry condition (upstream release >5.6.3 without
   pickle-by-default) and how to re-check it. Reference REPORT_LOG
   2026-09-21/22 entries as the worked example.
2. Maintainer checklist (GitHub settings; not agent-executable without
   explicit authorization): protect `main`; require at minimum `test (3.13)`,
   `security`, `security-tooling`, `hygiene`, `docs`; sketch the
   `gh api -X PUT repos/<repo>/branches/main/protection` shape in the
   checklist for convenience.
3. REPORT_LOG entry.

Suggested commit: `docs(security): dependency-audit runbook and branch-protection checklist`

## 5. Validation and acceptance

Per-slice gates above, then before handoff:

```bash
uv sync --all-groups --extra intel
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src && uv run mypy tests
uv run python scripts/check_repo_hygiene.py
uv run python scripts/check_docs_references.py
uv run mkdocs build --strict
uv lock --check
git diff --check
```

- [ ] caafe/llmfe/malmas failure records carry `gains`/`kept` (where the
      record shape has them) + typed root cause; `IterationRecord` docstring
      no longer lists pending migrations.
- [ ] Sandbox sentinel cannot be stringified into module globals; pinned by
      test; preload assumption documented; escape matrix still green.
- [ ] `SandboxLimits` default matches the executor default or is explicitly
      removed; no silent default change (construction always explicit).
- [ ] `security/README.md` runbook exists and matches the lanes'
      actual behavior; maintainer branch-protection checklist recorded.
- [ ] Full local suite + every gate above green; CI green on the pushed
      result (test lanes, both hamilton cells, security, security-tooling).
- [ ] REPORT_LOG updated; `tests/unit/test_plan23_sandbox_hardening.py`
      untouched (or changed only by the main agent).

## 6. Out of scope

- Predecessor handoff §5 deferred architecture candidates
  (`BaseMethod.from_run_context`, full `ResourceConfig`/`EffectiveResourcePlan`
  application, automatic lower-concurrency retry, implicit sequential
  fallback, DuckDB catalog / Astro UI, verification-service port) — each
  still needs demonstrated demand and, where boundary-touching, a new ADR.
- Deadline-aware pipe drain loop (B3 residual): only on measured evidence of
  transfer-dominated latencies; the 4096-char source cap bounds the worst
  case today.
- Deleting `feat/medallion-refactor` (separate, explicitly authorized
  operation; tag fetchability must be re-verified first).
- Plan 23 PR 6 / ADR 0020 LLM cache identity work (specified in
  `docs/handoffs/2026-09-18-plan23-pr6-llm-cache-identity-v2.md`).
- Any ADR reopening (0002, 0010, 0013, 0016–0019) or supplied-data change.

## 7. Provenance

Items 2.1–2.7 were recorded during the 2026-09-21/22 salvage session by the
worker (`opencode-go/deepseek-v4.1-flash`), the reviewer
(`zai/glm-5.3-flash`), and main-agent verification; anchors re-verified
against `9d42f6c` on 2026-09-22. This handoff drafting session changed only
documentation.
