# Plan 23 PR 4 — Platinum v2 Evidence and Uncertainty Handoff

**Date:** 2026-09-16  
**Status:** Complete 2026-09-18 — successor:
  `docs/handoffs/2026-09-18-plan23-pr5-sandbox-containment.md`  
**Authoritative plan:** `docs/plan/23_evaluation_integrity_security_hardening.md` (§4.5, §6 PR 4, §7 tests 8–9, §8)  
**Governing decision:** ADR 0018 (Accepted 2026-09-15; decision 7 uncertainty, decision 8 evidence schema v2) — implement literally  
**Predecessor:** `docs/handoffs/2026-09-16-plan23-pr3-fold-local-preprocessing.md` (PR 3, complete 2026-09-16)

## Start here

1. `AGENTS.md` (decision order, validation gates)
2. `docs/plan/00_index.md` → item 12
3. `docs/plan/23_evaluation_integrity_security_hardening.md` (§0, §1, §4.5, §6 PR 4, §7, §9)
4. ADR 0018 decisions 7–8, ADR 0013 (medallion boundaries), ADR 0014 (write-once packages)
5. `REPORT_LOG.md` entries dated 2026-09-15/16
6. this handoff

The checkout contains intentional uncommitted work (PRs 1–3 files are
untracked/modified). Preserve it; never reset, clean, broadly stash, reformat
unrelated files, or commit unless the maintainer explicitly requests it.
`.env` is local/ignored; never print or inspect its values.

## PR 3 state you build on

- Fold-local preprocessing lives in
  `src/feature_forge/evaluation/preprocessing.py::FoldPreprocessor`
  (train-fold-only median imputation + ordinal first-appearance encoding,
  documented `-1` unknown-category sentinel, JSON-able `specification()`).
  Platinum's `_evaluate_frame` fits it per fold on training rows and returns
  per-fold specifications; they flow into `aggregate_metrics["preprocessing"]`
  and the manifest's `RunRequest.options["evaluation_provenance"]` — durable,
  first-class persistence is PR 4's job (plan §4.5 "preprocessing identity").
- `dataflows/platinum.py` is partition-aware: candidate evidence + greedy
  selection on discovery folds (`CandidateFoldEvidence` carries the frozen
  `GreedySelectionOutcome` and the post-freeze `selected_arm`); the reported
  baseline/enhanced arms use evaluation folds; rejected winners mirror
  baseline. Fold metrics carry an additive `partition` column and stay
  loadable by `load_platinum_package`.
- `GreedySelectionOutcome.steps` already records, per greedy step: the full
  ranking (`feature`/`directional_gain`/`lower_bound` per candidate),
  `chosen`, `selected_before`, and the chosen arm's `fold_scores` — ready to
  persist as discovery step evidence. Note for reconstruction semantics: the
  per-decision `PlatinumSelectionDecision.lower_bound` field today carries the
  **evaluation-aggregate** bound, not the per-step discovery bound that
  actually gated greedy; PR 4's evidence must persist the discovery step
  metrics so offline reconstruction reproduces the gating (consider extending
  the decision schema or documenting the field's meaning).
- The interval margin is a single constant,
  `platinum._NORMAL_APPROXIMATION_MARGIN = 1.96`, used by both
  `uncertainty_summary` and the greedy `_directional_lower_bound` gate — swap
  it in one place. The PR-3 constraint "1.96 stays" is now lifted: PR 4 owns
  the margin.
- Suite baseline after PR 3: **1039 passed / 9 expected skips / 26 xfailed,
  zero XPASS** (23 xfail-strict remain in the sandbox/cache-identity modules
  owned by PRs 5–6, plus the three interval tests below).

Marker ownership map (`tests/unit/test_plan23_evaluation_integrity.py`) —
findings 2–6 are remediated and unmarked; the only remaining markers are:

| Test | Owner PR |
|---|---|
| `test_interval_uses_student_t` | **PR 4** |
| `test_interval_responds_to_confidence_level` | **PR 4** |
| `test_interval_is_directional_for_minimizing_metrics` | **PR 4** |

Do not touch `tests/unit/test_plan23_sandbox_hardening.py` (PR 5) or
`tests/unit/test_plan23_llm_cache_identity.py` (PR 6). Remove each PR-4
marker in the same change that makes its test pass; remove any `# type:
ignore` comments that become unused (`warn_unused_ignores` is on).

## Exact next assignment: PR 4 — Platinum v2 evidence and uncertainty

Implement ADR 0018 decisions 7–8 exactly as plan §6 PR 4 scopes them:

1. **Directional Student-t intervals.** `uncertainty_summary` computes
   paired **directional** fold deltas (sign-adjusted by
   `metric_direction`), uses the Student-t critical value from the
   configured `uncertainty_policy.confidence_level` and `pair_count - 1`
   degrees of freedom, and derives both bounds from the directional deltas.
   SciPy becomes a direct dependency (`uv add scipy`) per ADR 0018
   decision 7 — accepted by the maintainer 2026-09-16 (decision 1 below);
   it is already present transitively via scikit-learn, so this only
   formalizes it. The existing marker test pins `t_star = 4.302652729911275`
   for 95%/df=2.
2. **Evidence schema v2.** Platinum persists, under a bumped evidence
   schema: discovery candidate/step fold metrics (the per-candidate
   discovery arms and `GreedySelectionOutcome.steps`), final evaluation
   fold metrics (the two reported arms), selected-set membership, policy
   decisions, preprocessing identity (the per-fold
   `FoldPreprocessor.specification()` values), and the directional
   interval. Extend semantic verification so reconstructed decisions and
   aggregates must match persisted fold evidence (offline reconstruction,
   plan §7.9).
3. **v1 read compatibility + v2 reuse rejection.** Old verified v1
   packages remain readable but cannot satisfy v2 reuse fingerprints
   (fingerprint invalidation, never deletion — ADR 0013/0014). Mechanism
   decided by the maintainer 2026-09-16 (decision record below): a
   versioned `identity_schema_version` acceptance — no legacy-recompute
   fallback.
4. **Result/report field updates.** Update result and report fields (and
   the operational verification commands) so user-facing output reflects
   the directional interval and the evaluation-partition estimate;
   compatibility-protocol results stay explicitly labeled selection-biased.

Out of scope for PR 4 (do not start): sandbox containment and bounded worker
lifecycle (PR 5), LLM cache identity v2 (PR 6), docs/claims audit (PR 7).
Do not regress the PR-3 semantics: selection stays frozen on discovery folds
before any evaluation-fold read, and the reported enhanced arm never comes
from a rejected best trial.

## v1 read compatibility — decision record (maintainer accepted 2026-09-16)

Carried from the PR-2/PR-3 reviews: pre-ADR Gold/Platinum packages fail
request-fingerprint validation on **load** (intended reuse invalidation —
no durable Gold/Platinum packages exist on disk today, and reverting PR 2
restores v1 readability). In-place refingerprinting is impossible by
design (write-once, hash-verified packages — ADR 0014); regeneration is
just re-running the case. **Maintainer call (2026-09-16):**

- **Accepted:** a versioned `identity_schema_version` acceptance —
  historical identities stay integrity-bound and loadable while reuse
  stays keyed on the new scheme. A legacy-recompute fallback that would
  silently re-derive old fingerprints is explicitly rejected.
- **Accepted:** PR 4 applies this to **Platinum only** (the plan charters
  Platinum compat); Gold keeps load-only-when-regenerated for now, and
  the Gold stance is recorded in the PR-5 handoff.

## Validation gates

Full set from `AGENTS.md` (`uv sync --all-groups --extra intel`; pytest;
ruff check/format; mypy src; hygiene; docs references) plus plan §8:
`uv run mypy tests`, `uv run python scripts/generate_stage_dag_docs.py --check`
(run without `--check` first if node signatures change), `uv run mkdocs build
--strict`, `uv lock --check`, `git diff --check`. The suite baseline is
1039 passed / 9 expected skips / 26 xfailed; after PR 4 it must be
1039+new passed / 9 skips / 23 xfailed with zero XPASS failures.

## Working style for this thread

- Leverage subagents for implementation and review: glm-5.3-flash `worker`
  agents for well-scoped code/test slices (suggested split: one for the
  directional Student-t interval + margin swap; one for evidence schema v2
  persistence + offline reconstruction/verification; one for v1 read
  compatibility + reuse rejection + result/report fields), a `reviewer`
  agent pass before hand-back; the main agent integrates and runs the full
  gates.
- Update `REPORT_LOG.md` (disclose AI assistance) and flip this handoff's
  successor when PR 4 lands; write the PR 5 handoff the same way.

## Execution bounds (worker → reviewer dispatch)

Sequenced single dispatches with main-agent integration between slices
  (all three touch `dataflows/platinum.py`; never run two workers
  concurrently). Each brief must include the repo ground rules: read
  `AGENTS.md` first; surgical edits only; never commit/reset/clean/stash
  (the checkout intentionally carries uncommitted PR 1–3 work); never edit
  `tests/unit/test_plan23_evaluation_integrity.py` or the PR 5/6 test
  modules (the main agent owns marker removals); run only the pinned
  per-slice validation commands, never the full suite while a sibling
  slice is in flight.

### Decision record (maintainer accepted 2026-09-16 — all three gates cleared)

1. **SciPy direct dependency** — **ACCEPTED: yes, `uv add scipy`**
   (ADR 0018 decision 7 explicitly sanctions it; scipy is already in the
   environment transitively via scikit-learn, so this only formalizes it).
   The pure-Python t-quantile alternative is not taken.
2. **v1 reuse rejection mechanism** — **ACCEPTED: versioned
   `identity_schema_version`** entering the Platinum (and only Platinum)
   input-fingerprint inputs, so every PR-4 request fingerprint differs
   from pre-ADR v1 packages: v1 stays integrity-verified and loadable,
   reuse is keyed on the new identity, and nothing is recomputed or
   rewritten in place (ADR 0014 forbids in-place refingerprinting). The
   legacy-recompute fallback is rejected.
3. **Gold treatment** — **ACCEPTED: Platinum-only in PR 4** (the plan
   charters Platinum compat); Gold keeps load-only-when-regenerated, and
   the stance is recorded in the PR-5 handoff.

### Slice A — directional Student-t intervals (worker)

- `uv add scipy` (decision 1, accepted). In `dataflows/platinum.py`: replace
  `_NORMAL_APPROXIMATION_MARGIN` with
  `_t_critical(confidence_level: float, degrees_of_freedom: int) -> float`
  (`scipy.stats.t.ppf((1 + confidence_level) / 2, df)`, typed, docstring
  noting ADR 0018 decision 7).
- `uncertainty_summary`: compute directional deltas
  (`fold_deltas * (+1 maximize / -1 minimize)`); `standard_deviation`,
  `standard_error`, `lower_bound`, `upper_bound` all directional; margin
  `t(confidence_level, pair_count - 1)`; keep the
  `minimum_successful_folds` `DatasetError` and the field semantics of
  `mean_directional_gain`. The three PR-4 marker tests pin the exact
  contract (t* = 4.302652729911275 for 95%/df=2; 80% vs 99% margins
  differ; minimizing metrics get positive-direction bounds).
- Thread `confidence_level` into the greedy gate:
  `_directional_lower_bound(deltas, confidence_level)` called with
  `platinum_request.uncertainty_policy.confidence_level` (tightens PR-3's
  1.96 gate; `test_require_positive_lower_bound_enforced` must still
  pass — df=1 gives t=12.706, an even more negative bound).
- NEW focused unit tests (e.g. `tests/unit/test_platinum_uncertainty.py`)
  for `_t_critical` and directional bounds; no edits to existing test
  files.
- Interim state after A: the three `test_interval_*` tests XPASS-strict
  (markers still present — main agent removes them),
  `tests/unit/test_plan23_evaluation_integrity.py` otherwise unchanged;
  `tests/unit/test_platinum_dataflow.py` still 4 passed; mypy/ruff green
  on touched files.

### Slice B — evidence schema v2 persistence + offline reconstruction (worker)

- Persist under a v2 evidence marker (recommended shape: a new required
  `evidence.json` carrying `evidence_schema_version: "2"`; per-artifact
  `schema_version` fields stay `"1"` — bumping every contract Literal is
  churn without benefit): `discovery_fold_metrics.parquet` (per-candidate
  discovery arms + greedy-step arms, arm-labeled),
  `selection_steps.json` (`GreedySelectionOutcome.steps` records),
  `preprocessing.json` (the per-fold `FoldPreprocessor.specification()`
  values for the reported arms), alongside the existing artifacts.
- Keep `PLATINUM_REQUIRED_ARTIFACTS`, the staging writes, and
  `load_platinum_package` consistent: v2 packages load and reconstruct
  (aggregate scores, directional interval, selected-set membership, greedy
  step gains) from persisted fold evidence alone; tampering with any
  persisted artifact fails verification/`DatasetError` (plan §7.9).
- v1 read compatibility: packages without `evidence.json` still load with
  today's v1 semantics (construct one in tests via the pre-PR-4 artifact
  set) but are never reusable (slice C makes that structural).
- `platinum_materialization`/`platinum_manifest` node signatures stay
  unchanged where possible; if they change, the main agent regenerates
  `docs/generated/stage_dags.md`.
- NEW tests alongside `tests/unit/test_platinum_dataflow.py` (e.g.
  `test_platinum_evidence_v2.py`): full-chain persistence → offline
  reconstruction → tamper detection → v1 read compat. Interim state: all
  new tests pass; existing suites unchanged (interval markers still on).

### Slice C — reuse rejection + result/report fields (worker)

- `contracts/identity.py`: `platinum_input_fingerprint` gains the
  `identity_schema_version` input (decision 2, accepted) — every Platinum
  fingerprint changes exactly once, invalidating cross-version reuse
  without touching Gold/Silver identities (decision 3, accepted). Add the
  version to
  cache/manifest provenance where the fingerprint already appears.
- Result/report fields: `ExperimentResult`
  (`src/feature_forge/experiment/execution.py:54`) and the executor's
  construction of it (`hamilton_executor.py` ~line 523) — report the
  directional gain and interval bounds (e.g. `directional_gain`,
  `gain_lower_bound`/`gain_upper_bound` or equivalent typed fields;
  `legacy_gain` may remain for compat but must not be the headline), and
  keep the evaluation-partition estimate explicit (compatibility-protocol
  results stay labeled selection-biased wherever results/manifests are
  rendered).
- Operational verification commands: update `docs/operations.md` (and
  `scripts/qualify_hamilton_cache.py` only if it asserts on changed
  fields) so offline verification reflects the v2 evidence set.
- NEW tests for the fingerprint invalidation (v1 package never reused,
  still loadable) and the result fields; e2e
  (`tests/integration/test_platform_e2e.py`) must stay green untouched.

### Main agent owns

- Integration between slices; removal of exactly the three PR-4 xfail
  markers in the same change that keeps them passing; `# type: ignore`
  cleanup; DAG-docs regeneration if node signatures changed; full gate
  suite; REPORT_LOG/STATE updates; PR-5 handoff (same style, including the
  Gold-treatment stance from decision 3 and any carry-forward notes).

### Reviewer brief (after integration)

Independent `reviewer` pass over the full PR-4 diff against: ADR 0018
  decisions 7–8 (literal), plan §4.5/§6 PR 4/§7.8–7.9/§9; scope guard (no
  sandbox/PR-5 or cache-identity/PR-6 work; the 23 remaining xfail markers
  in those modules untouched; uncertainty margin semantics now owned by
  `_t_critical` everywhere); write-once boundaries (ADR 0014: no in-place
  refingerprinting, atomic staging/commit only); read-compat honesty (v1
  readable + integrity-verified + never reused; no silent legacy
  recompute); determinism (t-critical pure function of inputs; no
  timestamps/randomness in fingerprints); call-site compatibility
  (`_io.build_platinum_request`, executor request construction ~line 471,
  `platinum_request` node round-trip, qualify script); leakage re-check
  (nothing in evidence v2 exposes evaluation-fold scores to selection);
  test quality (tamper tests genuinely mutate persisted evidence; marker
  removals exactly the three interval tests).

### Final gates + arithmetic

All AGENTS.md + plan §8 gates. Suite must end at **1039+new passed /
  9 expected skips / 23 xfailed, zero XPASS** (26 − 3 unmarked PR-4
  markers; the 23 remaining xfails belong to PRs 5–6).
