# Plan 23 PR 3 — Fold-Local Preprocessing and Selection Handoff

**Date:** 2026-09-16  
**Status:** Complete — landed 2026-09-16; successor:
`docs/handoffs/2026-09-16-plan23-pr4-platinum-v2-evidence-uncertainty.md`  
**Authoritative plan:** `docs/plan/23_evaluation_integrity_security_hardening.md` (§4.3–4.4, §6 PR 3, §7 tests 2/3/6/7)  
**Governing decision:** ADR 0018 (Accepted 2026-09-15; decisions 4–6) — implemented literally  
**Predecessor:** `docs/handoffs/2026-09-15-plan23-pr2-partition-aware-discovery.md` (PR 2, complete 2026-09-16)

## Start here

1. `AGENTS.md` (decision order, validation gates)
2. `docs/plan/00_index.md` → item 12
3. `docs/plan/23_evaluation_integrity_security_hardening.md` (§0, §1, §4.3–4.4, §6 PR 3, §7, §9)
4. ADR 0018 decisions 4–6, ADR 0013 (medallion boundaries), ADR 0014 (atomic packages)
5. `REPORT_LOG.md` entries dated 2026-09-15/16
6. this handoff

The checkout contains intentional uncommitted work (PRs 1–2 files are
untracked/modified). Preserve it; never reset, clean, broadly stash, reformat
unrelated files, or commit unless the maintainer explicitly requests it.
`.env` is local/ignored; never print or inspect its values.

## PR 2 state you build on

- Silver `fold_assignments` carry `partition` labels (`discovery`/`evaluation`);
  under the default `holdout` protocol both partitions independently cover
  folds `0..cv_folds-1` (duplicate fold ids across partitions are expected —
  disambiguate by partition). Small data fails closed under `holdout`;
  `compatibility` (`fraction=0.0`) is all-`discovery` and explicit-only.
- `EvaluationPolicy.evaluation_protocol` exists (contracts/platinum.py);
  the executor threads it from `Settings.evaluation.protocol`. Silver/Gold/
  Platinum fingerprints already distinguish protocols.
- Methods are fit on discovery rows only
  (`hamilton_executor._fit_method_on_discovery`); the Gold scope contract
  (`evaluation/scope.py`, `tests/unit/test_scope_contract.py`) accepts only
  probe-verified row-local candidates under holdout — accepted features are
  row-local by construction, which simplifies fold-local preprocessing.
- Suite baseline after PR 2: **1021 passed / 9 expected skips / 34 xfailed,
  zero XPASS** (11 `xfail(strict=True)` + 1 pin in
  `tests/unit/test_plan23_evaluation_integrity.py`, 23 more in the sandbox /
  cache-identity modules owned by PRs 5–6).

Marker ownership map (`tests/unit/test_plan23_evaluation_integrity.py`):

| Test | Owner PR |
|---|---|
| `test_preprocessing_uses_training_fold_statistics` | **PR 3** |
| `test_unseen_validation_category_follows_documented_policy` | **PR 3** |
| `test_candidate_evidence_excludes_evaluation_partition` | **PR 3** |
| `test_reported_enhanced_uses_evaluation_partition_only` | **PR 3** |
| `test_rejected_winner_mirrors_baseline` | **PR 3** |
| `test_selection_partition_validation` | **PR 3** |
| `test_require_positive_lower_bound_enforced` | **PR 3** |
| `test_greedy_selection_combines_complementary_features` | **PR 3** |
| `test_max_selected_features_caps_selection` (pin — extend, do not merely keep) | **PR 3** |
| `test_interval_uses_student_t` | PR 4 |
| `test_interval_responds_to_confidence_level` | PR 4 |
| `test_interval_is_directional_for_minimizing_metrics` | PR 4 |

Do not touch `tests/unit/test_plan23_sandbox_hardening.py` (PR 5) or
`tests/unit/test_plan23_llm_cache_identity.py` (PR 6). Remove each PR-3
marker in the same change that makes its test pass; remove any `# type:
ignore` comments that become unused (`warn_unused_ignores` is on).

## Exact next assignment: PR 3 — Fold-local preprocessing and selection

Implement ADR 0018 decisions 4–6 exactly as plan §6 PR 3 scopes them:

1. **Fold-local preprocessing.** Replace `dataflows/platinum.py::_prepared`
   (whole-frame imputation/encoding before fold splitting) with a typed
   fold-local transformer (plan §4.3): per fold, fit numeric imputation and
   categorical encoding on training rows only; transform validation rows
   with the documented stable unknown-category sentinel (`-1`); retain the
   fitted preprocessing specification in evaluation provenance. Use a
   scikit-learn `Pipeline`/`ColumnTransformer` or an equivalently typed small
   helper — no second evaluation framework. The existing spy tests
   (`_register_spy` pattern) pin the exact train/validation row semantics.
2. **Discovery-fold candidate evidence.** `candidate_fold_evidence` scores
   candidates on discovery-partition folds only (evaluation rows never enter
   selection evidence). Selection-decision code must be unable to read
   evaluation-fold scores until selection freezes (plan §7.2 spy test).
3. **Evaluation-fold reporting.** After selection freezes, exactly two arms
   (baseline and the combined selected set) are evaluated on
   evaluation-partition folds only; the persisted enhanced arm is this
   accepted combination. When no candidate qualifies (or the policy rejects
   the numerical winner), the enhanced arm mirrors baseline and gain is zero
   — never report a rejected best trial (plan §7.7).
4. **Greedy selection on discovery folds** (plan §4.4): evaluate each
   remaining candidate appended to the selected set; rank by mean directional
   gain with a stable feature-name tie-break; require
   `minimum_practical_gain`; when `require_positive_lower_bound` is set,
   require a positive directional lower confidence bound (enforce against the
   current `uncertainty_summary.lower_bound` — PR 4 later swaps the 1.96
   margin for Student-t, which only tightens it); stop at
   `max_selected_features` or when nothing qualifies.
   `selection_partition="discovery"` is required under `holdout`; unsupported
   partition/profile combinations fail during `PlatinumRequest` validation
   (compatibility keeps today's behavior, explicitly).
5. **Tests.** Remove exactly the eight PR-3 xfail markers and extend the
   `max_selected_features` pin into real greedy-cap coverage. Add the
   selection-freeze spy (plan §7.2). Keep fold metrics schemas compatible
   with `load_platinum_package` reconstruction (schema v2 evidence is PR 4).
6. **Keep the change independently revertible** (plan §9); no sandbox or
   cache changes, no Student-t/evidence-v2 work (PR 4), no new fingerprint
   fields beyond what selection semantics require.

Out of scope for PR 3 (do not start): Student-t intervals, Platinum evidence
schema v2 / v1-reuse rejection, result/report field updates (PR 4), sandbox
containment (PR 5), LLM cache identity (PR 6), docs/claims audit (PR 7).

## Carry into the PR-4 handoff

- Open decision from the PR-2 review: pre-ADR Gold/Platinum packages now
  fail request-fingerprint validation on **load** (intended reuse
  invalidation — note no durable Gold/Platinum packages exist on disk today,
  and reverting PR 2 restores v1 readability). In-place refingerprinting is
  impossible by design (write-once, hash-verified packages — ADR 0014);
  regeneration is just re-running the case. When writing the PR-4 handoff,
  record the maintainer's call on PR 4's planned "v1 read compatibility":
  prefer a versioned `identity_schema_version` acceptance (historical
  identities stay integrity-bound and loadable while reuse stays keyed on
  the new scheme) over a legacy-recompute fallback, and decide whether Gold
  gets the same treatment (the plan only charters Platinum compat).

## Validation gates

Full set from `AGENTS.md` (`uv sync --all-groups --extra intel`; pytest;
ruff check/format; mypy src; hygiene; docs references) plus plan §8:
`uv run mypy tests`, `uv run python scripts/generate_stage_dag_docs.py --check`
(run without `--check` first if node signatures changed), `uv run mkdocs build
--strict`, `uv lock --check`, `git diff --check`. The suite baseline is
1021 passed / 9 expected skips / 34 xfailed; after PR 3 it must be
1021+new passed / 9 skips / 26 xfailed with zero XPASS failures.

## Working style for this thread

- Leverage subagents for implementation and review: glm-5.3-flash `worker`
  agents for well-scoped code/test slices (suggested split: one for the
  fold-local preprocessing helper; one for partition-aware candidate/baseline
  evidence + evaluation-fold reporting; one for greedy selection + policy
  validation + marker removals), a `reviewer` agent pass before hand-back;
  the main agent integrates and runs the full gates.
- Update `REPORT_LOG.md` (disclose AI assistance) and flip this handoff's
  successor when PR 3 lands; write the PR 4 handoff the same way.
