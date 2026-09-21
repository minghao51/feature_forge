# Plan 23 PR 2 — Partition-Aware Discovery Handoff

**Date:** 2026-09-15  
**Status:** Complete (landed 2026-09-16; suite 1021 passed / 9 skips / 34
xfailed, zero XPASS) — superseded by
`docs/handoffs/2026-09-16-plan23-pr3-fold-local-preprocessing.md`  
**Authoritative plan:** `docs/plan/23_evaluation_integrity_security_hardening.md` (§4.1–4.2, §6 PR 2, §7 tests 1/4/5)  
**Governing decision:** ADR 0018 (Accepted 2026-09-15) — implement literally  
**Predecessor:** `docs/handoffs/2026-09-15-evaluation-security-audit-remediation.md` (PR 1, complete)

## Start here

1. `AGENTS.md` (decision order, validation gates)
2. `docs/plan/00_index.md` → item 12
3. `docs/plan/23_evaluation_integrity_security_hardening.md` (§0, §1, §4.1–4.2, §6 PR 2, §7, §9)
4. ADR 0018, ADR 0013 (medallion boundaries), ADR 0014 (atomic packages)
5. `REPORT_LOG.md` entries dated 2026-09-15
6. this handoff

The checkout contains intentional uncommitted work (PR 1 files are untracked).
Preserve it; never reset, clean, broadly stash, reformat unrelated files, or
commit unless the maintainer explicitly requests it. `.env` is local/ignored;
never print or inspect its values.

## PR 1 state you build on

- `tests/unit/test_plan23_evaluation_integrity.py` — 15 tests for findings
  2–7: 14 `xfail(strict=True)` + 1 pin. **Each xfail marker is owned by the PR
  that fixes its finding**; markers must be removed in the same change that
  makes the test pass (strict markers XPASS-fail otherwise).
- `tests/unit/test_plan23_sandbox_hardening.py` (findings 1/9 → PR 5) and
  `tests/unit/test_plan23_llm_cache_identity.py` (finding 8 → PR 6) — do not
  touch in PR 2.
- `docs/spikes/2026-09-15-sandbox-os-isolation.md` + evidence (PR 5 input).
- ADRs 0018–0020 Accepted; plan 23 header is Active.

Marker ownership map:

| Test (in `test_plan23_evaluation_integrity.py`) | Owner PR |
|---|---|
| `test_settings_expose_evaluation_protocol` | **PR 2** |
| `test_executor_does_not_force_zero_holdout` (source pin — replace with a behavioral test) | **PR 2** |
| `test_holdout_fails_closed_for_small_datasets` | **PR 2** |
| `test_preprocessing_uses_training_fold_statistics` | PR 3 |
| `test_unseen_validation_category_follows_documented_policy` | PR 3 |
| `test_candidate_evidence_excludes_evaluation_partition` | PR 3 |
| `test_reported_enhanced_uses_evaluation_partition_only` | PR 3 |
| `test_rejected_winner_mirrors_baseline` | PR 3 |
| `test_selection_partition_validation` | PR 3 |
| `test_require_positive_lower_bound_enforced` | PR 3 |
| `test_greedy_selection_combines_complementary_features` | PR 3 |
| `test_max_selected_features_caps_selection` (pin — extend, do not merely keep) | PR 3 |
| `test_interval_uses_student_t` | PR 4 |
| `test_interval_responds_to_confidence_level` | PR 4 |
| `test_interval_is_directional_for_minimizing_metrics` | PR 4 |

Also remove the `# type: ignore[attr-defined]` comments that become unused when
markers come off (repo runs mypy with `warn_unused_ignores`).

## Exact next assignment: PR 2 — Partition-aware discovery

Implement ADR 0018 decisions 1–3 exactly as plan §6 PR 2 scopes them:

1. **Typed settings.** Add `protocol: Literal["holdout", "compatibility"] =
   "holdout"` and an evaluation-holdout fraction (default `0.25`, `0 < f < 0.5`)
   to `Settings.evaluation` (`src/feature_forge/config.py`). Both enter the
   resolved configuration and the Silver dataset fingerprint
   (`dataflows/silver.py::dataset_fingerprint_value`, already keyed on
   `evaluation_holdout_fraction` — extend to the protocol).
2. **Stop hard-coding the zero holdout.** `experiment/hamilton_executor.py`
   (~line 251) must thread the configured protocol/fraction into
   `DatasetRequest` instead of forcing `evaluation_holdout_fraction=0.0`.
   Replace the source-pin test with a behavioral one.
3. **Fail closed.** Under `holdout`, partition creation raises an actionable
   `DatasetError` when either partition cannot independently host `cv_folds`
   rows (`evaluation/holdout.py::resolve_partition` currently degrades
   silently; `assign_row_partitions` and `silver.fold_assignments` checks
   follow). `compatibility` keeps today's all-row behavior, explicitly
   selected only.
4. **Discovery-boundary fitting.** Gold fits methods only on rows whose Silver
   partition is `discovery`; evaluation-partition rows and targets never reach
   method fitting, candidate trials, or selection (spy-test per plan §7.1).
   Generated code may still execute on the full feature frame after fitting,
   but never receives evaluation targets.
5. **Scope contract for generated transformations.** `row_local` candidates
   must pass deterministic row-subset/permutation metamorphic checks before
   acceptance (note: `tests/unit/test_metamorphic.py` exists — reuse its
   patterns); candidates requiring whole-frame statistics (`fitted`) are
   rejected in holdout mode until that contract exists (plan §7.4).
6. **Partition provenance.** Thread partition labels through Gold and Platinum
   requests/fingerprints (`contracts/gold.py`, `contracts/platinum.py`,
   `dataflows/_io.py::build_platinum_request`, `contracts/identity.py`) so
   reuse chains distinguish protocols. Fingerprints invalidate reuse; do not
   delete old packages (ADR 0014).
7. **Tests.** Remove exactly the three PR-2 markers above; add the new
   behavioral/fail-closed/scope-contract tests. Compatibility mode remains
   explicit and reproducible (plan §7.5: small data fails closed under
   holdout, succeeds only in explicit compatibility).

Out of scope for PR 2 (do not start): fold-local preprocessing, greedy
selection, evaluation-fold reporting (PR 3), Student-t intervals/evidence v2
(PR 4), anything sandbox or cache (PR 5/6). Keep the change independently
revertible (plan §9).

## Validation gates

Full set from `AGENTS.md` (`uv sync --all-groups --extra intel`; pytest;
ruff check/format; mypy src; hygiene; docs references) plus plan §8:
`uv run mypy tests`, `uv run python scripts/generate_stage_dag_docs.py --check`,
`uv run mkdocs build --strict`, `uv lock --check`, `git diff --check`.
The suite baseline is 1003 passed / 9 expected skips / 37 xfailed; after PR 2
it must be 1003+new passed / 9 skips / 34 xfailed with zero XPASS failures.

## Working style for this thread

- Leverage subagents for implementation and review: glm-5.3-flash `worker`
  agents for well-scoped code/test slices, a `reviewer` agent pass before
  hand-back; the main agent integrates and runs the full gates.
- Update `REPORT_LOG.md` (disclose AI assistance) and flip this handoff's
  successor when PR 2 lands; write the PR 3 handoff the same way.
