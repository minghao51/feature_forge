# ADR 0018: Independent discovery and evaluation protocol

- **Status:** Accepted (maintainer accepted on 2026-09-15)
- **Date:** 2026-09-15
- **Related:** ADR 0001, ADR 0009, ADR 0013, ADR 0017,
  `docs/plan/23_evaluation_integrity_security_hardening.md`,
  `tests/unit/test_plan23_evaluation_integrity.py`, `REPORT_LOG.md`

## Context

The 2026-09-14 audit (plan 23 §1) confirmed that the reported score is not
independent of feature discovery and selection: the Hamilton executor forces
`evaluation_holdout_fraction=0.0` and fits methods on every Silver row; Silver
already labels discovery/evaluation partitions but Platinum ignores them;
`_prepared` imputes and encodes the complete frame before fold splitting;
candidate selection and the reported aggregate read the same folds; the
reported enhanced arm mirrors the numerically best candidate even when the
selection policy rejects it; declared policy fields (`selection_partition`,
`require_positive_lower_bound`, `max_selected_features`) are dormant; and the
paired interval hard-codes a 1.96 normal margin, ignores the configured
confidence level, and applies raw deltas for minimizing metrics. Each gap has
a deterministic characterization test (plan 23 PR 1).

On 2026-09-15 the maintainer accepted this ADR, unlocking the runtime
evaluation changes (plan 23 PRs 2–4).

Alternatives considered: keep all-row selection (status quo — selection-biased
reporting), or full nested cross-validation (unbiased but multiplies LLM and
evaluation cost per case). The two-partition protocol buys independence at a
fixed row budget and keeps one trained model per fold.

## Decision

1. **Protocol profiles.** `Settings.evaluation` gains
   `protocol: Literal["holdout", "compatibility"] = "holdout"` and an
   evaluation-holdout fraction defaulting to `0.25` (`0 < f < 0.5`). The
   protocol and fraction enter the Silver dataset fingerprint and the resolved
   configuration.
2. **Fail-closed holdout.** Under `holdout`, partition creation raises an
   actionable `DatasetError` when either partition cannot independently host
   `cv_folds` rows; it never silently falls back to all-row discovery.
   `compatibility` reproduces legacy all-row behavior only when selected
   explicitly; its results and manifests label the estimate selection-biased,
   and user documentation must not present it as held-out performance.
3. **Discovery boundary.** Gold fits methods only on rows whose Silver
   partition is `discovery`; evaluation targets never reach method fitting,
   candidate trials, or any selection decision. Generated code may execute on
   the complete feature frame only after the method is fit and never receives
   evaluation targets. Row-local candidates (a row's value independent of
   companion rows) must pass deterministic row-subset/permutation metamorphic
   checks before acceptance; `fitted` transformations (separate fit/transform
   with serializable learned state) are rejected in holdout mode until that
   contract is implemented. The compatibility profile may replay legacy
   whole-frame scripts, with provenance identifying the transductive risk.
4. **Fold-local preprocessing.** Numeric imputation and categorical encoding
   are fit on each fold's training rows only, with a stable unknown-category
   policy, via a scikit-learn `Pipeline`/`ColumnTransformer` (or an
   equivalently typed small helper — no second evaluation framework). The
   fitted preprocessing specification is retained in evaluation provenance.
5. **Greedy selection on discovery folds.** Deterministic greedy forward
   selection: evaluate each remaining candidate appended to the already
   selected set; rank by mean directional gain with a stable feature-name
   tie-break; require `minimum_practical_gain`; require a positive directional
   lower confidence bound when `require_positive_lower_bound` is set; stop at
   `max_selected_features` or when no candidate qualifies.
   `selection_partition="discovery"` is required by the holdout profile;
   unsupported partition/profile combinations fail during request validation.
6. **Final evaluation.** After selection freezes, exactly two arms are
   evaluated on evaluation folds: baseline and the combined selected feature
   set. When no candidate qualifies, the enhanced arm mirrors baseline and the
   gain is zero. The public result comes only from this accepted enhanced arm —
   never from a rejected best trial.
7. **Uncertainty.** Compute paired **directional** fold deltas and a
   Student-t critical value from the configured confidence level and
   `pair_count - 1` degrees of freedom. If SciPy is imported directly, it
   becomes a direct dependency.
8. **Evidence schema v2.** Platinum persists discovery candidate/step fold
   metrics, final evaluation fold metrics, selected-set membership, policy
   decisions, preprocessing identity, and the directional interval under
   schema v2. Old v1 packages remain readable but cannot satisfy v2 reuse
   fingerprints; semantic verification reconstructs decisions and aggregates
   from persisted fold evidence alone.

## Consequences

- Default reported scores become independent of discovery and selection, with
  fold-local preprocessing; every leakage path above gains a focused test.
- Discovery loses the evaluation rows (default 25%); small datasets fail fast
  under holdout instead of degrading, pushing them to explicit compatibility.
- Platinum schema v2 invalidates cross-version reuse through fingerprints
  (v1 packages stay readable, nothing is deleted); compatibility results are
  permanently labeled selection-biased.
- Revisit the `fitted`-transformation scope contract when a method needs
  fitted features under holdout; plan 23 PRs 2–4 implement this record
  literally.
- **Rollback.** Each plan 23 PR is independently revertible; protocol and
  Platinum schema changes invalidate reuse through fingerprints rather than
  deleting old packages, so reverting restores v1 reuse without data loss,
  and compatibility evaluation remains available and truthfully labeled
  throughout.
