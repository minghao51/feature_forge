"""Shared selection-policy logic for the Platinum layer.

Both the Platinum executor (``_evaluate``) and the Platinum loader
(``load_platinum_package``) must derive the *same* selected feature set and the
*same* per-feature reason codes from candidate evidence. Centralizing the policy
here means a policy edit can never silently desync the loader's verification
from the executor's decisions.
"""

from __future__ import annotations

from feature_forge.contracts import AggregateMetric, SelectionPolicy, UncertaintySummary

# Map of feature name to its paired (aggregate, uncertainty) evidence.
CandidateSummaries = dict[str, tuple[AggregateMetric, UncertaintySummary]]

# Map of feature name to (reason_code, human-readable reason).
ReasonCodes = dict[str, tuple[str, str]]

_MAX_SELECTED_REASON = (
    "max_selected_features",
    "Excluded by max_selected_features after evidence ranking",
)


def _is_selected(
    aggregate: AggregateMetric,
    uncertainty: UncertaintySummary,
    policy: SelectionPolicy,
) -> tuple[bool, str]:
    """Return (selected, reason_code) for one candidate under ``policy``."""
    if policy.profile == "compatibility":
        selected = aggregate.legacy_gain > 0
        return selected, "legacy_gain_positive" if selected else "legacy_gain_not_positive"
    practical = aggregate.directional_gain >= policy.minimum_practical_gain
    lower_ok = not policy.require_positive_lower_bound or uncertainty.lower_bound > 0
    selected = practical and lower_ok
    if selected:
        code = "recommended_evidence_passed"
    elif not practical:
        code = "practical_gain_below_threshold"
    else:
        code = "lower_bound_not_positive"
    return selected, code


def apply_selection_policy(
    candidate_summaries: CandidateSummaries,
    policy: SelectionPolicy,
) -> tuple[set[str], ReasonCodes]:
    """Apply ``policy`` to per-candidate evidence.

    Returns ``(final_selected_names, reason_codes)`` where:

    * ``final_selected_names`` is the set of feature names kept after the policy
      predicate *and* the optional ``max_selected_features`` cap (ranked by
      descending ``directional_gain``).
    * ``reason_codes[name] = (code, reason)`` reconstructs the persisted
      ``PlatinumSelectionDecision`` metadata for every candidate, including the
      ``max_selected_features`` override for candidates demoted by the cap.

    Both the executor and the loader build ``candidate_summaries`` in the same
    shape (discovery-partition-paired evidence) before calling this, so the two
    paths cannot diverge on policy semantics.
    """
    policy_selected: set[str] = set()
    codes: dict[str, tuple[str, str]] = {}
    for name, (aggregate, uncertainty) in candidate_summaries.items():
        selected, code = _is_selected(aggregate, uncertainty, policy)
        if selected:
            policy_selected.add(name)
        codes[name] = (
            code,
            f"Selection profile {policy.profile}: {code}",
        )

    final_selected = set(policy_selected)
    cap = policy.max_selected_features
    if cap is not None and len(final_selected) > cap:
        ranked = sorted(
            final_selected,
            key=lambda n: candidate_summaries[n][0].directional_gain,
            reverse=True,
        )
        final_selected = set(ranked[:cap])
        for name in policy_selected - final_selected:
            codes[name] = _MAX_SELECTED_REASON
    return final_selected, codes
