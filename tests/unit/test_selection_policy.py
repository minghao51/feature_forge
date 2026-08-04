"""Focused tests for the shared Platinum selection-policy logic.

Covers the three branches (compatibility, recommended, max_selected_features
cap) that the executor and loader previously duplicated.
"""

from __future__ import annotations

import pytest

from feature_forge.contracts import (
    AggregateMetric,
    SelectionPolicy,
    UncertaintySummary,
)
from feature_forge.dataflows._selection import apply_selection_policy
from feature_forge.evaluation.metrics import MetricDirection


def _agg(legacy_gain: float, directional_gain: float) -> AggregateMetric:
    return AggregateMetric(
        metric="auc",
        metric_direction=MetricDirection.MAXIMIZE,
        aggregation="unweighted_mean_of_fold_metrics",
        baseline_score=0.5,
        enhanced_score=0.5 + directional_gain,
        legacy_gain=legacy_gain,
        directional_gain=directional_gain,
        successful_folds=5,
    )


def _unc(lower_bound: float) -> UncertaintySummary:
    return UncertaintySummary(
        method="paired_t",
        confidence_level=0.95,
        pair_count=5,
        mean_directional_gain=0.01,
        standard_deviation=0.0,
        standard_error=0.0,
        lower_bound=lower_bound,
        upper_bound=0.1,
    )


class TestApplySelectionPolicyCompatibility:
    def test_positive_legacy_gain_is_selected(self) -> None:
        summaries = {"f1": (_agg(0.05, 0.05), _unc(0.0))}
        selected, codes = apply_selection_policy(summaries, SelectionPolicy())
        assert selected == {"f1"}
        assert codes["f1"] == (
            "legacy_gain_positive",
            "Selection profile compatibility: legacy_gain_positive",
        )

    def test_zero_legacy_gain_is_not_selected(self) -> None:
        # compatibility uses strict ``> 0``; zero gain is rejected.
        summaries = {"f1": (_agg(0.0, 0.0), _unc(0.0))}
        selected, codes = apply_selection_policy(summaries, SelectionPolicy())
        assert selected == set()
        assert codes["f1"][0] == "legacy_gain_not_positive"


class TestApplySelectionPolicyRecommended:
    def _policy(self, **kwargs: object) -> SelectionPolicy:
        base = {"profile": "recommended", "minimum_practical_gain": 0.01}
        base.update(kwargs)
        return SelectionPolicy(**base)  # type: ignore[arg-type]

    def test_practical_gain_passes(self) -> None:
        summaries = {"f1": (_agg(0.05, 0.02), _unc(-0.1))}
        selected, codes = apply_selection_policy(summaries, self._policy())
        assert selected == {"f1"}
        assert codes["f1"][0] == "recommended_evidence_passed"

    def test_below_threshold_rejected(self) -> None:
        summaries = {"f1": (_agg(0.0, 0.005), _unc(0.5))}
        selected, codes = apply_selection_policy(summaries, self._policy())
        assert selected == set()
        assert codes["f1"][0] == "practical_gain_below_threshold"

    def test_positive_lower_bound_required_when_gate_on(self) -> None:
        # practical gain passes but lower_bound <= 0 with the gate on → rejected.
        summaries = {"f1": (_agg(0.05, 0.02), _unc(0.0))}
        selected, codes = apply_selection_policy(
            summaries, self._policy(require_positive_lower_bound=True)
        )
        assert selected == set()
        assert codes["f1"][0] == "lower_bound_not_positive"

    def test_negative_lower_bound_allowed_when_gate_off(self) -> None:
        summaries = {"f1": (_agg(0.05, 0.02), _unc(-0.5))}
        selected, _ = apply_selection_policy(summaries, self._policy())
        assert selected == {"f1"}


class TestApplySelectionPolicyCap:
    def test_cap_ranks_by_directional_gain_desc(self) -> None:
        summaries = {
            "low": (_agg(0.1, 0.01), _unc(0.0)),
            "high": (_agg(0.1, 0.09), _unc(0.0)),
            "mid": (_agg(0.1, 0.05), _unc(0.0)),
        }
        selected, codes = apply_selection_policy(
            summaries,
            SelectionPolicy(
                profile="recommended", minimum_practical_gain=0.001, max_selected_features=2
            ),
        )
        # Top-2 by directional_gain: high (0.09) and mid (0.05); low demoted.
        assert selected == {"high", "mid"}
        assert codes["low"] == (
            "max_selected_features",
            "Excluded by max_selected_features after evidence ranking",
        )
        assert codes["high"][0] == "recommended_evidence_passed"

    def test_cap_below_count_is_noop(self) -> None:
        summaries = {"f1": (_agg(0.1, 0.02), _unc(0.0))}
        selected, _ = apply_selection_policy(
            summaries,
            SelectionPolicy(
                profile="recommended", minimum_practical_gain=0.001, max_selected_features=5
            ),
        )
        assert selected == {"f1"}


class TestApplySelectionPolicyDeterminism:
    @pytest.mark.parametrize("profile", ["compatibility", "recommended"])
    def test_same_inputs_yield_same_outputs(self, profile: str) -> None:
        summaries = {"f1": (_agg(0.05, 0.02), _unc(0.0))}
        policy = SelectionPolicy(
            profile=profile,  # type: ignore[arg-type]
            minimum_practical_gain=0.001,
        )
        first = apply_selection_policy(summaries, policy)
        second = apply_selection_policy(summaries, policy)
        assert first == second
