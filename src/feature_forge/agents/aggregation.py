"""Aggregation construct feature agent."""

from __future__ import annotations

from feature_forge.agents.base import BaseFeatureAgent


class AggregationConstructAgent(BaseFeatureAgent):
    """Generates aggregation-based features."""

    prompt_filename = "aggregation.txt"
    agent_name = "aggregation"
