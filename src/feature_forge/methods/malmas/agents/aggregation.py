"""Aggregation construct feature agent."""

from __future__ import annotations

from feature_forge.methods.malmas.agents.base import BaseFeatureAgent


class AggregationConstructAgent(BaseFeatureAgent):
    """Generates aggregation-based features."""

    prompt_key = "aggregation"
    agent_name = "aggregation"
