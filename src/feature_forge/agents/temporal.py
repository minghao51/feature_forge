"""Temporal feature agent."""

from __future__ import annotations

from feature_forge.agents.base import BaseFeatureAgent


class TemporalFeatureAgent(BaseFeatureAgent):
    """Generates time-based features."""

    prompt_key = "temporal"
    agent_name = "temporal"
