"""Temporal feature agent."""

from __future__ import annotations

from feature_forge.agents.base import BaseFeatureAgent


class TemporalFeatureAgent(BaseFeatureAgent):
    """Generates time-based features."""

    prompt_filename = "temporal.txt"
    agent_name = "temporal"
