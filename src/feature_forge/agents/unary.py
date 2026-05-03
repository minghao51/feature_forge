"""Unary feature agent."""

from __future__ import annotations

from feature_forge.agents.base import BaseFeatureAgent


class UnaryFeatureAgent(BaseFeatureAgent):
    """Generates derived features from single columns."""

    prompt_filename = "unary.txt"
    agent_name = "unary"
