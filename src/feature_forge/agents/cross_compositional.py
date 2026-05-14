"""Cross-compositional feature agent."""

from __future__ import annotations

from feature_forge.agents.base import BaseFeatureAgent


class CrossCompositionalAgent(BaseFeatureAgent):
    """Generates cross-composed features between 2+ columns."""

    prompt_key = "cross_compositional"
    agent_name = "cross_compositional"
