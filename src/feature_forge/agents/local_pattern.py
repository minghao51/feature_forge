"""Local pattern feature agent."""

from __future__ import annotations

from feature_forge.agents.base import BaseFeatureAgent


class LocalPatternAgent(BaseFeatureAgent):
    """Generates features based on distributional patterns."""

    prompt_key = "local_pattern"
    agent_name = "local_pattern"
