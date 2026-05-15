"""Local transform feature agent."""

from __future__ import annotations

from feature_forge.methods.malmas.agents.base import BaseFeatureAgent


class LocalTransformAgent(BaseFeatureAgent):
    """Generates local transformation features."""

    prompt_key = "local_transform"
    agent_name = "local_transform"
