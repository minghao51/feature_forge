"""Local transform feature agent."""

from __future__ import annotations

from feature_forge.agents.base import BaseFeatureAgent


class LocalTransformAgent(BaseFeatureAgent):
    """Generates local transformation features."""

    prompt_filename = "local_transform.txt"
    agent_name = "local_transform"
