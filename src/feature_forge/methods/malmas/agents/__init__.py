"""Agent system for feature_forge."""

from feature_forge.methods.malmas.agents.base import Agent, AgentRegistry, BaseFeatureAgent
from feature_forge.utils import _create_lazy_getattr

__all__ = [
    "Agent",
    "AgentRegistry",
    "BaseFeatureAgent",
]

__getattr__ = _create_lazy_getattr(
    {
        "RouterAgent": "feature_forge.methods.malmas.agents.router",
    },
    __name__,
)
