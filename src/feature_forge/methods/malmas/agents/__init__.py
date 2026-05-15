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
        "AggregationConstructAgent": "feature_forge.methods.malmas.agents.aggregation",
        "CrossCompositionalAgent": "feature_forge.methods.malmas.agents.cross_compositional",
        "LocalPatternAgent": "feature_forge.methods.malmas.agents.local_pattern",
        "LocalTransformAgent": "feature_forge.methods.malmas.agents.local_transform",
        "RouterAgent": "feature_forge.methods.malmas.agents.router",
        "TemporalFeatureAgent": "feature_forge.methods.malmas.agents.temporal",
        "UnaryFeatureAgent": "feature_forge.methods.malmas.agents.unary",
    },
    __name__,
)
