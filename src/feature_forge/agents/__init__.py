"""Agent system for feature_forge."""

from feature_forge.agents.base import Agent, AgentRegistry, BaseFeatureAgent
from feature_forge.utils import _create_lazy_getattr

__all__ = [
    "Agent",
    "AgentRegistry",
    "BaseFeatureAgent",
]

__getattr__ = _create_lazy_getattr(
    {
        "AggregationConstructAgent": "feature_forge.agents.aggregation",
        "CrossCompositionalAgent": "feature_forge.agents.cross_compositional",
        "LocalPatternAgent": "feature_forge.agents.local_pattern",
        "LocalTransformAgent": "feature_forge.agents.local_transform",
        "RouterAgent": "feature_forge.agents.router",
        "TemporalFeatureAgent": "feature_forge.agents.temporal",
        "UnaryFeatureAgent": "feature_forge.agents.unary",
    },
    __name__,
)
