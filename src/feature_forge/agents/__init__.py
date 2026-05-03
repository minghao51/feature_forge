"""Agent system for feature_forge."""

from feature_forge.agents.aggregation import AggregationConstructAgent
from feature_forge.agents.base import Agent, AgentRegistry, BaseFeatureAgent
from feature_forge.agents.cross_compositional import CrossCompositionalAgent
from feature_forge.agents.local_pattern import LocalPatternAgent
from feature_forge.agents.local_transform import LocalTransformAgent
from feature_forge.agents.router import RouterAgent
from feature_forge.agents.temporal import TemporalFeatureAgent
from feature_forge.agents.unary import UnaryFeatureAgent

__all__ = [
    "Agent",
    "AgentRegistry",
    "AggregationConstructAgent",
    "BaseFeatureAgent",
    "CrossCompositionalAgent",
    "LocalPatternAgent",
    "LocalTransformAgent",
    "RouterAgent",
    "TemporalFeatureAgent",
    "UnaryFeatureAgent",
]
