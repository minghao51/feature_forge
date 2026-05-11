"""Agent system for feature_forge."""

from feature_forge.agents.base import Agent, AgentRegistry, BaseFeatureAgent
from feature_forge.agents.router import RouterAgent

__all__ = [
    "Agent",
    "AgentRegistry",
    "BaseFeatureAgent",
    "RouterAgent",
]


def __getattr__(name: str) -> type:
    _lazy = {
        "AggregationConstructAgent": "feature_forge.agents.aggregation",
        "CrossCompositionalAgent": "feature_forge.agents.cross_compositional",
        "LocalPatternAgent": "feature_forge.agents.local_pattern",
        "LocalTransformAgent": "feature_forge.agents.local_transform",
        "TemporalFeatureAgent": "feature_forge.agents.temporal",
        "UnaryFeatureAgent": "feature_forge.agents.unary",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
