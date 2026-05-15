"""Pipeline layer for feature_forge."""

from feature_forge.methods.malmas.pipeline.ablations import (
    NoMemoryPipeline,
    NoMemoryStaticRouterPipeline,
    NoRouterPipeline,
    SingleAgentPipeline,
)
from feature_forge.methods.malmas.pipeline.core import CorePipeline
from feature_forge.methods.malmas.pipeline.iterative import IterativePipeline

__all__ = [
    "CorePipeline",
    "IterativePipeline",
    "NoMemoryPipeline",
    "NoMemoryStaticRouterPipeline",
    "NoRouterPipeline",
    "SingleAgentPipeline",
]
