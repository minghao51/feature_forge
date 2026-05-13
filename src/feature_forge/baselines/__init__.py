"""Baseline methods for feature_forge."""

from feature_forge.baselines.base import Baseline, BaselineProtocol, BaselineRegistry
from feature_forge.baselines.caafe import CAAFEBaseline
from feature_forge.baselines.llmfe import LLMFEBaseline
from feature_forge.baselines.malmus import MalmusBaseline
from feature_forge.baselines.openfe import OpenFEBaseline

__all__ = [
    "Baseline",
    "BaselineProtocol",
    "BaselineRegistry",
    "CAAFEBaseline",
    "LLMFEBaseline",
    "MalmusBaseline",
    "OpenFEBaseline",
]
