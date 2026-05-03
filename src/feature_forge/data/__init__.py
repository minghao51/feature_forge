"""Data layer for feature_forge."""

from feature_forge.data.ingestion import KaggleFetcher
from feature_forge.data.registry import DatasetRegistry

__all__ = [
    "DatasetRegistry",
    "KaggleFetcher",
]
