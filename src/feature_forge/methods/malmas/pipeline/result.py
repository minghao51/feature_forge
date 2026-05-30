from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from feature_forge.types import FeatureSpec


@dataclass(frozen=True)
class PipelineResult:
    features_train: pd.DataFrame
    features_test: pd.DataFrame
    all_features_train: pd.DataFrame
    all_features_test: pd.DataFrame
    selected_features_train: pd.DataFrame
    selected_features_test: pd.DataFrame
    top_features_train: pd.DataFrame
    top_features_test: pd.DataFrame
    agent_gains: dict[str, pd.DataFrame]
    specs: list[FeatureSpec]
    baseline_score: float
    gains: dict[str, float]
    generated_code: str
