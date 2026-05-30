from __future__ import annotations

import pandas as pd

from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


def prefilter_candidate_columns(
    features_df: pd.DataFrame,
    max_candidates: int,
) -> list[str]:
    if features_df.empty:
        return []

    candidates = []
    for col in features_df.columns:
        series = features_df[col]
        if series.nunique(dropna=False) <= 1:
            continue
        candidates.append(col)

    if len(candidates) <= max_candidates:
        return candidates

    variances: list[tuple[str, float]] = []
    for col in candidates:
        series = features_df[col]
        if pd.api.types.is_numeric_dtype(series):
            variances.append((col, float(series.var(ddof=0))))
        else:
            ratio = series.nunique(dropna=True) / len(series)
            variances.append((col, ratio))
    variances.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in variances[:max_candidates]]
