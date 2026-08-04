"""Reporter for generating comparison tables and summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.evaluation.metrics import MetricDirection, MetricRegistry


class Reporter:
    """Generate markdown/HTML reports from experiment results.

    Usage:
        reporter = Reporter(results)
        md = reporter.to_markdown()
    """

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.df = pd.DataFrame(results)

    def to_markdown(self) -> str:
        """Generate markdown comparison table."""
        if self.df.empty:
            return "No results to report."
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        group_cols = [c for c in ["dataset", "method", "model"] if c in self.df.columns]
        if group_cols and numeric_cols:
            summary = self.df.groupby(group_cols)[numeric_cols].mean().reset_index()
        else:
            summary = self.df
        try:
            return summary.to_markdown(index=False)
        except ImportError:
            # tabulate not installed, return simple string representation
            return str(summary)

    def get_best(
        self,
        metric: str = "cv_score",
        group_by: str = "dataset",
        *,
        direction: MetricDirection | str | None = None,
    ) -> pd.DataFrame:
        """Get best result per group.

        ``metric`` defaults to ``"cv_score"`` to match the column produced by
        ``ExperimentalPlatform.run`` (and ``ExperimentalPlatform.report_best``);
        a direction is auto-resolved from the ``metric`` column when exactly one
        metric name is present, so minimize metrics (rmse/mae) select the lowest
        score rather than silently defaulting to maximize.
        """
        if group_by not in self.df.columns or metric not in self.df.columns:
            return pd.DataFrame()
        resolved = MetricDirection(direction) if direction is not None else None
        if resolved is None and "metric" in self.df.columns:
            metric_names = self.df["metric"].dropna().unique().tolist()
            if len(metric_names) == 1:
                resolved = MetricRegistry.get_direction(str(metric_names[0]))
        idx = (
            self.df.groupby(group_by)[metric].idxmin()
            if resolved is MetricDirection.MINIMIZE
            else self.df.groupby(group_by)[metric].idxmax()
        )
        return self.df.loc[idx]
