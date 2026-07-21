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
        # Select numeric columns for aggregation
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        group_cols = [c for c in ["dataset", "method", "model"] if c in self.df.columns]
        try:
            if group_cols and numeric_cols:
                summary = self.df.groupby(group_cols)[numeric_cols].mean().reset_index()
                return summary.to_markdown(index=False)
            return self.df.to_markdown(index=False)
        except ImportError:
            # tabulate not installed, return simple string representation
            if group_cols and numeric_cols:
                summary = self.df.groupby(group_cols)[numeric_cols].mean().reset_index()
                return str(summary)
            return str(self.df)

    def to_html(self) -> str:
        """Generate HTML comparison table."""
        if self.df.empty:
            return "<p>No results to report.</p>"
        return self.df.to_html(index=False)

    def get_best(
        self,
        metric: str = "score",
        group_by: str = "dataset",
        *,
        direction: MetricDirection | str | None = None,
    ) -> pd.DataFrame:
        """Get best result per group."""
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

    def summary_stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        errors = (
            self.df["error"]
            if "error" in self.df.columns
            else pd.Series([None] * len(self.df), index=self.df.index, dtype="object")
        )
        aggregate_numeric = [
            col
            for col in numeric_cols
            if col
            not in {
                "seed",
                "num_features_generated",
                "num_candidate_features",
                "num_executed_features",
                "num_accepted_features",
                "num_accepted_output_columns",
            }
        ]
        return {
            "total_runs": len(self.df),
            "successful_runs": int(errors.isna().sum()),
            "failed_runs": int(errors.notna().sum()),
            "mean_metrics": {col: float(self.df[col].mean()) for col in aggregate_numeric},
        }
