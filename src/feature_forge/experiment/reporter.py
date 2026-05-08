"""Reporter for generating comparison tables and summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd


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

    def get_best(self, metric: str = "score", group_by: str = "dataset") -> pd.DataFrame:
        """Get best result per group."""
        if group_by not in self.df.columns or metric not in self.df.columns:
            return pd.DataFrame()
        idx = self.df.groupby(group_by)[metric].idxmax()
        return self.df.loc[idx]

    def summary_stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        return {
            "total_runs": len(self.df),
            "successful_runs": len(self.df[self.df.get("error", pd.Series()).isna()]),
            "failed_runs": len(self.df[self.df.get("error", pd.Series()).notna()]),
            "mean_metrics": {
                col: float(self.df[col].mean()) for col in numeric_cols if col != "error"
            },
        }
