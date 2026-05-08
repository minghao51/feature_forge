"""Offline HTML dashboard for artifact comparison.

Generates a self-contained HTML report comparing artifact bundles
from multiple feature engineering methods.
"""

from __future__ import annotations

import json
from pathlib import Path

from feature_forge.artifacts.diff import ArtifactDiff
from feature_forge.artifacts.schema import ArtifactBundle


class ArtifactDashboard:
    """Generate an offline HTML report from artifact bundles.

    Usage:
        >>> bundles = {"llmfe": llmfe_bundle, "caafe": caafe_bundle}
        >>> dash = ArtifactDashboard(bundles)
        >>> dash.save("report.html")
    """

    def __init__(self, bundles: dict[str, ArtifactBundle]) -> None:
        self.bundles = bundles
        self.diff = ArtifactDiff(bundles)

    def _html_head(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Feature Forge Artifact Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem; background: #f8f9fa; }
  h1, h2, h3 { color: #1a1a2e; }
  .card { background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  th, td { text-align: left; padding: 0.5rem; border-bottom: 1px solid #e9ecef; }
  th { background: #e9ecef; font-weight: 600; }
  .badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.85rem; font-weight: 500; }
  .badge-success { background: #d4edda; color: #155724; }
  .badge-info { background: #d1ecf1; color: #0c5460; }
  .badge-warning { background: #fff3cd; color: #856404; }
  pre { background: #f8f9fa; padding: 1rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
  .metric { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
  .metric-label { font-size: 0.9rem; color: #6c757d; }
</style>
</head>
<body>
<h1>Feature Forge Artifact Report</h1>
"""

    def _summary_section(self) -> str:
        summary = self.diff.summary()
        html = '<div class="card"><h2>Summary</h2><div class="grid">'
        html += f"""
          <div><div class="metric">{summary["total_unique_features"]}</div>
               <div class="metric-label">Total Unique Features</div></div>
          <div><div class="metric">{summary["shared_across_all"]}</div>
               <div class="metric-label">Shared Across All Methods</div></div>
        """
        for method, data in summary.get("per_method", {}).items():
            html += f"""
              <div><div class="metric">{data["total_features"]}</div>
                   <div class="metric-label">{method} Features</div></div>
            """
        html += "</div></div>"
        return html

    def _overlap_section(self) -> str:
        df = self.diff.overlap_matrix()
        if df.empty:
            return '<div class="card"><h2>Feature Overlap</h2><p>No features to compare.</p></div>'
        html = '<div class="card"><h2>Feature Overlap Matrix</h2><table><thead><tr><th>Feature</th>'
        for method in self.diff.all_methods:
            html += f"<th>{method}</th>"
        html += "</tr></thead><tbody>"
        for idx, row in df.iterrows():
            html += f"<tr><td><strong>{idx}</strong></td>"
            for method in self.diff.all_methods:
                val = row[method]
                badge = (
                    '<span class="badge badge-success">&#10003;</span>'
                    if val
                    else '<span class="badge badge-warning">-</span>'
                )
                html += f"<td>{badge}</td>"
            html += "</tr>"
        html += "</tbody></table></div>"
        return html

    def _gain_section(self) -> str:
        df = self.diff.gain_comparison()
        if df.empty:
            return '<div class="card"><h2>Gain Comparison</h2><p>No gain data available.</p></div>'
        html = '<div class="card"><h2>Gain Comparison</h2><table><thead><tr><th>Feature</th>'
        for method in self.diff.all_methods:
            html += f"<th>{method} Gain</th>"
        html += "</tr></thead><tbody>"
        for idx, row in df.iterrows():
            html += f"<tr><td><strong>{idx}</strong></td>"
            for method in self.diff.all_methods:
                val = row[method]
                cell = f"{val:.4f}" if val is not None else "-"
                html += f"<td>{cell}</td>"
            html += "</tr>"
        html += "</tbody></table></div>"
        return html

    def _method_detail_section(self, method: str, bundle: ArtifactBundle) -> str:
        html = f'<div class="card"><h2>{method}</h2>'
        html += f'<p><span class="badge badge-info">{len(bundle.generated_scripts)} scripts</span> '
        html += f'<span class="badge badge-info">{len(bundle.feature_metadata)} features</span></p>'

        # Feature metadata table
        if bundle.feature_metadata:
            html += "<h3>Features</h3><table><thead><tr><th>Name</th><th>Gain</th><th>Round</th><th>Agent</th></tr></thead><tbody>"
            for fm in bundle.feature_metadata:
                gain = f"{fm.gain:.4f}" if fm.gain is not None else "-"
                rnd = fm.round if fm.round is not None else "-"
                agent = fm.agent or "-"
                html += f"<tr><td>{fm.name}</td><td>{gain}</td><td>{rnd}</td><td>{agent}</td></tr>"
            html += "</tbody></table>"

        # Generated code snippets
        if bundle.generated_scripts:
            html += "<h3>Generated Code</h3>"
            for i, script in enumerate(bundle.generated_scripts):
                html += f"<details><summary>Script {i + 1}</summary><pre><code>{script}</code></pre></details>"

        html += "</div>"
        return html

    def _provenance_section(self) -> str:
        html = '<div class="card"><h2>Provenance</h2>'
        for method, bundle in self.bundles.items():
            if bundle.provenance_records:
                html += f"<h3>{method}</h3><table><thead><tr><th>Feature</th><th>Agent</th><th>Round</th><th>Gain</th></tr></thead><tbody>"
                for pr in bundle.provenance_records:
                    gain = f"{pr.cv_gain:.4f}" if pr.cv_gain is not None else "-"
                    agent = pr.source_agent or "-"
                    rnd = pr.round_index if pr.round_index is not None else "-"
                    html += f"<tr><td>{pr.feature_name}</td><td>{agent}</td><td>{rnd}</td><td>{gain}</td></tr>"
                html += "</tbody></table>"
        html += "</div>"
        return html

    def to_html(self) -> str:
        """Generate the full HTML report as a string."""
        html = self._html_head()
        html += self._summary_section()
        html += self._overlap_section()
        html += self._gain_section()
        for method, bundle in self.bundles.items():
            html += self._method_detail_section(method, bundle)
        html += self._provenance_section()
        html += "</body></html>"
        return html

    def save(self, path: str | Path) -> None:
        """Save the HTML report to disk."""
        Path(path).write_text(self.to_html(), encoding="utf-8")

    def to_json(self) -> str:
        """Export the diff summary as JSON."""
        return json.dumps(self.diff.summary(), indent=2)
