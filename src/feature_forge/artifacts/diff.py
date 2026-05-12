"""Cross-method artifact comparison and diffing.

Provides ``ArtifactDiff`` for computing structured differences between
artifact bundles from different feature engineering methods.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.artifacts.schema import ArtifactBundle, FeatureMetadata


class ArtifactDiff:
    """Structured diff between two or more artifact bundles.

    Computes overlap, unique features, and gain differences across methods.
    """

    def __init__(self, bundles: dict[str, ArtifactBundle]) -> None:
        """Initialize with a mapping from method name to bundle.

        Args:
            bundles: Dict mapping method name to ArtifactBundle.
        """
        self.bundles = bundles
        self._feature_index: dict[str, dict[str, FeatureMetadata]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Index features by name across all methods."""
        for method, bundle in self.bundles.items():
            for fm in bundle.feature_metadata:
                name = fm.name
                if name not in self._feature_index:
                    self._feature_index[name] = {}
                self._feature_index[name][method] = fm

    @property
    def all_methods(self) -> list[str]:
        """Return sorted list of method names."""
        return sorted(self.bundles.keys())

    @property
    def all_features(self) -> list[str]:
        """Return sorted list of all unique feature names."""
        return sorted(self._feature_index.keys())

    def overlap_matrix(self) -> pd.DataFrame:
        """Return a binary matrix of feature presence per method.

        Rows = features, columns = methods, values = 1 if present.
        """
        records: list[dict[str, Any]] = []
        for feat in self.all_features:
            row: dict[str, Any] = {"feature": feat}
            for method in self.all_methods:
                row[method] = 1 if method in self._feature_index[feat] else 0
            records.append(row)
        if not records:
            return pd.DataFrame(columns=["feature", *self.all_methods]).set_index("feature")
        return pd.DataFrame(records).set_index("feature")

    def shared_features(self) -> list[str]:
        """Return feature names present in *all* methods."""
        return [
            feat
            for feat in self.all_features
            if len(self._feature_index[feat]) == len(self.bundles)
        ]

    def unique_features(self, method: str) -> list[str]:
        """Return feature names unique to a single method."""
        return [
            feat
            for feat in self.all_features
            if method in self._feature_index[feat] and len(self._feature_index[feat]) == 1
        ]

    def gain_comparison(self) -> pd.DataFrame:
        """Return a DataFrame comparing gains per feature per method.

        Rows = features, columns = methods, values = gain (or NaN).
        """
        records: list[dict[str, Any]] = []
        for feat in self.all_features:
            row: dict[str, Any] = {"feature": feat}
            for method in self.all_methods:
                meta = self._feature_index[feat].get(method)
                row[method] = meta.gain if meta else None
            records.append(row)
        if not records:
            return pd.DataFrame(columns=["feature", *self.all_methods]).set_index("feature")
        return pd.DataFrame(records).set_index("feature")

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary dict."""
        total_features = len(self.all_features)
        shared = self.shared_features()
        summary: dict[str, Any] = {
            "total_unique_features": total_features,
            "shared_across_all": len(shared),
            "shared_feature_names": shared,
            "per_method": {},
        }
        for method in self.all_methods:
            method_features = [
                feat for feat in self.all_features if method in self._feature_index[feat]
            ]
            unique = self.unique_features(method)
            gains = [
                g
                for f in method_features
                for g in [self._feature_index[f][method].gain]
                if g is not None
            ]
            summary["per_method"][method] = {
                "total_features": len(method_features),
                "unique_features": len(unique),
                "unique_feature_names": unique,
                "mean_gain": sum(gains) / len(gains) if gains else None,
                "max_gain": max(gains) if gains else None,
            }
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Export the full diff as a flat DataFrame."""
        records: list[dict[str, Any]] = []
        for feat in self.all_features:
            methods_present = list(self._feature_index[feat].keys())
            gains = {m: self._feature_index[feat][m].gain for m in methods_present}
            record: dict[str, Any] = {
                "feature": feat,
                "methods": ", ".join(methods_present),
                "num_methods": len(methods_present),
                "shared_all": len(methods_present) == len(self.bundles),
            }
            for method in self.all_methods:
                record[f"{method}_gain"] = gains.get(method)
            records.append(record)
        return pd.DataFrame(records)
