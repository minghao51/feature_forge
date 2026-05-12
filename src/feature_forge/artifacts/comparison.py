"""Method comparison utility.

Runs all feature engineering methods on the same data and returns
unified artifact dictionaries per method.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter


def compare_methods(
    methods: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None = None,
    tracker: Any | None = None,
    artifact_config: ArtifactConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all methods on the same data and collect unified artifacts.

    Args:
        methods: Dict mapping method names to Baseline instances.
        X_train: Training features.
        y_train: Training target.
        X_test: Optional test features.
        tracker: Optional ExperimentTracker for auto-logging.

    Returns:
        Dict mapping method name to its artifact dictionary.
    """
    artifact_config = artifact_config or ArtifactConfig()
    results: dict[str, dict[str, Any]] = {}

    for name, method in methods.items():
        if hasattr(method, "artifact_config"):
            method.artifact_config = artifact_config
        try:
            method.fit(X_train, y_train)
            artifacts = method.get_artifacts()

            if X_test is not None:
                try:
                    transformed = method.transform(X_test)
                    artifacts["transformed_test"] = transformed
                except Exception as exc:
                    artifacts["transformed_test_error"] = str(exc)

            results[name] = artifacts

        except Exception as exc:
            results[name] = {"error": str(exc)}
            continue

        if tracker is not None and isinstance(method, ArtifactExporter):
            try:
                method.log_artifacts(tracker, prefix=f"{name}_")
            except Exception as exc:
                results[name]["tracker_error"] = str(exc)

    return results
