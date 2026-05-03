"""OpenFE baseline wrapper with best-effort artifact extraction.

OpenFE is a strong non-LLM baseline that doesn't generate code.
Artifacts include selected operators and (best-effort) candidate
operators and feature importances.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig
from feature_forge.baselines.base import Baseline
from feature_forge.exceptions import EvaluationError


class OpenFEBaseline(Baseline):
    """OpenFE baseline for automated feature engineering.

    Requires: pip install openfe
    """

    def __init__(
        self,
        n_jobs: int = 1,
        metric: str = "auc",
        artifact_config: ArtifactConfig | None = None,
    ) -> None:
        super().__init__("openfe", artifact_config=artifact_config)
        self.n_jobs = n_jobs
        self.metric = metric
        self._ofe = None
        self._features = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> OpenFEBaseline:
        try:
            from openfe import OpenFE, transform
        except ImportError as exc:
            raise EvaluationError("openfe not installed. Run: uv pip install openfe") from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ofe = OpenFE(n_jobs=self.n_jobs)
            self._features = self._ofe.fit(data=X_train, label=y_train, metric=self.metric)
            _, self._train_features = transform(X_train, X_train, self._features, n_jobs=self.n_jobs)

        self._extract_artifacts()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._ofe is None or self._features is None:
            raise EvaluationError("OpenFEBaseline not fitted yet")
        try:
            from openfe import transform
        except ImportError as exc:
            raise EvaluationError("openfe not installed") from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, X_transformed = transform(X, X, self._features, n_jobs=self.n_jobs)
        return X_transformed

    def _extract_artifacts(self) -> None:
        """Extract artifacts from the fitted OpenFE object.

        OpenFE's internal API stores selected operators as a list of Node
        objects. We attempt to extract human-readable names/formulas.
        If unavailable, we store None and warn rather than coercing to str().
        """
        selected_names = self._operator_names(self._features)
        self._artifacts["selected_operators"] = selected_names

        if self._ofe is not None:
            candidate = getattr(self._ofe, "candidate_features_list", None)
            if candidate is None:
                warnings.warn(
                    "OpenFE 'candidate_features_list' not available; "
                    "set to None. OpenFE internal API may have changed.",
                    stacklevel=2,
                )
            self._artifacts["candidate_operators"] = self._operator_names(candidate)

            importances = getattr(self._ofe, "feature_importances_", None)
            if importances is None:
                warnings.warn(
                    "OpenFE 'feature_importances_' not available; "
                    "set to None. OpenFE internal API may have changed.",
                    stacklevel=2,
                )
            self._artifacts["feature_importances"] = (
                self._importance_df(importances) if importances is not None else None
            )

    @staticmethod
    def _operator_names(ops: Any) -> list[str] | None:
        """Try to extract readable names from OpenFE operator objects."""
        if ops is None:
            return None
        names: list[str] = []
        for op in ops:
            # OpenFE Node objects may have .name or be convertible via tree_to_formula
            if hasattr(op, "name"):
                names.append(str(op.name))
            elif hasattr(op, "__repr__"):
                names.append(repr(op))
            else:
                names.append(str(op))
        return names

    @staticmethod
    def _importance_df(importances: Any) -> pd.DataFrame | None:
        """Convert OpenFE importance array into a DataFrame."""
        try:
            import numpy as np
            arr = np.asarray(importances)
            return pd.DataFrame({"rank": range(len(arr)), "importance": arr})
        except Exception:
            return None

    @property
    def generated_scripts(self) -> list[str]:
        return []

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        meta: list[dict[str, Any]] = []
        selected = self._artifacts.get("selected_operators")
        if selected:
            meta.append({"name": "selected_operators", "value": selected})
        return meta
