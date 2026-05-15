"""MALMAS method adapter implementing MethodProtocol."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.api import FeatureForge
from feature_forge.config import Settings, get_settings
from feature_forge.methods.base import BaseMethod


class MALMASMethod(BaseMethod):
    """MALMAS method adapter implementing BaseMethod.

    Wraps FeatureForge (the sklearn-compatible API) to make it
    interchangeable with other methods via the unified protocol.
    """

    def __init__(self, config: Settings | None = None, **kwargs: Any) -> None:
        super().__init__("malmas")
        self._config = config or get_settings()
        self._kwargs = kwargs
        self._forge: FeatureForge | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> MALMASMethod:
        self._forge = FeatureForge(config=self._config, **self._kwargs)
        self._forge.fit(X_train, y_train, **kwargs)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._forge is None:
            raise RuntimeError("MALMASMethod not fitted yet")
        return self._forge.transform(X)

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any
    ) -> pd.DataFrame:
        self.fit(X_train, y_train, **kwargs)
        return self.transform(X_train)

    @property
    def generated_scripts(self) -> list[str]:
        return self._forge.generated_scripts if self._forge else []

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return self._forge.feature_metadata if self._forge else []

    def get_artifacts(self) -> dict[str, Any]:
        return self._forge.get_artifacts() if self._forge else {}
