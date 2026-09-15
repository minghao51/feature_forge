"""MALMAS method adapter implementing MethodProtocol."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.api import FeatureForge, validate_mode
from feature_forge.config import Settings, get_settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.kit import EvaluationKit
from feature_forge.llm.base import LLMClient
from feature_forge.methods.base import BaseMethod


class MALMASMethod(BaseMethod):
    """MALMAS method adapter implementing BaseMethod.

    Wraps FeatureForge (the sklearn-compatible API) to make it
    interchangeable with other methods via the unified protocol.
    """

    def __init__(
        self,
        config: Settings | None = None,
        mode: str = "full",
        llm_client: LLMClient | None = None,
        evaluator: CVEvaluator | None = None,
        warm_start: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__("malmas")
        self._config = config or get_settings()
        self._mode = validate_mode(mode)
        self._llm_client = llm_client
        self._evaluator = evaluator
        self._warm_start = warm_start
        self._kwargs = kwargs
        self._forge: FeatureForge | None = None

    def _new_forge(self) -> FeatureForge:
        evaluation_kit = None
        if self._evaluator is not None:
            evaluation_kit = EvaluationKit.from_settings(self._config)
            evaluation_kit.evaluator = self._evaluator
            evaluation_kit.model_factory = self._evaluator.model_factory
        return FeatureForge(
            config=self._config,
            mode=self._mode,
            llm_client=self._llm_client,
            warm_start=self._warm_start,
            evaluation_kit=evaluation_kit,
            **self._kwargs,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> MALMASMethod:
        self._forge = self._new_forge()
        self._forge.fit(X_train, y_train)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._forge is None:
            raise RuntimeError("MALMASMethod not fitted yet")
        return self._forge.transform(X)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self._forge = self._new_forge()
        return self._forge.fit_transform(X_train, y_train)

    @property
    def generated_scripts(self) -> list[str]:
        return self._forge.generated_scripts if self._forge else []

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return self._forge.feature_metadata if self._forge else []

    def get_artifacts(self) -> dict[str, Any]:
        return self._forge.get_artifacts() if self._forge else {}
