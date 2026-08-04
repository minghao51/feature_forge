"""MALMAS method adapter implementing MethodProtocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from feature_forge.api import FeatureForge
from feature_forge.config import Settings, get_settings
from feature_forge.methods.base import BaseMethod

if TYPE_CHECKING:
    from feature_forge.experiment.context import CaseExecutionContext


class MALMASMethod(BaseMethod):
    """MALMAS method adapter implementing BaseMethod.

    Wraps FeatureForge (the sklearn-compatible API) to make it
    interchangeable with other methods via the unified protocol.
    """

    def __init__(
        self,
        config: Settings | None = None,
        mode: str = "full",
        llm_client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("malmas")
        self._config = config or get_settings()
        # `mode` and `llm_client` are promoted to explicit kwargs so they are
        # no longer silently dropped by the historical `inspect.signature`-
        # based construction in `CaseComputation`. They are forwarded to
        # `FeatureForge` at fit time via `self._kwargs`.
        self._kwargs: dict[str, Any] = {"mode": mode, **kwargs}
        if llm_client is not None:
            self._kwargs["llm_client"] = llm_client
        self._forge: FeatureForge | None = None

    @classmethod
    def from_run_context(cls, ctx: CaseExecutionContext) -> MALMASMethod:
        """Build a MALMAS adapter from a resolved case context.

        Forwards the case-resolved ``settings`` and ``mode`` (defaulting to
        the full pipeline when unset) plus the context-owned ``llm_client``
        into ``FeatureForge`` at fit time. This closes the historical gap
        where ``mode`` on the case was silently dropped because it was not
        part of ``MALMASMethod.__init__``'s signature.
        """
        return cls(
            config=ctx.settings,
            mode=ctx.case.mode or "full",
            llm_client=ctx.llm_client,
        )

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
        # Reuse the cached enhanced training frame from the wrapped FeatureForge
        # rather than re-executing generated code via transform(). Matches
        # BaseMethod.fit_transform and avoids wasted sandbox work on fit data.
        assert self._forge is not None
        return self._forge.fit_transform(X_train, y_train)

    @property
    def generated_scripts(self) -> list[str]:
        return self._forge.generated_scripts if self._forge else []

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return self._forge.feature_metadata if self._forge else []

    def get_artifacts(self) -> dict[str, Any]:
        return self._forge.get_artifacts() if self._forge else {}
