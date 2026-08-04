"""LLM-FE baseline with artifact collection.

Supports two modes:
- single_shot (default): One LLM call generates all features at once.
- iterative: Sequential LLM calls, one feature at a time, with CV-based
  keep/discard decisions after each iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig
from feature_forge.config import Settings, get_settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.kit import EvaluationKit
from feature_forge.llm.base import LLMClient
from feature_forge.llm.factory import create_llm_client
from feature_forge.methods.base import BaseMethod
from feature_forge.methods.llmfe.prompts import (
    LLMFEIterativeParams,
    LLMFESingleShotParams,
    get_registry,
)
from feature_forge.observability.structlog_config import get_logger
from feature_forge.utils import run_coro_sync, strip_markdown_fences

if TYPE_CHECKING:
    from feature_forge.experiment.context import CaseExecutionContext

logger = get_logger(__name__)


class LLMFEMethod(BaseMethod):
    """LLM-based feature engineering baseline with artifact tracking.

    Parameters:
        llm_client: LLM client for generation.
        n_features: Number of features to generate (single_shot) or
            iterations (iterative).
        mode: 'single_shot' or 'iterative'.
        evaluator: CVEvaluator for iterative mode. If None, uses default.
        artifact_config: Configuration for artifact storage.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        settings: Settings | None = None,
        n_features: int = 5,
        mode: Literal["single_shot", "iterative"] = "single_shot",
        evaluator: CVEvaluator | None = None,
        artifact_config: ArtifactConfig | None = None,
    ) -> None:
        super().__init__("llmfe", artifact_config=artifact_config)
        self._singleton_fallback_name = "single_shot"
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            s = settings or get_settings()
            self.llm_client = create_llm_client(s.llm, s.retry)
        self.n_features = n_features
        self.mode = mode
        self.evaluator = evaluator
        self.sandbox = EvaluationKit.from_settings(settings).sandbox
        self._iteration_codes: list[str] = []

    @classmethod
    def from_run_context(cls, ctx: CaseExecutionContext) -> LLMFEMethod:
        """Build an LLMFE adapter from a resolved case context.

        Reuses the context-owned ``llm_client`` (instead of self-building one
        from settings), the case-resolved ``settings``/``evaluator``, and the
        case's ``mode`` (defaulting to single_shot when unset).
        """
        mode: Literal["single_shot", "iterative"] = ctx.case.mode or "single_shot"  # type: ignore[assignment]
        return cls(
            llm_client=ctx.llm_client,
            settings=ctx.settings,
            evaluator=ctx.evaluator,
            mode=mode,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> LLMFEMethod:
        run_coro_sync(self.async_fit(X_train, y_train))
        return self

    async def async_fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        if self.mode == "iterative":
            await self._fit_iterative(X_train, y_train)
        else:
            await self._fit_single_shot(X_train, y_train)

    async def _fit_single_shot(self, X: pd.DataFrame, y: pd.Series) -> None:
        template = get_registry().get("single_shot").system
        params = LLMFESingleShotParams(
            columns=", ".join(X.columns),
            task="classification" if y.nunique() <= 10 else "regression",
            n_features=self.n_features,
        )
        prompt = params.render(template)
        raw_response = await self._call_llm(prompt)
        code = strip_markdown_fences(raw_response)
        self._iteration_codes = [code]
        self._artifacts["prompt"] = prompt
        self._artifacts["raw_response"] = raw_response
        self._artifacts["generated_code"] = code

    async def _fit_iterative(self, X: pd.DataFrame, y: pd.Series) -> None:
        evaluator = self.evaluator or CVEvaluator()
        baseline_score = evaluator.evaluate_baseline(X, y)
        self._artifacts["baseline_score"] = baseline_score

        iterations: list[dict[str, Any]] = []
        cumulative_cols: list[str] = []
        iteration_codes: list[str] = []
        task: Literal["classification", "regression"] = (
            "classification" if y.nunique() <= 10 else "regression"
        )
        cols = ", ".join(X.columns)

        for i in range(self.n_features):
            template = get_registry().get("iterative").system
            params = LLMFEIterativeParams(
                columns=cols,
                task=task,
                n_iterations=self.n_features,
                iteration=i + 1,
                existing_features=", ".join(cumulative_cols) if cumulative_cols else "none",
            )
            prompt = params.render(template)
            raw_response = await self._call_llm(prompt)
            code_block = strip_markdown_fences(raw_response)
            iteration_record: dict[str, Any] = {
                "iteration": i,
                "prompt": prompt,
                "raw_response": raw_response,
                "generated_code": code_block,
            }

            try:
                new_features = self.sandbox.execute(code_block, X)
                kept_features, kept_gains = self._evaluate_and_select(
                    X,
                    y,
                    new_features,
                    evaluator,
                    baseline_score,
                    cumulative_cols,
                )

                iteration_record["all_new_features"] = self._storage.store(
                    f"llmfe_iter_{i}_all", new_features
                )
                iteration_record["kept_features"] = self._storage.store(
                    f"llmfe_iter_{i}_kept", kept_features
                )
                iteration_record["gains"] = kept_gains
                iteration_record["kept"] = len(kept_gains) > 0

            except Exception as exc:
                iteration_record["error"] = str(exc)
                iteration_record["kept"] = False
                logger.warning("llmfe_iteration_failed", iteration=i, error=str(exc))

            iteration_codes.append(code_block)
            iterations.append(iteration_record)

        self._finalize_iterative_fit(X, iterations, cumulative_cols, iteration_codes)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform_via_iteration_codes(X)

    @property
    def generated_scripts(self) -> list[str]:
        return list(self._iteration_codes)

    async def _call_llm(self, prompt: str) -> str:
        response = await self.llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.content.strip()
