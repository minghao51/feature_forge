"""LLM-FE baseline with artifact collection.

Supports two modes:
- single_shot (default): One LLM call generates all features at once.
- iterative: Sequential LLM calls, one feature at a time, with CV-based
  keep/discard decisions after each iteration.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig
from feature_forge.config import Settings, get_settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.sandbox import SandboxedExecutor
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
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            s = settings or get_settings()
            self.llm_client = create_llm_client(s.llm, s.retry)
        self.n_features = n_features
        self.mode = mode
        self.evaluator = evaluator
        eval_cfg = evaluator.config.evaluation if evaluator else None
        self.sandbox = SandboxedExecutor(
            timeout_seconds=eval_cfg.sandbox_timeout_seconds if eval_cfg else 5.0,
            max_memory_mb=eval_cfg.sandbox_max_memory_mb if eval_cfg else 512,
        )
        self._iteration_codes: list[str] = []

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
                kept_features = pd.DataFrame(index=X.index)
                kept_gains: dict[str, float] = {}

                for col in new_features.columns:
                    gain = evaluator.evaluate_feature(
                        X,
                        y,
                        new_features[[col]],
                        baseline_score=baseline_score,
                    )
                    if gain > 0:
                        kept_features[col] = new_features[col].values
                        kept_gains[col] = gain
                        cumulative_cols.append(col)

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

        self._iteration_codes = iteration_codes
        self._artifacts["iterations"] = iterations
        self._artifacts["generated_code"] = "\n\n".join(iteration_codes)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._iteration_codes:
            raise RuntimeError("LLMFEMethod not fitted yet")
        result = X.copy()
        for code in self._iteration_codes:
            try:
                features = self.sandbox.execute(code, result)
                for col in features.columns:
                    if col not in result.columns:
                        result[col] = features[col].values
            except Exception as exc:
                logger.warning("llmfe_transform_step_failed", error=str(exc))
                if self.evaluator and self.evaluator.config.evaluation.fail_on_feature_error:
                    raise
        new_cols = [c for c in result.columns if c not in X.columns]
        return result[new_cols]

    @property
    def generated_scripts(self) -> list[str]:
        return list(self._iteration_codes)

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        iterations = self._artifacts.get("iterations")
        if iterations:
            meta = []
            for it in iterations:
                for col, gain in it.get("gains", {}).items():
                    meta.append(
                        {
                            "name": col,
                            "method": "llmfe",
                            "iteration": it.get("iteration"),
                            "gain": gain,
                            "kept": it.get("kept", False),
                            "code": it.get("generated_code", ""),
                        }
                    )
            return meta
        code = self._artifacts.get("generated_code", "")
        if code:
            return [{"name": "single_shot", "method": "llmfe", "code": code}]
        return []

    @property
    def provenance_records(self) -> list[dict[str, Any]]:
        iterations = self._artifacts.get("iterations")
        if not iterations:
            return []
        records = []
        for it in iterations:
            for col, gain in it.get("gains", {}).items():
                records.append(
                    {
                        "feature_name": col,
                        "source_method": "llmfe",
                        "iteration_index": it.get("iteration"),
                        "generated_code": it.get("generated_code", ""),
                        "cv_gain": gain,
                    }
                )
        return records

    async def _call_llm(self, prompt: str) -> str:
        response = await self.llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.content.strip()
