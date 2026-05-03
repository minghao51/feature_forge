"""CAAFE baseline with unified and fidelity variants.

Variants:
- unified (default): Reimplements CAAFE's iterative prompting using our
  CVEvaluator and SandboxedExecutor for full artifact control.
- fidelity: Wraps the original caafe library for exact reproduction
  of the published behavior.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig
from feature_forge.baselines.base import Baseline
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import EvaluationError
from feature_forge.llm.base import LLMClient
from feature_forge.utils import strip_markdown_fences


class CAAFEBaseline(Baseline):
    """CAAFE baseline for LLM-based feature engineering.

    Parameters:
        llm_client: LLM client (required for unified variant).
        llm_model: Model name string (used by fidelity variant).
        iterations: Number of CAAFE iterations.
        variant: 'unified' or 'fidelity'.
        evaluator: CVEvaluator for unified variant.
        artifact_config: Configuration for artifact storage.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        llm_model: str = "gpt-4",
        iterations: int = 2,
        variant: Literal["unified", "fidelity"] = "unified",
        evaluator: CVEvaluator | None = None,
        artifact_config: ArtifactConfig | None = None,
    ) -> None:
        super().__init__("caafe", artifact_config=artifact_config)
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.iterations = iterations
        self.variant = variant
        self.evaluator = evaluator
        self.sandbox = SandboxedExecutor()
        self._caafe = None
        self._iteration_codes: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> CAAFEBaseline:
        if self.variant == "fidelity":
            self._fit_fidelity(X_train, y_train)
        else:
            import asyncio
            asyncio.run(self._fit_unified(X_train, y_train))
        return self

    def _fit_fidelity(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            import caafe
        except ImportError as exc:
            raise EvaluationError("caafe not installed. Run: uv pip install caafe") from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._caafe = caafe.CAAFEClassifier(
                llm_model=self.llm_model,
                iterations=self.iterations,
            )
            self._caafe.fit_pandas(X_train, y_train)

        self._artifacts["variant"] = "fidelity"
        if hasattr(self._caafe, "code"):
            self._artifacts["generated_code"] = self._caafe.code
            self._iteration_codes = [self._caafe.code]
        if hasattr(self._caafe, "mappings"):
            self._artifacts["categorical_mappings"] = self._caafe.mappings

    async def _fit_unified(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Iterative CAAFE using our CVEvaluator and SandboxedExecutor.

        Each iteration prompts the LLM for one code block, executes it,
        and evaluates each new feature against the *original* baseline.
        Features with positive gain are kept; all code blocks are stored
        for sequential transform execution.
        """
        if self.llm_client is None:
            raise EvaluationError("llm_client required for unified CAAFE variant")

        evaluator = self.evaluator or CVEvaluator()
        baseline_score = evaluator.evaluate_baseline(X_train, y_train)

        description = self._build_dataset_description(X_train)
        self._artifacts["dataset_description"] = description
        self._artifacts["baseline_score"] = baseline_score
        self._artifacts["variant"] = "unified"

        iterations: list[dict[str, Any]] = []
        cumulative_cols: list[str] = []
        iteration_codes: list[str] = []
        feedback_str = ""

        for i in range(self.iterations):
            prompt = self._build_caafe_prompt(
                X_train, description, i, cumulative_cols, feedback_str,
            )
            raw_response = await self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
            code_block = strip_markdown_fences(raw_response.content.strip())

            iteration_record: dict[str, Any] = {
                "iteration": i,
                "prompt": prompt,
                "raw_response": raw_response.content,
                "generated_code": code_block,
            }

            try:
                new_features = self.sandbox.execute(code_block, X_train)
                kept_features = pd.DataFrame(index=X_train.index)
                kept_gains: dict[str, float] = {}

                for col in new_features.columns:
                    gain = evaluator.evaluate_feature(
                        X_train, y_train, new_features[[col]],
                        baseline_score=baseline_score,
                    )
                    if gain > 0:
                        kept_features[col] = new_features[col].values
                        kept_gains[col] = gain
                        cumulative_cols.append(col)

                iteration_record["all_new_features"] = self._storage.store(f"caafe_iter_{i}_all", new_features)
                iteration_record["kept_features"] = self._storage.store(f"caafe_iter_{i}_kept", kept_features)
                iteration_record["gains"] = kept_gains
                iteration_record["kept"] = len(kept_gains) > 0

                if kept_gains:
                    feedback_parts = [
                        f"{col}: gain={g:.4f} ({'kept' if g > 0 else 'discarded'})"
                        for col, g in kept_gains.items()
                    ]
                    feedback_str = "Previous iteration feedback: " + "; ".join(feedback_parts)

            except Exception as exc:
                iteration_record["error"] = str(exc)
                iteration_record["kept"] = False
                feedback_str = f"Previous iteration failed: {exc}"

            iteration_codes.append(code_block)
            iterations.append(iteration_record)

        self._iteration_codes = iteration_codes
        self._artifacts["iterations"] = iterations
        self._artifacts["generated_code"] = "\n\n".join(iteration_codes)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.variant == "fidelity":
            return self._transform_fidelity(X)
        return self._transform_unified(X)

    def _transform_fidelity(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._caafe is None:
            raise EvaluationError("CAAFEBaseline not fitted yet")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._caafe.transform_pandas(X)

    def _transform_unified(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._iteration_codes:
            raise EvaluationError("CAAFEBaseline produced no viable features")
        result = X.copy()
        for code in self._iteration_codes:
            try:
                features = self.sandbox.execute(code, result)
                for col in features.columns:
                    if col not in result.columns:
                        result[col] = features[col].values
            except Exception:
                continue
        new_cols = [c for c in result.columns if c not in X.columns]
        return result[new_cols] if new_cols else result

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
                    meta.append({
                        "name": col,
                        "method": "caafe",
                        "iteration": it.get("iteration"),
                        "gain": gain,
                        "kept": it.get("kept", False),
                        "code": it.get("generated_code", ""),
                    })
            return meta
        code = self._artifacts.get("generated_code", "")
        if code:
            return [{"name": "fidelity", "method": "caafe", "code": code}]
        return []

    @property
    def provenance_records(self) -> list[dict[str, Any]]:
        iterations = self._artifacts.get("iterations")
        if not iterations:
            return []
        records = []
        for it in iterations:
            for col, gain in it.get("gains", {}).items():
                records.append({
                    "feature_name": col,
                    "source_method": "caafe",
                    "iteration_index": it.get("iteration"),
                    "generated_code": it.get("generated_code", ""),
                    "cv_gain": gain,
                })
        return records

    @staticmethod
    def _build_dataset_description(X: pd.DataFrame) -> str:
        lines = [f"Columns ({len(X.columns)}): {', '.join(X.columns)}"]
        lines.append(f"Rows: {len(X)}")
        for col in X.columns:
            dtype = X[col].dtype
            nan_pct = X[col].isna().mean() * 100
            sample = X[col].dropna().head(3).tolist()
            lines.append(f"  {col}: dtype={dtype}, nan%={nan_pct:.1f}, sample={sample}")
        return "\n".join(lines)

    def _build_caafe_prompt(
        self,
        X: pd.DataFrame,
        description: str,
        iteration: int,
        existing_cols: list[str],
        feedback: str,
    ) -> str:
        existing = ", ".join(existing_cols) if existing_cols else "none"
        prompt = (
            f"You are a feature engineering assistant using CAAFE methodology.\n\n"
            f"Dataset description:\n{description}\n\n"
            f"Iteration {iteration + 1}/{self.iterations}.\n"
            f"Already created features: {existing}.\n"
        )
        if feedback:
            prompt += f"\n{feedback}\n"
        prompt += (
            "\nGenerate new features using pandas operations.\n"
            "Write a Python function `generate_features(df)` that returns a DataFrame "
            "with only the new features.\n\n"
            "Output ONLY the code, no explanation."
        )
        return prompt
