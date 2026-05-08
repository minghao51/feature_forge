"""Sklearn-compatible API for MALMAS feature engineering.

Usage:
    >>> from feature_forge.api import MALMASFeatureEngineer
    >>> fe = MALMASFeatureEngineer()
    >>> fe.fit(X_train, y_train)
    >>> X_test_enhanced = fe.transform(X_test)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog
from sklearn.base import BaseEstimator, TransformerMixin

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter
from feature_forge.config import Settings, get_settings
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.llm.base import LLMClient
from feature_forge.llm.providers.deepseek import DeepSeekProvider
from feature_forge.pipeline.ablations import (
    NoMemoryPipeline,
    NoRouterPipeline,
    SingleAgentPipeline,
)
from feature_forge.pipeline.iterative import IterativePipeline
from feature_forge.utils import run_coro_sync

logger = structlog.get_logger()


class MALMASFeatureEngineer(BaseEstimator, TransformerMixin, ArtifactExporter):
    """Sklearn-compatible feature engineering transformer.

    Runs the full MALMAS iterative pipeline during fit() and
    applies generated feature code during transform().

    Implements ArtifactExporter for unified artifact access.

    Parameters:
        config: feature_forge Settings instance.
        llm_client: LLM client for generation.
        mode: Pipeline mode — 'full', 'no_memory', 'no_router', or agent name.
        artifact_config: Configuration for artifact storage.
    """

    def __init__(
        self,
        config: Settings | None = None,
        llm_client: LLMClient | None = None,
        mode: str = "full",
        artifact_config: ArtifactConfig | None = None,
    ) -> None:
        self.config = config or get_settings()
        self.llm_client = llm_client or self._default_llm_client()
        self.mode = mode
        self.selected_features: list[str] = []
        self.feature_codes: list[str] = []
        self.sandbox = SandboxedExecutor(
            timeout_seconds=self.config.evaluation.sandbox_timeout_seconds,
            max_memory_mb=self.config.evaluation.sandbox_max_memory_mb,
        )
        self.pipeline_result: dict[str, Any] | None = None
        self.transform_failures: list[dict[str, str]] = []
        ArtifactExporter.__init__(self, artifact_config=artifact_config)

    def _default_llm_client(self) -> LLMClient:
        """Create default LLM client from settings."""
        cfg = self.config or get_settings()
        return DeepSeekProvider(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key.get_secret_value() if cfg.llm.api_key else None,
            base_url=cfg.llm.base_url,
        )

    def _get_pipeline(self) -> IterativePipeline:
        """Get the appropriate pipeline based on mode."""
        if self.mode == "no_memory":
            return NoMemoryPipeline(self.config, self.llm_client)
        if self.mode == "no_router":
            return NoRouterPipeline(self.config, self.llm_client)
        if self.mode in (
            "unary",
            "cross_compositional",
            "aggregation",
            "temporal",
            "local_transform",
            "local_pattern",
        ):
            return SingleAgentPipeline(self.mode, self.config, self.llm_client)
        return IterativePipeline(self.config, self.llm_client)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MALMASFeatureEngineer:
        """Run iterative feature engineering.

        Args:
            X: Training features.
            y: Training target.

        Returns:
            Self.
        """
        self.pipeline_result = run_coro_sync(self.async_fit(X, y))
        return self

    async def async_fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Async variant of fit() for notebook/service environments."""
        pipeline = self._get_pipeline()
        self.pipeline_result = await pipeline.run(X, y)
        self.selected_features = self.pipeline_result["selected_features"]
        self.feature_codes = self.pipeline_result.get("feature_codes", [])
        return self.pipeline_result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply generated features to new data.

        Re-executes cached feature generation code in the sandbox.

        Args:
            X: Input features.

        Returns:
            DataFrame with original + generated features.
        """
        X_out = X.copy()
        self.transform_failures = []
        for code in self.feature_codes:
            if not code:
                continue
            try:
                features = self.sandbox.execute(code, X_out)
                for col in features.columns:
                    if col not in X_out.columns:
                        X_out[col] = features[col].values
            except Exception as exc:
                error_msg = str(exc)
                self.transform_failures.append({"code": code[:200], "error": error_msg})
                logger.warning("transform_feature_generation_failed", error=error_msg)
                if self.config.evaluation.fail_on_feature_error:
                    raise
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and return enhanced training data."""
        self.fit(X, y)
        return self.pipeline_result["X_train_enhanced"] if self.pipeline_result else X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return output feature names."""
        if input_features is None:
            input_features = []
        return list(input_features) + self.selected_features

    @property
    def generated_scripts(self) -> list[str]:
        """Return all generated feature code blocks."""
        return list(self.feature_codes)

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        """Aggregate feature specs across all rounds."""
        if not self.pipeline_result:
            return []
        all_specs: list[dict[str, Any]] = []
        for ra in self.pipeline_result.get("round_artifacts", []):
            all_specs.extend(ra.get("specs", []))
        return all_specs

    @property
    def provenance_records(self) -> list[dict[str, Any]]:
        """Build structured provenance records for each selected feature.

        Returns:
            List of ProvenanceRecord-compatible dicts.
        """
        if not self.pipeline_result:
            return []
        records: list[dict[str, Any]] = []
        for ra in self.pipeline_result.get("round_artifacts", []):
            round_idx = ra.get("round", 0)
            agents = ra.get("agents", [])
            gains = ra.get("gains", {})
            code = ra.get("generated_code", "")
            specs = ra.get("specs", [])
            for spec in specs:
                if isinstance(spec, dict):
                    feat_name = spec.get("name", "")
                    agent = spec.get("agent", agents[0] if agents else None)
                    gain = gains.get(feat_name) if isinstance(gains, dict) else None
                    records.append(
                        {
                            "feature_name": feat_name,
                            "source_method": "malmas",
                            "source_agent": agent,
                            "round_index": round_idx,
                            "iteration_index": None,
                            "generated_code": code,
                            "cv_gain": gain,
                        }
                    )
        return records

    def get_artifacts(self) -> dict[str, Any]:
        """Flatten round artifacts with round_N_ prefixes."""
        if not self.pipeline_result:
            return {}
        artifacts: dict[str, Any] = {}
        for ra in self.pipeline_result.get("round_artifacts", []):
            prefix = f"round_{ra['round']}_"
            artifacts[f"{prefix}generated_code"] = ra.get("generated_code", "")
            artifacts[f"{prefix}all_features_train"] = ra.get("all_features_train")
            artifacts[f"{prefix}all_features_test"] = ra.get("all_features_test")
            artifacts[f"{prefix}selected_features_train"] = ra.get("selected_features_train")
            artifacts[f"{prefix}selected_features_test"] = ra.get("selected_features_test")
            artifacts[f"{prefix}specs"] = ra.get("specs", [])
            artifacts[f"{prefix}baseline_score"] = ra.get("baseline_score", 0.0)
            artifacts[f"{prefix}gains"] = ra.get("gains", {})
            artifacts[f"{prefix}agent_gains"] = ra.get("agent_gains", {})
            artifacts[f"{prefix}agents"] = ra.get("agents", [])
        artifacts["selected_features"] = self.selected_features
        artifacts["feature_codes"] = self.feature_codes
        artifacts["provenance"] = self.provenance_records
        artifacts["transform_failures"] = self.transform_failures
        return artifacts
