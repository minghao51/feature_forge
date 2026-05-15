"""Sklearn-compatible API for Feature Forge automated feature engineering.

Usage:
    >>> from feature_forge.api import FeatureForge
    >>> fe = FeatureForge()
    >>> fe.fit(X_train, y_train)
    >>> X_test_enhanced = fe.transform(X_test)
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter
from feature_forge.config import Settings, get_settings
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.llm.base import LLMClient
from feature_forge.llm.factory import create_llm_client
from feature_forge.observability.structlog_config import get_logger
from feature_forge.utils import run_coro_sync

logger = get_logger(__name__)

_SINGLE_AGENT_MODES = frozenset(
    {
        "unary",
        "cross_compositional",
        "aggregation",
        "temporal",
        "local_transform",
        "local_pattern",
    }
)


class FeatureForge(BaseEstimator, TransformerMixin, ArtifactExporter):  # type: ignore[misc]
    """Sklearn-compatible automated feature engineering transformer.

    Runs the iterative multi-agent pipeline during fit() and
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
        config: Settings | dict[str, Any] | None = None,
        llm_client: LLMClient | None = None,
        mode: str = "full",
        artifact_config: ArtifactConfig | None = None,
    ) -> None:
        if isinstance(config, dict):
            config = Settings(**config)
        self.config = config or get_settings()
        self.mode = mode
        # Defer LLM client creation: allow None so notebooks can
        # explore the API without an API key.  fit() will raise if
        # no client is available.
        self.llm_client: LLMClient | None = llm_client
        if llm_client is None:
            try:
                self.llm_client = self._default_llm_client()
            except Exception:
                pass  # will be caught at fit() time
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
        """Create default LLM client from settings using the provider factory."""
        cfg = self.config or get_settings()
        return create_llm_client(cfg.llm, retry_config=cfg.retry)

    def _get_pipeline(self) -> Any:
        """Get the appropriate pipeline based on mode.

        Imports are lazy — only the needed pipeline variant is loaded.
        """
        import importlib

        _PIPELINE_SPEC: dict[str, tuple[str, str]] = {
            "full": ("feature_forge.methods.malmas.pipeline.iterative", "IterativePipeline"),
            "no_memory": ("feature_forge.methods.malmas.pipeline.ablations", "NoMemoryPipeline"),
            "no_router": ("feature_forge.methods.malmas.pipeline.ablations", "NoRouterPipeline"),
        }

        spec = _PIPELINE_SPEC.get(self.mode)
        if spec is not None:
            module_path, class_name = spec
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(self.config, self.llm_client, sandbox=self.sandbox)

        if self.mode in _SINGLE_AGENT_MODES:
            from feature_forge.methods.malmas.pipeline.ablations import SingleAgentPipeline

            assert self.llm_client is not None
            return SingleAgentPipeline(
                self.mode,
                self.config,
                self.llm_client,
                sandbox=self.sandbox,
            )

        from feature_forge.methods.malmas.pipeline.iterative import IterativePipeline

        assert self.llm_client is not None
        return IterativePipeline(self.config, self.llm_client, sandbox=self.sandbox)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeatureForge:
        if self.llm_client is None:
            raise RuntimeError(
                "FeatureForge.fit() requires an LLM client. "
                "Set DEEPSEEK_API_KEY (or your provider's key) "
                "or pass llm_client= explicitly."
            )
        fit_t0 = time.perf_counter()
        logger.info(
            "fit_start",
            mode=self.mode,
            model=self.llm_client.model if hasattr(self.llm_client, "model") else "unknown",
            train_shape=X.shape,
            n_rounds=self.config.n_rounds,
        )
        self.pipeline_result = run_coro_sync(self.async_fit(X, y))
        latency_ms = round((time.perf_counter() - fit_t0) * 1000, 1)
        logger.info(
            "fit_complete",
            num_selected_features=len(self.selected_features),
            latency_ms=latency_ms,
        )
        return self

    async def async_fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Async variant of fit() for notebook/service environments."""
        pipeline = self._get_pipeline()
        self.pipeline_result = await pipeline.run(X, y)
        self.selected_features = self.pipeline_result["selected_features"]
        self.feature_codes = self.pipeline_result.get("feature_codes", [])
        return self.pipeline_result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transform_t0 = time.perf_counter()
        logger.info("transform_start", input_shape=X.shape, num_codes=len(self.feature_codes))
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
        latency_ms = round((time.perf_counter() - transform_t0) * 1000, 1)
        logger.info(
            "transform_complete",
            output_shape=X_out.shape,
            latency_ms=latency_ms,
            num_failures=len(self.transform_failures),
        )
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


MALMASFeatureEngineer = FeatureForge
"""Backwards-compatible alias. Use :class:`FeatureForge` in new code."""
