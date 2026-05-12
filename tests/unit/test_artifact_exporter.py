"""Tests for ArtifactExporter ABC and its implementations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.artifacts.base import ArtifactConfig, ArtifactExporter
from feature_forge.llm.base import LLMResponse


class FakeLLM:
    def __init__(self, code: str) -> None:
        self.code = code

    async def complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        return LLMResponse(content=self.code, model="fake")

    async def _do_complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        return await self.complete(messages, temperature, max_tokens, **kwargs)


class ConcreteExporter(ArtifactExporter):
    """Minimal concrete implementation for testing the ABC."""

    @property
    def generated_scripts(self) -> list[str]:
        return ["print('hello')"]

    def get_artifacts(self) -> dict[str, Any]:
        return {"code": "print('hello')", "score": 0.95}


class TestArtifactConfig:
    def test_defaults(self):
        cfg = ArtifactConfig()
        assert cfg.storage_mode == "memory"
        assert cfg.storage_format == "parquet"


class TestArtifactExporterABC:
    def test_concrete_exporter_generated_scripts(self):
        exporter = ConcreteExporter()
        assert exporter.generated_scripts == ["print('hello')"]

    def test_concrete_exporter_get_artifacts(self):
        exporter = ConcreteExporter()
        artifacts = exporter.get_artifacts()
        assert "code" in artifacts
        assert "score" in artifacts

    def test_intermediate_dataframes_filters_df(self):
        class DFExporter(ArtifactExporter):
            @property
            def generated_scripts(self) -> list[str]:
                return []

            def get_artifacts(self) -> dict[str, Any]:
                return {
                    "df1": pd.DataFrame({"a": [1]}),
                    "score": 0.5,
                    "df2": pd.DataFrame({"b": [2]}),
                }

        exporter = DFExporter()
        dfs = exporter.intermediate_dataframes
        assert "df1" in dfs
        assert "df2" in dfs
        assert "score" not in dfs

    def test_feature_metadata_default_empty(self):
        exporter = ConcreteExporter()
        assert exporter.feature_metadata == []

    def test_log_artifacts_delegates_to_tracker(self):
        exporter = ConcreteExporter()

        class MockTracker:
            def __init__(self):
                self.logged = None

            def log_artifacts_dict(self, artifacts, prefix=""):
                self.logged = (artifacts, prefix)

        tracker = MockTracker()
        exporter.log_artifacts(tracker, prefix="test_")
        assert tracker.logged is not None
        assert tracker.logged[1] == "test_"


class TestBaselineArtifacts:
    def test_baseline_has_artifact_config(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        llm = FakeLLM("def generate_features(df): return df")
        baseline = LLMFEBaseline(llm_client=llm)
        assert isinstance(baseline.artifact_config, ArtifactConfig)

    def test_baseline_get_artifacts_empty_before_fit(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        llm = FakeLLM("def generate_features(df): return df")
        baseline = LLMFEBaseline(llm_client=llm)
        assert baseline.get_artifacts() == {}

    def test_baseline_inherits_artifact_exporter(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        llm = FakeLLM("def generate_features(df): return df")
        baseline = LLMFEBaseline(llm_client=llm)
        assert isinstance(baseline, ArtifactExporter)

    def test_llmfe_single_shot_artifacts(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        code = """
import pandas as pd

def generate_features(df):
    result = pd.DataFrame(index=df.index)
    result['sum_ab'] = df['a'] + df['b']
    return result
"""
        llm = FakeLLM(code)
        baseline = LLMFEBaseline(llm_client=llm, mode="single_shot")
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        baseline.fit(X, y)

        artifacts = baseline.get_artifacts()
        assert "prompt" in artifacts
        assert "raw_response" in artifacts
        assert "generated_code" in artifacts
        assert "sum_ab" in artifacts["generated_code"]

    def test_llmfe_generated_scripts_after_fit(self):
        from feature_forge.baselines.llmfe import LLMFEBaseline

        code = """
import pandas as pd

def generate_features(df):
    return pd.DataFrame({'new': df['a'] * 2}, index=df.index)
"""
        llm = FakeLLM(code)
        baseline = LLMFEBaseline(llm_client=llm)
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        baseline.fit(X, y)

        scripts = baseline.generated_scripts
        assert len(scripts) == 1
        assert "generate_features" in scripts[0]
